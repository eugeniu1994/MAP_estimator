
#include "ALS.hpp"
#include <regex>
#include <liblas/liblas.hpp>

#include <pcl/registration/gicp.h>

#include <pcl/filters/voxel_grid_covariance.h> 

using namespace gnss;
using namespace ekf;

FixedSizeQueue<std::string> historyTiles(4);

struct FileDistance
{
    std::string filename;
    double distance;
    bool operator<(const FileDistance &other) const
    {
        return distance < other.distance;
    }
};

ALS_Handler::ALS_Handler(const std::string &folder_root_, bool donwsample_, int closest_N_files_, double leaf_size_)
{
    folder_root = folder_root_;
    downsampled_ALS = donwsample_;
    closest_N_files = closest_N_files_;
    als_to_mls = Sophus::SE3();
    gps_origin_ENU = Zero3d;
    R_to_mls = Eye3d;
    leaf_size = leaf_size_;
    prev_mls_pos = Zero3d;
    downSizeFilterSurf.setLeafSize(leaf_size, leaf_size, leaf_size);
    als_cloud.reset(new PointCloudXYZI());

    localKdTree_map_als.reset(new pcl::KdTreeFLANN<PointType>());

    if (!als_manager_setup)
        setupALS_Manager();
}

void ALS_Handler::setupALS_Manager()
{
    std::cout << "\033[31mStart reading ALS files\033[0m" << std::endl;
    all_las_files.reset(new pcl::PointCloud<pcl::PointXYZ>());
    boost::filesystem::path folderRoot(folder_root);
    if (!boost::filesystem::exists(folderRoot))
    {
        std::cerr << "Error: The directory " << folder_root << " does not exist." << std::endl;
        als_manager_setup = false;
        return;
    }

    std::regex filePattern(R"(tile_x_(\d+)_y_(\d+)\.las)");
    std::smatch match;
    all_las_files->clear();

    std::vector<boost::filesystem::path> files_;

    for (auto &entry : boost::filesystem::directory_iterator(folder_root))
    {
        files_.push_back(entry.path());
    }

    std::stable_sort(files_.begin(), files_.end());
    int x_min = 0, y_min = 0;
    for (const auto &entry : files_)
    {
        std::string filename = entry.filename().string();
        // std::cout << "\n check " << filename << " ";
        if (std::regex_match(filename, match, filePattern))
        {
            x_min = std::stoi(match[1].str());
            y_min = std::stoi(match[2].str());

            pcl::PointXYZ center_point_file;
            center_point_file.x = x_min + boxSize / 2; // center_x;
            center_point_file.y = y_min + boxSize / 2; // center_y;
            center_point_file.z = 0;                   // z is not used in this example

            all_las_files->push_back(center_point_file);
        }
        else
        {
            std::cerr << "Error: " << filename << " REGEX FAILED" << std::endl;
        }
    }

    std::cout << "setupALS_Manager ALS files:" << all_las_files->size() << std::endl;
    ALS_manager.setInputCloud(all_las_files);
    als_manager_setup = true;

    std::cout << "\033[32mSetup ALS files successfully\033[0m" << std::endl;
}

bool ALS_Handler::init(const Sophus::SE3 &known_als2mls)
{
    bool rv = true;
    std::cout << "\033[31mALS init from known T\033[0m" << std::endl;
    std::cout<<"known_als2mls:\n"<<known_als2mls.matrix()<<std::endl;
    R_to_mls = known_als2mls.so3().matrix();
    als_to_mls = known_als2mls;
    refine_als = true;
    initted_ = true;
    shift_initted_ = true;
    if (!als_manager_setup)
        setupALS_Manager();

    return rv;
}

bool ALS_Handler::init(const V3D &gps_origin_ENU_, const M3D &init_R_2_mls, const PointCloudXYZI::Ptr &mls_cloud_full)
{
    bool rv = false;
    std::cout << "\033[31mALS init\033[0m" << std::endl;
    gps_origin_ENU = gps_origin_ENU_;
    R_to_mls = init_R_2_mls;

    if (!als_manager_setup)
        setupALS_Manager();

    double N = gps_origin_ENU[1], E = gps_origin_ENU[0]; // north and east
    std::regex filePattern(R"(tile_x_(\d+)_y_(\d+)\.las)");
    std::smatch match;
    std::multiset<FileDistance> closestFiles;

    // Get the closest files for initialization
    double distance_sq = 0, max_allowed_distance = 5.0 * boxSize;
    double max_allowed_distance_sq = max_allowed_distance * max_allowed_distance;
    std::string filename;
    int c = boxSize / 2;
    for (const auto &center_point : all_las_files->points)
    {
        distance_sq = std::pow(center_point.x - E, 2) + std::pow(center_point.y - N, 2);
        std::cout << "distance_:" << sqrt(distance_sq) << ", als E:" << center_point.x << ", N:" << center_point.y << std::endl;
        if (distance_sq < max_allowed_distance_sq)
        {
            filename = "tile_x_" + std::to_string(int(center_point.x - c)) + "_y_" + std::to_string(int(center_point.y - c)) + ".las";
            if (closestFiles.size() < closest_N_files)
            {
                closestFiles.insert({filename, distance_sq});
            }
            else if (distance_sq < closestFiles.rbegin()->distance)
            {
                closestFiles.erase(std::prev(closestFiles.end()));
                closestFiles.insert({filename, distance_sq});
            }
        }
    }

    // Load the points
    if (closestFiles.empty())
    {
        std::cout << "No matching files found, closestFiles:" << closestFiles.size() << std::endl;
        std::cout << "gps_origin_ENU:" << gps_origin_ENU.transpose() << std::endl;
        throw std::runtime_error("init - Cannot find the right init file in las/laz for current position");
        // std::cout<<"Cannot find the right init file in las/laz for current position"<<std::endl;
        found_ALS = false;
        return rv;
    }
    else
    {
        found_ALS = true;
        std::cout << "\n Closest files:" << std::endl;
        for (const auto &file : closestFiles)
        {
            std::cout << file.filename << " with distance " << std::sqrt(file.distance) << std::endl;
            AddPoints_from_file(file.filename);
        }
    }

    if (!refine_als)
    {
        std::cout << "\033[31mStart initialization registration ALS2MLS...\033[0m" << std::endl;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

        PointCloudXYZI::Ptr mls_cloud(new PointCloudXYZI());
        float _10_m_above_sensor_hight = 10;
        for (const auto &point : mls_cloud_full->points)
        {
            if (point.z < _10_m_above_sensor_hight) // skip the tree tops
                mls_cloud->push_back(point);
        }

        getCloud(als_cloud);

        std::cout << "mls:" << mls_cloud->size() << ", als:" << als_cloud->size() << std::endl;
        std::cout << "\033[31mStart registration...\033[0m" << std::endl;
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        
        if (als_cloud->size() < mls_cloud->size()) // als to mls
        {
            icp.setInputSource(als_cloud); // als
            icp.setInputTarget(mls_cloud); // mls
        }
        else // mls to als
        {
            icp.setInputSource(mls_cloud); // mls
            icp.setInputTarget(als_cloud); // als
        }

        icp.setMaximumIterations(200);
        icp.setMaxCorrespondenceDistance(2.); // m
        pcl::PointCloud<PointType> Final;
        icp.align(Final);

        if (icp.hasConverged())
        {
            std::cout << "ICP converged." << std::endl
                      << "The score is " << icp.getFitnessScore() << std::endl;
            Eigen::Matrix4f transformation = icp.getFinalTransformation();

            Eigen::Matrix4d T = transformation.cast<double>();
            Sophus::SE3 refinement_T(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
            std::cout << "refinement_T:" << refinement_T.log().transpose() << std::endl;

            auto init_T = als_to_mls;
            std::cout << "prev als_to_mls:" << als_to_mls.log().transpose() << std::endl;
            if (als_cloud->size() < mls_cloud->size()) // als to mls
            {
                als_to_mls = refinement_T * init_T;
            }
            else // mls to als
            {
                als_to_mls = refinement_T.inverse() * init_T;
            }

            std::cout << "curr als_to_mls:" << als_to_mls.log().transpose() << std::endl;
        }
        else
        {
            std::cout << "\033[31mICP did not converge...Handle this\033[0m" << std::endl;
            // TODO - handle this
        }

        end = std::chrono::system_clock::now();
        std::chrono::duration<float> elapsed_seconds = end - start;
        float duration_milliseconds = elapsed_seconds.count() * 1000; // to get milliseconds
        std::cout << "ALS 2 MLS refinement " << elapsed_seconds.count() << " (s), and " << duration_milliseconds << " (ms)" << std::endl;

        als_cloud->clear();

        R_to_mls = als_to_mls.so3().matrix(); // refined rotation from als to mls

        refine_als = true;
        initted_ = false;

        // std::cout << "Refined Initialization: " << als_to_mls.log().transpose() << std::endl;
        // call init one more time to get into account refined ALS2MLS transformation
        this->init(gps_origin_ENU_, R_to_mls, mls_cloud_full);
        std::cout << "\033[32mALS to MLS initializaion converged successfully\033[0m" << std::endl;
    }
    rv = true;

    return rv;
}

void ALS_Handler::getCloud(PointCloudXYZI::Ptr &in_)
{
    *in_ = *als_cloud;
}

void ALS_Handler::AddPoints_from_file(const std::string &filename)
{
    std::cout << "\nAttempt AddPoints_from_file----------------------------" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    std::istream *ifs = liblas::Open(folder_root + filename, std::ios::in | std::ios::binary);
    if (!ifs)
    {
        std::cerr << "Cannot open " << folder_root + filename << " for read.  Exiting..." << std::endl;
        throw std::invalid_argument("Cannot open the las/laz file");
    }

    if (!ifs->good())
        throw std::runtime_error("Reading went wrong!");

    liblas::ReaderFactory readerFactory;
    liblas::Reader reader = readerFactory.CreateWithStream(*ifs);
    liblas::Header const &header = reader.GetHeader();

    // std::cout << "Compressed: " << (header.Compressed() == true) ? "true" : "false";
    // std::cout << " Signature: " << header.GetFileSignature() << '\n';
    // std::cout << "Start reading the data" << std::endl;
    // std::cout << "header.GetOffsetZ():" << header.GetOffsetZ() << std::endl;
    // if (header.GetPointRecordsCount() < 1000){
    //    ALS_has_points = false;
    //    std::cout << "This scan has only " << header.GetPointRecordsCount() << " points" << std::endl;
    //    return;
    //}

    if (!shift_initted_)
    {
        liblas::Reader mean_reader = readerFactory.CreateWithStream(*ifs);
        // min_points_per_patch = header.GetPointRecordsCount()/5;
        min_points_per_patch = 100; // 1000;

        std::cout << "min_points_per_patch:" << min_points_per_patch << ", header.GetPointRecordsCount():" << header.GetPointRecordsCount() << std::endl;

        Eigen::Vector2d ref_gps = gps_origin_ENU.head<2>();
        double dist = 40 * 40;        // take 40m radius

        std::vector<double> min_z;
        while (mean_reader.ReadNextPoint())
        {
            liblas::Point const &p = mean_reader.GetPoint();
            std::cout << "p.x:" << p.GetX() << " p.y:" << p.GetY() << " p.z:" << p.GetZ() << std::endl;
            std::cout << "ref_gps:" << ref_gps.transpose() << std::endl;
            auto d = (Eigen::Vector2d(p.GetX(), p.GetY()) - ref_gps).squaredNorm();
            std::cout << "distance:" << sqrt(d) << std::endl;
            break;
        }
        while (mean_reader.ReadNextPoint()) // collect all z in 10 m radius ball
        {
            liblas::Point const &p = mean_reader.GetPoint();
            if ((Eigen::Vector2d(p.GetX(), p.GetY()) - ref_gps).squaredNorm() < dist)
            {
                min_z.push_back(p.GetZ());
            }
        }
        int vec_size = min_z.size();
        if (vec_size > 10)
        {
            size_t numElementsToFilter = std::min(vec_size, 100); // take at max 100 points
            std::cout << "numElementsToFilter:" << numElementsToFilter << ", vec_size:" << vec_size << std::endl;
            std::partial_sort(min_z.begin(), min_z.begin() + numElementsToFilter, min_z.end()); // sort by z axis
            std::vector<double> indices, values;
            for (int i = 0; i < numElementsToFilter; i++)
            {
                std::cout << ", " << min_z[i];
                indices.push_back(i);
                values.push_back(min_z[i]);
            }

            double median = calculateMedian(values);

            std::cout << "Median: " << median << std::endl;
            std::cout << "init gps_origin_ENU:" << gps_origin_ENU.transpose() << std::endl;

            gps_origin_ENU[2] = median; // corrent the als cloud hight

            // init guess from GNSS
            als_to_mls = Sophus::SE3(R_to_mls, -R_to_mls * gps_origin_ENU);

            std::cout << "shift_initted_:  als_to_mls-> " << als_to_mls.log().transpose() << std::endl;
            shift_initted_ = true;
        }
        else
        {
            std::cout << "min_z vec_size:" << min_z.size() << std::endl;
            // TODO - handle when not enough points
            throw std::runtime_error("There a no points for ALS Z-axis initialization");
        }
    }

    int patch_points = header.GetPointRecordsCount();
    std::cout << "ALS patch points:" << patch_points << std::endl;
    if (downsampled_ALS)
    {
        if (patch_points < min_points_per_patch)
        {
            std::cout << "min_points_per_patch:" << min_points_per_patch << ",  this has only " << patch_points << std::endl;
            // do not add this patch
            return;
        }
    }

    PointType point;
    point.x = 0;
    point.y = 0;
    point.z = 0;
    point.intensity = key;
    point.time = 0;
    key++;

    PointCloudXYZI::Ptr original_als_cloud(new PointCloudXYZI());
    original_als_cloud->resize(header.GetPointRecordsCount());
    size_t index = 0;
    while (reader.ReadNextPoint())
    {
        liblas::Point const &p = reader.GetPoint();
        // V3D cloudPoint = R_to_mls * (V3D(p.GetX(), p.GetY(), p.GetZ()) - gps_origin_ENU);

        V3D cloudPoint = als_to_mls * V3D(p.GetX(), p.GetY(), p.GetZ()); // align to als
        point.x = cloudPoint.x();
        point.y = cloudPoint.y();
        point.z = cloudPoint.z();

        // point.intensity = random_number; // for debug purposes

        original_als_cloud->points[index] = point;
        index++;
    }

    PointCloudXYZI::Ptr downsampled_als_cloud(new PointCloudXYZI());

    if (downsampled_ALS)
    {
        downSizeFilterSurf.setInputCloud(original_als_cloud);
        downSizeFilterSurf.filter(*downsampled_als_cloud);
    }
    else
    {
        *downsampled_als_cloud = *original_als_cloud;
    }


    if (!initted_)
    {
        *als_cloud = *downsampled_als_cloud;
        initted_ = true;
    }
    else
    {
        *als_cloud += *downsampled_als_cloud;
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    float duration_milliseconds = elapsed_seconds.count() * 1000; // to get milliseconds
    std::cout << "ALS AddPoints took " << elapsed_seconds.count() << " (s), and " << duration_milliseconds << " (ms)" << std::endl;
}

void ALS_Handler::RemovePointsFarFromLocation(const V3D &mls_position)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());

    double x_min = mls_position.x() - 75;
    double y_min = mls_position.y() - 75;
    double z_min = mls_position.z() - 75;
    double x_max = mls_position.x() + 75;
    double y_max = mls_position.y() + 75;
    double z_max = mls_position.z() + 75;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1));
    cropBoxFilter.setNegative(false);

    cropBoxFilter.setInputCloud(als_cloud);
    cropBoxFilter.filter(*tmpSurf);

    *als_cloud = *tmpSurf;

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    float duration_milliseconds = elapsed_seconds.count() * 1000; // to get milliseconds
    std::cout << "ALS RemovePointsFarFromLocation took " << elapsed_seconds.count() << " (s), and " << duration_milliseconds << " (ms)" << std::endl;
}

bool ALS_Handler::Update(const Sophus::SE3 &mls_pose)
{
    std::cout<<"Call update: motion:"<<(prev_mls_pos - mls_pose.translation()).norm()<<std::endl;
    std::cout<<"boxSize:"<<boxSize<<std::endl;

    if ((first_time) || ((prev_mls_pos - mls_pose.translation()).norm() > boxSize / 2))
    {
        first_time = false;
        std::cout << "====================== ALS Update =======================" << std::endl;
        prev_mls_pos = mls_pose.translation();
        RemovePointsFarFromLocation(prev_mls_pos);

        Sophus::SE3 mls_in_als = als_to_mls.inverse() * mls_pose;
        auto _mls_enu = mls_in_als.translation();

        std::regex filePattern(R"(tile_x_(\d+)_y_(\d+)\.las)");
        std::smatch match;
        std::multiset<FileDistance> closestFiles;

        pcl::PointXYZ curr_mls_position_enu;
        curr_mls_position_enu.x = _mls_enu[0];
        curr_mls_position_enu.y = _mls_enu[1];
        curr_mls_position_enu.z = 0;

        std::vector<int> pointIdxNKNSearch(closest_N_files);
        std::vector<float> pointNKNSquaredDistance(closest_N_files);

        if (ALS_manager.nearestKSearch(curr_mls_position_enu, closest_N_files, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            double th_sq = (2 * boxSize) * (2 * boxSize) + 2 * boxSize; // this is ugly change it
            std::string closest_file;
            int c = boxSize / 2;
            for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            {
                std::cout << "distance:" << sqrt(pointNKNSquaredDistance[i]) << std::endl;
                if (pointNKNSquaredDistance[i] <= th_sq)
                {
                    curr_mls_position_enu = (*all_las_files)[pointIdxNKNSearch[i]];
                    closest_file = "tile_x_" + std::to_string(int(curr_mls_position_enu.x - c)) + "_y_" + std::to_string(int(curr_mls_position_enu.y - c)) + ".las";

                    // check if closest_file exists
                    if (boost::filesystem::exists(folder_root + closest_file))
                    {
                        if (!historyTiles.contains(closest_file))
                        {
                            AddPoints_from_file(closest_file);
                            historyTiles.push(closest_file); // Mark file as loaded and add to the queue
                        }
                    }
                    else
                    {
                        std::cout << "closest_file:" << std::to_string(i) << " : " << closest_file << std::endl;
                        throw std::runtime_error("Update - Cannot find the right init file in las/laz");
                    }
                }
            }
        }

        if(als_cloud->size() > 5)
        {
            localKdTree_map_als->setInputCloud(als_cloud);
        }
        else 
        {
            std::cerr<<"ALS: there is no ALS data available..."<<std::endl;
            return false;
        }
        

        return true; // there was an update
    }

    return false;
}

