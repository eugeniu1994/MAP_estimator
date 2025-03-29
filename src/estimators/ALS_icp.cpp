#include "ALS.hpp"
#include <regex>
#include <liblas/liblas.hpp>
#include "p2p/core/Preprocessing.hpp"

// #include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

using namespace gnss;

FixedSizeQueue<std::string> historyTiles(7);

struct FileDistance
{
    std::string filename;
    double distance;
    bool operator<(const FileDistance &other) const
    {
        return distance < other.distance;
    }
};

ALS_Handler::ALS_Handler(const std::string &folder_root_, bool donwsample_, int closest_N_files_, double leaf_size_): config_(), 
          local_map_(leaf_size_, config_.max_range, 10)
{
    folder_root = folder_root_;
    downsampled_ALS = donwsample_;
    closest_N_files = closest_N_files_;
    als_to_mls = Sophus::SE3();
    gps_origin_ENU = Zero3d;
    R_to_mls = Eye3d;
    leaf_size = leaf_size_;
    prev_mls_pos = Zero3d;
    curr_mls_pos = Zero3d;
    downSizeFilterSurf.setLeafSize(leaf_size, leaf_size, leaf_size);
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

    std::cout << "ALS files:" << all_las_files->size() << std::endl;
    ALS_manager.setInputCloud(all_las_files);
    als_manager_setup = true;

    std::cout << "\033[32mSetup ALS files successfully\033[0m" << std::endl;
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
        throw std::runtime_error("Cannot find the right init file in las/laz for current position");
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
        //refine_als = true; // these 2 lines will avoid initialization refinement
        //return;

        std::cout << "\033[31mStart initialization registration ALS2MLS...\033[0m" << std::endl;

        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();

        PointCloudXYZI::Ptr mls_cloud(new PointCloudXYZI());
        PointCloudXYZI::Ptr als_cloud(new PointCloudXYZI());
        getCloud(als_cloud);
        float _10_m_above_sensor_hight = 10;
        for (const auto &point : mls_cloud_full->points)
        {
            if (point.z < _10_m_above_sensor_hight) // skip the tree tops
                mls_cloud->push_back(point);
        }

        std::cout << "mls:" << mls_cloud->size() << ", als:" << als_cloud->size() << std::endl;
        std::cout << "\033[31mStart registration...\033[0m" << std::endl;
        pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        // #include <pcl/registration/ndt.h>
        // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        // ndt.setTransformationEpsilon (0.01);
        //  Setting maximum step size for More-Thuente line search.
        // ndt.setStepSize (0.1);
        // Setting Resolution of NDT grid structure (VoxelGridCovariance).
        // ndt.setResolution (1.0);
        // Eigen::Matrix4f init_guess
        // ndt.align (*output_cloud, init_guess);

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

        local_map_.Clear();

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

void ALS_Handler::AddPoints_from_file(const std::string &filename)
{
    std::cout<<"start AddPoints_from_file"<<std::endl;
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
        Eigen::Vector2d ref_gps = gps_origin_ENU.head<2>();
        double dist = 10 * 10; // 10m radius
        std::vector<double> min_z;
        while (mean_reader.ReadNextPoint()) // collect all z in 10 m radius ball
        {
            liblas::Point const &p = mean_reader.GetPoint();
            if ((Eigen::Vector2d(p.GetX(), p.GetY()) - ref_gps).squaredNorm() < dist)
            {
                min_z.push_back(p.GetZ());
            }
        }
        int vec_size = min_z.size();

        min_points_per_patch = header.GetPointRecordsCount()/5;
        std::cout<<"min_points_per_patch:"<<min_points_per_patch<<", header.GetPointRecordsCount():"<<header.GetPointRecordsCount()<<std::endl;
        
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

            LineModel bestZ = ransacFitLine(indices, values, 200, .5);
            std::cout << "\n bestZ:" << bestZ.b << std::endl;

            double mean = calculateMean(values);
            double median = calculateMedian(values);

            std::cout << "Mean: " << mean << std::endl;
            std::cout << "Median: " << median << std::endl;

            std::cout << "init gps_origin_ENU:" << gps_origin_ENU.transpose() << std::endl;

            // gps_origin_ENU[2] = bestZ.b; // corrent the als cloud hight
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
        }
    }

    int patch_points = header.GetPointRecordsCount();
    if(downsampled_ALS)
    {
        if(patch_points < min_points_per_patch)
        {
            std::cout<<"min_points_per_patch:"<<min_points_per_patch<<",  this has only "<<patch_points<<std::endl;
            //do not add this patch
            return;
        }
    }

    PointType point;
    point.x = 0;
    point.y = 0;
    point.z = 0;

    std::cout<<"There are "<<patch_points<<" points"<<std::endl;
    PointCloudXYZI::Ptr original_als_cloud(new PointCloudXYZI());
    original_als_cloud->resize(patch_points);
    size_t index = 0;
    while (reader.ReadNextPoint())
    {
        const liblas::Point &p = reader.GetPoint();
        V3D cloudPoint = als_to_mls * V3D(p.GetX(), p.GetY(), p.GetZ()); 
        point.x = cloudPoint.x();
        point.y = cloudPoint.y();
        point.z = cloudPoint.z();

        original_als_cloud->points[index] = point;
        index++;
    }
    PointCloudXYZI::Ptr downsampled_als_cloud(new PointCloudXYZI());
    std::vector<V3D> eigen_als_cloud;
    if(downsampled_ALS)
    {
        downSizeFilterSurf.setInputCloud(original_als_cloud);
        downSizeFilterSurf.filter(*downsampled_als_cloud);
        PCL2EIGEN(downsampled_als_cloud, eigen_als_cloud);
    }else{
        PCL2EIGEN(original_als_cloud, eigen_als_cloud);
    }

    std::cout<<"start local map update"<<std::endl; 
    local_map_.Update(eigen_als_cloud, curr_mls_pos);
    
    //if (downsampled_ALS)
    //{
        //const auto &frame_downsample = p2p::VoxelDownsample(original_als_cloud, leaf_size);
        //local_map_.Update(frame_downsample, curr_mls_pos);
    //}
    //else
    //{
        //local_map_.Update(original_als_cloud, curr_mls_pos);
    //}

    initted_ = true;
    
    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds = end - start;
    float duration_milliseconds = elapsed_seconds.count() * 1000; // to get milliseconds
    std::cout << "ALS AddPoints took " << elapsed_seconds.count() << " (s), and " << duration_milliseconds << " (ms)" << std::endl;
}

void ALS_Handler::getCloud(PointCloudXYZI::Ptr &in_)
{
    const std::vector<V3D> &eigen_cloud = local_map_.Pointcloud();
    Eigen2PCL(in_, eigen_cloud);
}

void ALS_Handler::Update(const Sophus::SE3 &mls_pose)
{
    curr_mls_pos = mls_pose.translation();
    if ((prev_mls_pos - curr_mls_pos).norm() > boxSize / 2)
    {
        std::cout << "====================== ALS Update =======================" << std::endl;
        prev_mls_pos = curr_mls_pos;

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

        std::cout<<"Start nearestKSearch"<<std::endl;
        if (ALS_manager.nearestKSearch(curr_mls_position_enu, closest_N_files, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            double th_sq = (2 * boxSize) * (2 * boxSize) + boxSize; // this is ugly change it
            std::string closest_file;
            int c = boxSize / 2;
            std::cout<<"nearestKSearch found "<<pointIdxNKNSearch.size()<<" points"<<std::endl;
            for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
            {
                std::cout << "distance:" << sqrt(pointNKNSquaredDistance[i]) << std::endl;
                if (pointNKNSquaredDistance[i] <= th_sq)
                {
                    curr_mls_position_enu = (*all_las_files)[pointIdxNKNSearch[i]];
                    closest_file = "tile_x_" + std::to_string(int(curr_mls_position_enu.x - c)) + "_y_" + std::to_string(int(curr_mls_position_enu.y - c)) + ".las";
                    std::cout<<"closest_file:"<<closest_file<<std::endl;
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
                        throw std::runtime_error("Cannot find the right init file in las/laz");
                    }
                }
            }
        }
    }
}