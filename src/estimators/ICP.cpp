#include "Estimator_ICP.hpp"

#include "p2p/core/Deskew.hpp"
#include "p2p/core/Preprocessing.hpp"
#include "p2p/core/Registration.hpp"

using namespace p2p;

void ICP::LocalMap(PointCloudXYZI::Ptr &map_points) const
{
    const std::vector<V3D_4> &eigen_cloud = local_map_.Pointcloud();
    Eigen2PCL(map_points, eigen_cloud);
};

Sophus::SE3 ICP::GetPredictionModel() const
{
    Sophus::SE3 pred = Sophus::SE3();
    const size_t N = poses_.size();
    if (N < 2)
        return pred;
    return poses_[N - 2].inverse() * poses_[N - 1];
}

bool ICP::HasMoved()
{
    if (poses_.empty())
    {
        has_moved = false;
    }
    else
    {
        const double motion = (poses_.front().inverse() * poses_.back()).translation().norm();
        has_moved = motion > 5.0 * config_.min_motion_th;
    }
    return has_moved;
}

double ICP::GetAdaptiveThreshold()
{
    if (!HasMoved())
    {
        return config_.initial_threshold;
    }
    return adaptive_threshold_.ComputeThreshold();
}

double ComputeDisplacement(const Sophus::SE3 &corrected_distortion, double max_range)
{
    const double theta = Eigen::AngleAxisd(corrected_distortion.rotation_matrix()).angle();
    const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
    const double delta_trans = corrected_distortion.translation().norm();
    return delta_trans + delta_rot;
}

ICP::Vector3dVectorTuple ICP::Voxelize2(PointCloudXYZI::Ptr &frame, bool deskew_end) const
{
    const auto voxel_size = config_.voxel_size;
    const size_t N = frame->size();
    double first_point_time = frame->points[0].time;
    double last_point_time = frame->points.back().time;
    std::cout<<"first_point_time:"<<first_point_time<<", last_point_time:"<<last_point_time<<std::endl;
    double mid_ = 0;
    auto delta_pose = GetPredictionModel().log();
    auto delta_pose_dt = delta_pose / (last_point_time - first_point_time);
    if(deskew_end)
    {
        std::cout<<"deskew_end"<<std::endl;
        mid_ = (last_point_time - first_point_time); //end scan
    }
    else
    {
        std::cout<<"deskew_start"<<std::endl;
        mid_ = 0; 
    }

    tbb::parallel_for(size_t(0), N, [&](size_t i){
            const auto motion = Sophus::SE3::exp((frame->points[i].time - first_point_time - mid_) * delta_pose_dt);

            V3D_4 P_i(frame->points[i].x, frame->points[i].y, frame->points[i].z);
            V3D_4 P_compensate = motion * P_i;
            frame->points[i].x = P_compensate(0);
            frame->points[i].y = P_compensate(1);
            frame->points[i].z = P_compensate(2); 
            
            double displacement = ComputeDisplacement(motion, P_i.norm());
            frame->points[i].intensity = displacement;
        });

    const auto &frame_downsample = p2p::VoxelDownsample(frame, voxel_size * 0.5);
    const auto &source = p2p::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}

ICP::Vector3dVectorTuple ICP::Voxelize(PointCloudXYZI::Ptr &frame, bool deskew, bool sort) const
{
    const auto voxel_size = config_.voxel_size;

    if (deskew)
    {
        std::cout << "Cons vel model deskew...voxel_size:"<< voxel_size << std::endl;
        const size_t N = frame->size();

        double first_point_time = frame->points[0].time;
        double last_point_time = frame->points.back().time;
        std::cout<<"first_point_time:"<<first_point_time<<", last_point_time:"<<last_point_time<<std::endl;
        double mid_ = (last_point_time - first_point_time) / 2.; // this works just fine for lieksa
        //mid_ = 0;                                                // worked for evo 
        //mid_ = (last_point_time - first_point_time); //end scan

        // Divide scan into 5 equal segments and take the 3.5 / 5.0 time of the last segment
        //mid_ = (last_point_time - first_point_time) * 3.5 / 5.0; //for evo data 


        // std::cout<<"mid_:"<<mid_<<std::endl;

        auto delta_pose = GetPredictionModel().log();
        auto delta_pose_dt = delta_pose / (last_point_time - first_point_time);

        // std::vector<double> timestamps;
        // timestamps.reserve(N);
        // for (const auto &p : frame->points)
        //     timestamps.push_back(p.time);

        // // Find min and max times
        // auto [min_it, max_it] = std::minmax_element(timestamps.begin(), timestamps.end());
        // double tmin = *min_it;
        // double tmax = *max_it;
        // double range = std::max(1e-12, tmax - tmin); // prevent divide by zero
        // constexpr double mid_{0.5};
        // // Normalize to [0, 1]
        // for (auto &t : timestamps)
        //     t = (t - tmin) / range;

        tbb::parallel_for(size_t(0), N, [&](size_t i)
                          {
            //decoupled rotation and translation: Sophus::SE3d T_j = Sophus::interpolate(T_begin, T_end, scale);


            //coupled rotation and translation
            const auto motion = Sophus::SE3::exp((frame->points[i].time - first_point_time - mid_) * delta_pose_dt);
            //const auto motion = Sophus::SE3::exp((timestamps[i] - mid_) * delta_pose);

            V3D_4 P_i(frame->points[i].x, frame->points[i].y, frame->points[i].z);
            V3D_4 P_compensate = motion * P_i;
            
            //P_compensate(2) += (frame->points[i].time - first_point_time) * 10;

            frame->points[i].x = P_compensate(0);
            frame->points[i].y = P_compensate(1);
            frame->points[i].z = P_compensate(2); 
            
            // double displacement = ComputeDisplacement(motion, P_i.norm());
            // frame->points[i].intensity = displacement;
        });

        // if(sort)
        // {
        //     tbb::parallel_sort(frame->points.begin(), frame->points.end(),
        //            [](const auto &a, const auto &b) {
        //                return a.intensity < b.intensity; //smaller displacement points first
        //            });
        // }
        
        const auto &frame_downsample = p2p::VoxelDownsample(frame, voxel_size * 0.5);
        const auto &source = p2p::VoxelDownsample(frame_downsample, voxel_size * 1.5);
        return {source, frame_downsample};
    }
    const auto &frame_downsample = p2p::VoxelDownsample(frame, voxel_size * 0.5);
    const auto &source = p2p::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}

bool ICP::update(const std::vector<V3D_4> &source, const p2p::VoxelHashMap &local_map_, bool p2p_, bool save_nn)
{
    // Compute initial_guess for ICP
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();
    const auto prediction = GetPredictionModel();
    const auto initial_guess = last_pose * prediction; // const velocity model

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos);
    imu_init_guess = initial_guess; // const vel model

    Sophus::SE3 new_pose = p2p::RegisterPoint(source, local_map_, imu_init_guess,
                                            3.0 * sigma, // max_correspondence_distance
                                            sigma / 3.0); // kernel th

    // if (p2p_)
    // {
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    poses_.push_back(new_pose);
    //}
    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();

    return true;
}

Sophus::SE3 ICP::update_refine(const std::vector<V3D_4> &source, const p2p::VoxelHashMap &local_map_)
{
    // Compute initial_guess for ICP
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();
    const auto prediction = GetPredictionModel();
    const auto initial_guess = last_pose * prediction; // const velocity model

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos); // this is from the prev iteration
    // imu_init_guess = initial_guess; // const vel model

    Sophus::SE3 new_pose = p2p::RegisterPoint(source, local_map_, imu_init_guess,
                                              3.0 * sigma,  // max_correspondence_distance
                                              sigma / 3.0); // kernel th

    // if (!poses_.empty())
    // {
    //     poses_.back() = new_pose; // change the last pose with the one updated from ALS
    // }
    
    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();

    return new_pose;
}

bool ICP::update(const std::vector<V3D_4> &source, const PointCloudXYZI::Ptr &map, const pcl::KdTreeFLANN<PointType>::Ptr &tree)
{
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos);
    imu_init_guess = last_pose; // const vel model - take this from mls estimate

    Sophus::SE3 new_pose = p2p::RegisterPlane(source, map, tree, imu_init_guess,
                                              3.0 * sigma,  // max_correspondence_distance
                                              sigma / 3.0); // kernel th

    if (!poses_.empty())
    {
        poses_.back() = new_pose; // change the last pose with the one updated from ALS
    }

    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();

    return true;
}

bool ICP::update(const Sophus::SE3 &gnss, const std::vector<V3D_4> &source, const p2p::VoxelHashMap &local_map_)
{
    // Compute initial_guess for ICP
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();
    const auto prediction = GetPredictionModel();
    const auto initial_guess = last_pose * prediction; // const velocity model

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos);
    imu_init_guess = initial_guess; // const vel model

    Sophus::SE3 new_pose = p2p::RegisterPointAndGNSS(gnss, source, local_map_, imu_init_guess,
                                                     3.0 * sigma,  // max_correspondence_distance
                                                     sigma / 3.0); // kernel th

    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    poses_.push_back(new_pose);

    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();
    return true;
}

bool ICP::update_tightlyCoupled(const std::vector<V3D_4> &source, const p2p::VoxelHashMap &local_map_,
                                const PointCloudXYZI::Ptr &map, const pcl::KdTreeFLANN<PointType>::Ptr &tree)

{
    // Compute initial_guess for ICP
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();
    const auto prediction = GetPredictionModel();
    const auto initial_guess = last_pose * prediction; // const velocity model

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos);
    imu_init_guess = initial_guess; // const vel model

    Sophus::SE3 new_pose = p2p::RegisterTightly(source,
                                                local_map_,
                                                map, tree,
                                                imu_init_guess,
                                                3.0 * sigma, // max_correspondence_distance
                                                sigma / 3.0);

    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold_.UpdateModelDeviation(model_deviation);
    // poses_.push_back(new_pose);

    if (!poses_.empty())
    {
        poses_.back() = new_pose; // change the last pose with the one updated from ALS
    }

    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();

    return true;
}