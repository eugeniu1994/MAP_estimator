#include "Estimator_ICP.hpp"

#include "p2p/core/Deskew.hpp"
#include "p2p/core/Preprocessing.hpp"
#include "p2p/core/Registration.hpp"

using namespace p2p;

void ICP::LocalMap(PointCloudXYZI::Ptr &map_points) const
{
    const std::vector<V3D> &eigen_cloud = local_map_.Pointcloud();
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

ICP::Vector3dVectorTuple ICP::Voxelize(PointCloudXYZI::Ptr &frame, bool deskew ) const
{
    const auto voxel_size = config_.voxel_size;

    if(deskew)
    {
        std::cout<<"Cons vel model deskew..."<<std::endl;
        auto delta_pose = GetPredictionModel().log();

        double first_point_time = frame->points[0].time;
        const size_t N = frame->size();
        
        //PointCloudXYZI::Ptr frame_deskew(new PointCloudXYZI(*frame));
        
        std::vector<double> timestamps;
        timestamps.reserve(N);
        for (const auto &p : frame->points)
            timestamps.push_back(p.time);

        // Find min and max times
        auto [min_it, max_it] = std::minmax_element(timestamps.begin(), timestamps.end());
        double tmin = *min_it;
        double tmax = *max_it;
        double range = std::max(1e-12, tmax - tmin); // prevent divide by zero

        // Normalize to [0, 1]
        for (auto &t : timestamps)
            t = (t - tmin) / range;
        
        // std::cout<<"timestamps:"<<timestamps.size()<<std::endl;
        // if(!timestamps.empty())
        // {
        //     std::cout<<"Times: ("<<timestamps[0]<<" - "<<timestamps[timestamps.size()-1]<<")"<<std::endl;
        // }

        constexpr double mid_pose_timestamp{0.5};
        tbb::parallel_for(size_t(0), N, [&](size_t i)
        {
            //const auto motion = Sophus::SE3::exp((frame->points[i].time - first_point_time) * delta_pose);
            const auto motion = Sophus::SE3::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);

            V3D P_i(frame->points[i].x, frame->points[i].y, frame->points[i].z);
            V3D P_compensate = motion * P_i;
            
            frame->points[i].x = P_compensate(0);
            frame->points[i].y = P_compensate(1);
            frame->points[i].z = P_compensate(2);
        });

        const auto &frame_downsample = p2p::VoxelDownsample(frame, voxel_size * 0.5);
        const auto &source = p2p::VoxelDownsample(frame_downsample, voxel_size * 1.5);
        return {source, frame_downsample};
    }
    const auto &frame_downsample = p2p::VoxelDownsample(frame, voxel_size * 0.5);
    const auto &source = p2p::VoxelDownsample(frame_downsample, voxel_size * 1.5);
    return {source, frame_downsample};
}

bool ICP::update(const std::vector<V3D> &source, const p2p::VoxelHashMap &local_map_, bool p2p_, bool save_nn)
{
    // Compute initial_guess for ICP
    const auto last_pose = !poses_.empty() ? poses_.back() : Sophus::SE3();
    const auto prediction = GetPredictionModel();
    const auto initial_guess = last_pose * prediction; // const velocity model

    // Get motion prediction and adaptive_threshold
    const double sigma = GetAdaptiveThreshold();

    Sophus::SE3 imu_init_guess(x_.rot, x_.pos);
    imu_init_guess = initial_guess;  //const vel model

    Sophus::SE3 new_pose =
        (p2p_ == true) ? p2p::RegisterPoint(source, local_map_, imu_init_guess,
                                            3.0 * sigma, // max_correspondence_distance
                                            sigma / 3.0) // kernel th
                       : p2p::RegisterPlane(source, local_map_, imu_init_guess,
                                            3.0 * sigma,           // max_correspondence_distance
                                            sigma / 3.0, save_nn); // kernel th

    // if (save_nn)
    // {
    //     const auto &[src, tgt] = local_map_.GetPointCorrespondences(source, 1);//1m threshold 
    //     ALS_tgt = tgt;
    //     //for (auto &point : ALS_tgt) //for debug purposes
    //     //    point[2] += 50;
        
    // }

    if (p2p_)
    {
        const auto model_deviation = initial_guess.inverse() * new_pose;
        adaptive_threshold_.UpdateModelDeviation(model_deviation);
        poses_.push_back(new_pose);
    }
    x_.rot = new_pose.so3();
    x_.pos = new_pose.translation();

    return true;
}
