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

ICP::Vector3dVectorTuple ICP::Voxelize(const PointCloudXYZI::Ptr &frame) const
{
    const auto voxel_size = config_.voxel_size;
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
    // imu_init_guess = initial_guess;

    Sophus::SE3 new_pose =
        (p2p_ == true) ? p2p::RegisterPoint(source, local_map_, imu_init_guess,
                                            3.0 * sigma, // max_correspondence_distance
                                            sigma / 3.0) // kernel th
                       : p2p::RegisterPlane(source, local_map_, imu_init_guess,
                                            3.0 * sigma,           // max_correspondence_distance
                                            sigma / 3.0, save_nn); // kernel th

    if (save_nn)
    {
        const auto &[src, tgt] = local_map_.GetPointCorrespondences(source, 1);//1m threshold 
        ALS_tgt = tgt;
        //for (auto &point : ALS_tgt) //for debug purposes
        //    point[2] += 50;
        
    }

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
