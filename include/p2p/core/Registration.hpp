
#ifndef COMMON_REGISTRATION_H1
#define COMMON_REGISTRATION_H1

#pragma once

#include "VoxelHashMap.hpp"

namespace p2p
{
    using Vector3dVector = std::vector<V3D>;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;
    Sophus::SE3 RegisterPoint(const std::vector<V3D> &frame,
                               const VoxelHashMap &voxel_map,
                               const Sophus::SE3 &initial_guess,
                               double max_correspondence_distance,
                               double kernel);

    Sophus::SE3 RegisterPlane(const std::vector<V3D> &frame,
                               const VoxelHashMap &voxel_map,
                               const Sophus::SE3 &initial_guess,
                               double max_correspondence_distance,
                               double kernel, bool save_nn = false);
} // namespace p2p

#endif