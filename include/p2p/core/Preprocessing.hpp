
#ifndef COMMON_PREPROCESSING_H1
#define COMMON_PREPROCESSING_H1

#pragma once

#include "p2p/core/Deskew.hpp"

namespace p2p
{
    /// Voxelize point cloud keeping the original coordinates
    std::vector<V3D_4> VoxelDownsample(const std::vector<V3D_4> &frame, double voxel_size);

    std::vector<V3D_4> VoxelDownsample(const PointCloudXYZI::Ptr &frame, double voxel_size);

} // namespace p2p

#endif