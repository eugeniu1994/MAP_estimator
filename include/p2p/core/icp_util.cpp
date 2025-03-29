#include "icp_util.hpp"

Voxel PointToVoxel(const Eigen::Vector3d &point, const double &inv_voxel_size)
{
    return Voxel(static_cast<int>(std::floor(point.x() * inv_voxel_size)),
                 static_cast<int>(std::floor(point.y() * inv_voxel_size)),
                 static_cast<int>(std::floor(point.z() * inv_voxel_size)));
}