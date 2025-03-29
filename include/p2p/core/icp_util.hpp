#ifndef COMMON_ICPUTIL_H1
#define COMMON_ICPUTIL_H1

#include "../../utils.h"

using Vector3dVector = std::vector<V3D>;
using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
using Voxel = Eigen::Vector3i;

using Vector3dNormal = std::vector<Eigen::Matrix<double, 4, 1>>;
using Vector3dNormalTuple = std::tuple<Vector3dVector, Vector3dNormal>;

Voxel PointToVoxel(const Eigen::Vector3d &point, const double &inv_voxel_size);

#endif