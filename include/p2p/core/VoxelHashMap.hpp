#ifndef COMMON_VOXELMAP_H1
#define COMMON_VOXELMAP_H1

#pragma once

#include <tsl/robin_map.h>

#include "../../utils.h"
#include "icp_util.hpp"

namespace p2p
{
    struct VoxelHashMap
    {
        struct VoxelBlock
        {
            std::vector<V3D> points;
            int num_points_;

            inline void AddPoint(const V3D &point)
            {
                if (points.size() < num_points_)
                {
                    points.push_back(point);
                }
            }
        };

        struct VoxelHash
        {
            size_t operator()(const Voxel &voxel) const
            {
                const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
            }
        };

        explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
            : voxel_size_(voxel_size),
              max_distance_(max_distance),
              max_points_per_voxel_(max_points_per_voxel)
        {
            inv_voxel_size_ = 1.0 / voxel_size;
            std::cout << "\033[32mBuild VoxelHashMap with voxel_size:\033[0m" << voxel_size_ << "\033[32m and max points:\033[0m" << max_points_per_voxel_ << std::endl;
        }

        Vector3dVectorTuple GetPointCorrespondences(const Vector3dVector &points,
                                                    double max_correspondance_distance) const;

        Vector3dNormalTuple GetPlaneCorrespondences(const Vector3dVector &points,
                                                    double max_correspondance_distance) const;

        inline void Clear() { map_.clear(); }
        inline bool Empty() const { return map_.empty(); }

        void Update(const std::vector<V3D> &points, const V3D &origin);
        void Update(const std::vector<V3D> &points, const Sophus::SE3 &pose);

        void AddPoints(const std::vector<V3D> &points);
        void RemovePointsFarFromLocation(const V3D &origin);
        std::vector<V3D> Pointcloud() const;

        double voxel_size_, inv_voxel_size_;
        double max_distance_;
        int max_points_per_voxel_;
        tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

} // namespace p2p

#endif