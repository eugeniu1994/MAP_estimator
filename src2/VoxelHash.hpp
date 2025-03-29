#ifndef COMMON_VOXELMAP_H1
#define COMMON_VOXELMAP_H1

#pragma once

#include "DataHandler_vux.hpp"
#include <tsl/robin_map.h>

// TargetFrame   PointType VUX_PointType
// const typename pcl::PointCloud<PointType>::Ptr& reference_map,
// const typename pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
// const typename pcl::PointCloud<VUX_PointType>::Ptr& scan,

using Voxel = Eigen::Vector3i;

Voxel PointToVoxel(const V3D &point, const double &voxel_size);

static int plane_min_points = 7;// 5;

struct VoxelHashMap
{
    struct VoxelBlock
    {
        std::vector<V3D> points;
        int level;
        V3D normal = V3D::Zero();
        bool has_normal = false;
        double curvature = 0;

        inline void AddPoint(const V3D &point)
        {
            points.push_back(point);
        }

        inline bool ComputePlaneNormal(const double &planarity_th)
        {
            if (points.size() < plane_min_points)
                return false;

            const int size = points.size();
            if (points.size() >= plane_min_points)
            {
                if (true)
                {
                    Eigen::MatrixXd A(size, 3);
                    Eigen::VectorXd b(size);
                    A.setZero();
                    b.setOnes();
                    b *= -1.0;

                    // A/Dx + B/Dy + C/Dz + 1 = 0
                    for (int j = 0; j < size; j++)
                    {
                        A(j, 0) = points[j].x();
                        A(j, 1) = points[j].y();
                        A(j, 2) = points[j].z();
                    }

                    normal = A.colPivHouseholderQr().solve(b);

                    double d = 1.0 / normal.norm();
                    normal.normalize();

                    for (int j = 0; j < size; j++)
                    {
                        double tmp = fabs(normal(0) * points[j].x() + normal(1) * points[j].y() + normal(2) * points[j].z() + d);
                        if (curvature < tmp)
                            curvature = tmp;

                        if (fabs(normal(0) * points[j].x() + normal(1) * points[j].y() + normal(2) * points[j].z() + d) > planarity_th)
                        {
                            return false;
                        }
                    }

                    has_normal = true;
                }
                else
                {
                    V3D centroid = V3D::Zero();
                    for (const auto &p : points)
                    {
                        centroid += p;
                    }
                    centroid /= points.size();

                    M3D covariance = M3D::Zero();
                    for (const auto &p : points)
                    {
                        V3D centered = p - centroid;
                        covariance += centered * centered.transpose();
                    }

                    Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
                    normal = solver.eigenvectors().col(0);

                    V3D eigenvalues = solver.eigenvalues();
                    double lambda0 = eigenvalues(0); // Smallest eigenvalue
                    double lambda1 = eigenvalues(1);
                    double lambda2 = eigenvalues(2); // Largest eigenvalue

                    // Planarity measure: lower values mean flatter structure
                    // curvature = lambda0 / (lambda0 + lambda1 + lambda2); // Curvature: λ0 / (λ0 + λ1 + λ2)

                    // elongation = lambda1 / lambda2;
                    // isotropy = lambda0 / lambda2;

                    // if Elongation ≈ 1, the surface is more square or round.
                    //  If Elongation ≪ 1, the surface is stretched (long plane).
                    //  Isotropy checks if the spread is similar in all directions:
                    //      Isotropy ≈ 1 → Point cloud is more spherical.
                    //      Isotropy ≪ 1 → Long plane or line structure

                    // if(curvature > planarity_th)
                    //     return false;

                    for (const auto &p : points) // point-to-plane distance
                    {
                        double distance = std::abs(normal.dot(p - centroid));
                        if (curvature < distance)
                            curvature = distance;

                        if (distance > planarity_th)
                            return false;
                    }
                    has_normal = true;
                }
            }

            return has_normal;
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
        std::cout << "\033[32mBuild VoxelHash with voxel_size:\033[0m" << voxel_size_ << "\033[32m and max points:\033[0m" << max_points_per_voxel_ << std::endl;
    }

    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }

    void Build(const pcl::PointCloud<PointType>::Ptr &reference_map);
    int Pointcloud(pcl::PointCloud<PointType>::Ptr &cloud_) const;
    int Pointcloud_and_Normals(pcl::PointCloud<PointType>::Ptr &cloud_, pcl::PointCloud<pcl::Normal>::Ptr &normals) const;

    // void RemovePointsFarFromLocation(const V3D &origin);

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif