#pragma once

#include "DataHandler_vux.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_set>
#include <omp.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/ISAM2.h>

// #include <gtsam/geometry/Rot3.h>
// #include <gtsam/navigation/GPSFactor.h>
// #include <gtsam/navigation/ImuFactor.h>
// #include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Unit3.h> // For plane normals
#include <gtsam/base/numericalDerivative.h>

#define BA_NEIGH (10.0) // (5.0) // use min 5 points for nearest neighbours

namespace Eigen
{
    const int state_size_ = 6;
    using Matrix6d = Eigen::Matrix<double, state_size_, state_size_>;
    using Matrix3_6d = Eigen::Matrix<double, 3, state_size_>;
    using Vector6d = Eigen::Matrix<double, state_size_, 1>;
} // namespace Eigen

namespace registration
{
    struct P2Plane_global
    {
        P2Plane_global(const V3D &curr_point_, const V3D &plane_unit_norm_,
                       double d_)
            : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), d(d_) {}

        template <typename T>
        bool operator()(const T *q, const T *t, T *residual) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            // Transform raw scanner point to world frame
            Eigen::Matrix<T, 3, 1> point_world = q_w * curr_point.template cast<T>() + t_w;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = norm.dot(point_world) + T(d);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double d_)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_global, 1, 4, 3>(
                new P2Plane_global(curr_point_, plane_unit_norm_, d_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double d;
    };

    struct P2Plane_local
    {
        P2Plane_local(const V3D &src_point,
                      const V3D &target_point,
                      const V3D &normal)
            : src_point(src_point), target_point(target_point), normal(normal) {}

        template <typename T>
        bool operator()(const T *const q, const T *const t, T *residuals) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            Eigen::Matrix<T, 3, 1> transformed_point = q_w * src_point.template cast<T>() + t_w;
            Eigen::Matrix<T, 3, 1> target = target_point.template cast<T>();
            Eigen::Matrix<T, 3, 1> normal_ = normal.template cast<T>();

            residuals[0] = (transformed_point - target).dot(normal_);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &src_point,
                                           const V3D &target_point,
                                           const V3D &normal)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_local, 1, 4, 3>(
                new P2Plane_local(src_point, target_point, normal)));
        }

        V3D src_point;
        V3D target_point;
        V3D normal;
    };

    struct P2Point
    {
        P2Point(const V3D &curr_point_, const V3D &closest_point_)
            : curr_point(curr_point_), closest_point(closest_point_) {}

        template <typename T>
        bool operator()(const T *const q, const T *const t, T *residual) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            // Transform raw scanner point to world frame
            Eigen::Matrix<T, 3, 1> point_world = q_w * curr_point.template cast<T>() + t_w;

            // Compute residual as difference to the closest map point
            residual[0] = point_world.x() - T(closest_point.x());
            residual[1] = point_world.y() - T(closest_point.y());
            residual[2] = point_world.z() - T(closest_point.z());

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &closest_point_)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Point, 3, 4, 3>(
                new P2Point(curr_point_, closest_point_)));
        }

        V3D curr_point;
        V3D closest_point;
    };

    struct LidarPlaneNormFactor_extrinsics
    {

        LidarPlaneNormFactor_extrinsics(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                        double d_, const Sophus::SE3 &fixed_pose_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                                                     d(d_), fixed_pose(fixed_pose_) {}

        template <typename T>
        bool operator()(const T *q, const T *t, T *residual) const
        {
            // Convert extrinsic transformation (Scanner to some frame)
            Eigen::Quaternion<T> q_extrinsic(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_extrinsic(t[0], t[1], t[2]);

            // Convert Fixed pose to appropriate type
            Eigen::Matrix<T, 3, 3> R_fixed = fixed_pose.rotation_matrix().template cast<T>();
            Eigen::Matrix<T, 3, 1> t_fixed = fixed_pose.translation().template cast<T>();

            // Transform raw scanner point to desired frame with extrinsics
            Eigen::Matrix<T, 3, 1> point_in_frame = q_extrinsic * curr_point.template cast<T>() + t_extrinsic;

            // Georeference point
            Eigen::Matrix<T, 3, 1> point_world = R_fixed * point_in_frame + t_fixed;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = norm.dot(point_world) + T(d);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double d_, const Sophus::SE3 &fixed_pose_)
        {
            return (new ceres::AutoDiffCostFunction<
                    LidarPlaneNormFactor_extrinsics, 1, 4, 3>(
                new LidarPlaneNormFactor_extrinsics(curr_point_, plane_unit_norm_, d_, fixed_pose_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double d;
        Sophus::SE3 fixed_pose;
    };
};

using namespace registration;

double square(const double &x)
{
    return x * x;
}

double scan2map_GN_omp(pcl::PointCloud<PointType>::Ptr &src,
                       const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                       const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                       Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                       const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                       bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0,
                       bool p2plane_with_mesh_point = false, bool weighted_p2_plane = false)
{
    using namespace registration;

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };

    Eigen::Matrix6d JTJ_global = Eigen::Matrix6d::Zero();
    Eigen::Vector6d JTr_global = Eigen::Vector6d::Zero();
    double cost_total = 0.;

    // std::vector<V3D> tangents;
    // computeTangents(src, tangents, threshold_nn); // this should be called outside once

    // std::cout << "tangents:" << tangents.size() << std::endl;

    std::cout << "Run GN omp ..." << std::endl;
    if (weighted_p2_plane)
    {
        std::cout << "Perform weighted_p2_plane" << std::endl;
    }
    else if (p2plane_with_mesh_point)
    {
        std::cout << "Perform p2plane_with_mesh_point" << std::endl;
    }
    else if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    int num_points = src->points.size();
    std::cout << "num_points:" << num_points << ", reference_localMap_cloud:" << reference_localMap_cloud->size() << std::endl;
    if (reference_localMap_cloud->size() == 0)
    {
        throw std::runtime_error("reference_localMap_cloud not init - no points");
    }

#pragma omp parallel
    {
        Eigen::Matrix6d JTJ_private = Eigen::Matrix6d::Zero();
        Eigen::Vector6d JTr_private = Eigen::Vector6d::Zero();
        double cost_private = 0.;
#pragma omp for schedule(dynamic)
        for (int i = 0; i < num_points; i++)
        {
            const auto &raw_point = src->points[i];

            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
            V3D p_transformed = q * p_src + t;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            if (weighted_p2_plane)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                            matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                            matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double d = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + d) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {

                            {

                                // for (const auto& obs : observations) {
                                //     Eigen::Vector3d p = obs.source_point;
                                //     Eigen::Vector3d t = obs.source_tangent.normalized();  // ensure unit tangent
                                //     Eigen::Vector3d q = obs.target_point;
                                //     Eigen::Vector3d n = obs.target_normal.normalized();

                                //     Eigen::Vector3d p_trans = pose_estimate * p;
                                //     Eigen::Vector3d residual_vec = p_trans - q;

                                //     double point_plane_residual = n.dot(residual_vec);

                                //     double weight = 1.0;

                                //     if (!options.use_tangent_alignment) {
                                //         // Use angle between tangent and normal as a weight
                                //         weight = (t.cross(n)).norm(); // sin(angle)
                                //     }

                                //     // Jacobian of point-to-plane residual wrt se3 pose (6D)
                                //     Eigen::Matrix<double, 1, 6> J;
                                //     Eigen::Matrix3d skew = Sophus::SO3d::hat(p_trans);
                                //     J.block<1, 3>(0, 0) = -n.transpose() * skew;  // rotation part
                                //     J.block<1, 3>(0, 3) = n.transpose();          // translation part

                                //     double weighted_residual = weight * point_plane_residual;
                                //     Eigen::Matrix<double, 1, 6> weighted_J = weight * J;

                                //     H += weighted_J.transpose() * weighted_J;
                                //     b += weighted_J.transpose() * weighted_residual;

                                //     total_error += weighted_residual * weighted_residual;

                                //     if (options.use_tangent_alignment) {
                                //         double tangent_normal_residual = t.dot(n); // Should be 0 if orthogonal

                                //         // Add it as a soft cost term (like regularization)
                                //         double tangent_weight = 0.01;  // tunable

                                //         // No pose Jacobian for this — it’s independent of pose
                                //         H += tangent_weight * Eigen::Matrix<double, 6, 6>::Identity();
                                //         b += tangent_weight * tangent_normal_residual * Eigen::Matrix<double, 6, 1>::Zero(); // no derivative wrt pose
                                //     }
                                // }

                                // Eigen::Matrix<double, 6, 1> dx = -H.ldlt().solve(b);
                                // if (dx.norm() < 1e-6) {
                                //     break;
                                // }

                                // pose_estimate = Sophus::SE3d::exp(dx) * pose_estimate;
                            }

                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Matrix<double, 1, 6> J_r;
                            double tangent_penalty = 1.0;
                            //// Residual: n^T (T·p - q) * ||t × n||
                            // double point_to_plane = n.dot(p_to_q);
                            // double tangent_penalty = (t.cross(n)).norm();
                            // residuals[i] = point_to_plane * tangent_penalty;

                            // if (tangents[i].squaredNorm() > 0)
                            // {
                            //     tangent_penalty = (tangents[i].cross(norm)).norm();
                            // }

                            Eigen::Matrix3_6d J_se3;
                            double residual = (p_transformed - target_point).dot(norm);

                            J_se3.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                            J_se3.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                            double w = 1.;                                                    // Weight(residual * residual);

                            J_r = tangent_penalty * norm.transpose() * J_se3;

                            JTJ_private.noalias() += J_r.transpose() * w * J_r;      // 6x6
                            JTr_private.noalias() += J_r.transpose() * w * residual; // 6x1
                            cost_private += w * residual * residual;                 // Always non-negative

                            // Eigen::Vector6d J_r;                               // 6x1
                            // J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            // J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR
                            // // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + d;
                            // double residual = (p_transformed - target_point).dot(norm);
                            // double w = Weight(residual * residual);
                            // JTJ_private.noalias() += J_r * w * J_r.transpose();
                            // JTr_private.noalias() += J_r * w * residual;
                            // cost_private += w * residual * residual; // Always non-negative
                        }
                    }
                }
            }
            else if (p2plane_with_mesh_point)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                            matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                            matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double d = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + d) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            // Eigen::Vector6d J_r;                               // 6x1
                            // J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            // J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                            // // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + d;
                            // double residual = (p_transformed - target_point).dot(norm);

                            // double w = Weight(residual * residual);
                            // JTJ_private.noalias() += J_r * w * J_r.transpose();
                            // JTr_private.noalias() += J_r * w * residual;

                            // cost_private += w * residual * residual; // Always non-negative

                            // // Project point to plane
                            V3D vec = p_transformed - target_point; // p2p
                            double dist = vec.dot(norm);
                            target_point = p_transformed - dist * norm; // projected_point

                            // this is equivalent to p2plane but slow bc:
                            // residual = (p_transformed - p_transformed - dist * norm).dot(norm) = dist -> vec.dot(norm)  but slower

                            Eigen::Matrix3_6d J_r;

                            const V3D residual = p_transformed - target_point;
                            J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                            double w = Weight(residual.squaredNorm());
                            JTJ_private.noalias() += J_r.transpose() * w * J_r;
                            JTr_private.noalias() += J_r.transpose() * w * residual;

                            cost_private += w * residual.squaredNorm();
                        }
                    }
                }
            }
            else if (p2p && p2plane) // both
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                            matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                            matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double d = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + d) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Vector6d J_r;                               // 6x1
                            J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + d;
                            double residual = (p_transformed - target_point).dot(norm);

                            double w = Weight(residual * residual);
                            JTJ_private.noalias() += J_r * w * J_r.transpose();
                            JTr_private.noalias() += J_r * w * residual;

                            cost_private += w * residual * residual; // Always non-negative
                        }
                        else
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Matrix3_6d J_r;

                            const V3D residual = p_transformed - target_point;
                            J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR
                            double w = Weight(residual.squaredNorm());
                            JTJ_private.noalias() += J_r.transpose() * w * J_r;
                            JTr_private.noalias() += J_r.transpose() * w * residual;

                            cost_private += w * residual.squaredNorm();
                        }
                    }
                }
            }
            else if (p2plane)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                            matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                            matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double d = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + d) > .1)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            Eigen::Vector6d J_r;                               // 6x1
                            J_r.block<3, 1>(0, 0) = norm;                      // df/dt
                            J_r.block<3, 1>(3, 0) = p_transformed.cross(norm); // df/dR

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + d;
                            double residual = (p_transformed - target_point).dot(norm);

                            double w = Weight(residual * residual);
                            JTJ_private.noalias() += J_r * w * J_r.transpose();
                            JTr_private.noalias() += J_r * w * residual;

                            cost_private += w * residual * residual; // Always non-negative
                        }
                    }
                }
            }
            else if (p2p) // only p2p
            {
                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
                {
                    if (point_dist[0] < threshold_nn)
                    {
                        V3D target_point(
                            reference_localMap_cloud->points[point_idx[0]].x,
                            reference_localMap_cloud->points[point_idx[0]].y,
                            reference_localMap_cloud->points[point_idx[0]].z);

                        Eigen::Matrix3_6d J_r;

                        const V3D residual = p_transformed - target_point;

                        J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR

                        double w = Weight(residual.squaredNorm());
                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;

                        cost_private += w * residual.squaredNorm();
                    }
                }
            }
        }

#pragma omp critical
        {
            JTJ_global += JTJ_private;
            JTr_global += JTr_private;
            cost_total += cost_private;
        }
    }

    const Eigen::Vector6d dx = JTJ_global.ldlt().solve(-JTr_global);

    const Sophus::SE3 estimation = Sophus::SE3::exp(dx);
    T_icp = estimation * T_icp;

    q = Eigen::Quaterniond(T_icp.so3().matrix());
    t = T_icp.translation();

    return cost_total;
    // return dx.norm();
}

void debug_CloudWithNormals(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud_with_normals,
                            const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub)
{

    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_with_normals, cloud_msg);
    cloud_msg.header.frame_id = "world";

    // --- 2. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.05; // Line width
    normals_marker.color.a = 1.0;  // Full opacity
    double normal_length = 3.;     // 5.;     // Length of normal lines

    for (const auto &point : cloud_with_normals->points)
    {
        geometry_msgs::Point p1, p2;

        // Start of normal (point)
        p1.x = point.x;
        p1.y = point.y;
        p1.z = point.z;
        normals_marker.points.push_back(p1);

        // End of normal (point + normal * length)
        p2.x = point.x + normal_length * point.normal_x;
        p2.y = point.y + normal_length * point.normal_y;
        p2.z = point.z + normal_length * point.normal_z;
        normals_marker.points.push_back(p2);

        std_msgs::ColorRGBA color;
        // if (point.curvature < 2) // seen less than 10 times
        // {
        //     color.r = 1.0; // not seen enough
        //     color.g = 0.0;
        //     color.b = 0.0;
        // }
        // else
        // {
        color.r = 0.0;
        color.g = 1.0; // Green for high curvature
        color.b = 0.0;
        //}
        color.a = 1.0; // Full opacity

        normals_marker.colors.push_back(color); // Color for start point
        normals_marker.colors.push_back(color); // Color for end point
    }

    if (cloud_pub.getNumSubscribers() != 0)
    {
        cloud_pub.publish(cloud_msg);
    }

    if (normals_pub.getNumSubscribers() != 0)
    {
        normals_pub.publish(normals_marker);
    }
}

void debug_CloudWithNormals2(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud_with_normals,
                             const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
                             bool plot_tangents = false,
                             pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_tangents = boost::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>())
{
    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_with_normals, cloud_msg);
    cloud_msg.header.frame_id = "world";

    // --- 2. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.02; // Line width
    normals_marker.color.a = 1.0;  // Full opacity
    double normal_length = 2.;     // 5.;     // Length of normal lines
    double opacity = 1.0;

    // Flip normal to point consistently toward viewpoint
    // V3D point_on_the_plane(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
    // if (norm.dot(T.translation() - point_on_the_plane) < 0)
    // {
    //     norm = -norm;
    // }

    for (auto &point : cloud_with_normals->points)
    {
        geometry_msgs::Point p1, p2;

        auto norm = V3D(point.normal_x, point.normal_y, point.normal_z);
        if (norm.dot(V3D(0, 0, 1)) < 0)
        {
            norm *= -1;
        }

        if (point.curvature > 0)
        {
            norm *= -1;
        }

        point.normal_x = norm[0];
        point.normal_y = norm[1];
        point.normal_z = norm[2];

        // Start of normal (point)
        p1.x = point.x;
        p1.y = point.y;
        p1.z = point.z;
        normals_marker.points.push_back(p1);

        if (point.curvature > 0)
        {
            normal_length = 2.;
        }
        else
        {
            normal_length = 3.;
        }
        // End of normal (point + normal * length)
        p2.x = point.x + normal_length * point.normal_x;
        p2.y = point.y + normal_length * point.normal_y;
        p2.z = point.z + normal_length * point.normal_z;
        normals_marker.points.push_back(p2);

        std_msgs::ColorRGBA color;
        if (point.curvature > 0) // blue seen multiple times
        {
            color.r = 1.;
            color.g = 0.;
            color.b = 0.0;
            opacity = 1.0;
        }
        else if (point.curvature == -1)
        {
            color.r = 0.0;
            color.g = 1.0; // seen once
            color.b = 0.0;
            opacity = .5;
        }

        else if (point.curvature == -2)
        {
            color.r = 1.0;
            color.g = 1.0; // white line
            color.b = 1.0;
            opacity = 1;
        }

        color.a = opacity;

        normals_marker.colors.push_back(color); // Color for start point
        normals_marker.colors.push_back(color); // Color for end point
    }

    if (plot_tangents)
    {
        std::cout << "plot tangents...cloud_with_tangents:" << cloud_with_tangents->size() << std::endl;
        for (auto &point : cloud_with_tangents->points)
        {
            geometry_msgs::Point p1, p2;
            // Start of normal (point)
            p1.x = point.x;
            p1.y = point.y;
            p1.z = point.z;
            normals_marker.points.push_back(p1);

            normal_length = .4; // 1;

            // End of normal (point + normal * length)
            p2.x = point.x + normal_length * point.normal_x;
            p2.y = point.y + normal_length * point.normal_y;
            p2.z = point.z + normal_length * point.normal_z;
            normals_marker.points.push_back(p2);

            std_msgs::ColorRGBA color;
            color.r = 1.0f; // full red
            color.g = 1.0f; // full green
            color.b = 0.0f; // no blue
            color.a = .8;

            normals_marker.colors.push_back(color); // Color for start point
            normals_marker.colors.push_back(color); // Color for end point
        }
    }

    if (cloud_pub.getNumSubscribers() != 0)
    {
        cloud_pub.publish(cloud_msg);
    }

    if (normals_pub.getNumSubscribers() != 0)
    {
        normals_pub.publish(normals_marker);
    }
}

using namespace gtsam;
using gtsam::symbol_shorthand::A; // for anchor
using gtsam::symbol_shorthand::X; // Pose symbols

Pose3 sophusToGtsam(const Sophus::SE3 &pose)
{
    return Pose3(pose.matrix());
}

Sophus::SE3 GtsamToSophus(const Pose3 &pose)
{
    Eigen::Matrix4d T = pose.matrix(); // Get 4x4 matrix
    return Sophus::SE3(T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>());
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extractTranslations(const gtsam::Values &values)
{
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    for (const gtsam::Key &key : values.keys())
    {
        if (values.exists<gtsam::Pose3>(key))
        {
            gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
            gtsam::Point3 t = pose.translation();
            cloud->points.emplace_back(t.x(), t.y(), t.z());
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

void publishBACloud(ros::Publisher &pub, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    pub.publish(msg);
}

namespace custom_factor
{
    // Custom factor for point-to-point constraints
    class PointToPointFactor : public NoiseModelFactor1<Pose3>
    {
    private:
        Point3 measured_point_; // Point in sensor frame
        Point3 target_point_;   // Corresponding point in world frame

        double huber_delta_ = .1;
        // Huber weight calculation
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToPointFactor(Key pose_key,
                           const Point3 &measured_point,
                           const Point3 &target_point,
                           const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                            measured_point_(measured_point),
                                                            target_point_(target_point) {}

        // Vector evaluateError(const Pose3 &pose, boost::optional<Matrix &> H = boost::none) const override
        Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
        {
            // Transform measured point to world frame
            Matrix36 H_point;
            Point3 world_point = pose.transformFrom(measured_point_, H_point); // computes ∂(world_point)/∂pose

            // Calculate error vector
            Vector3 error = world_point - target_point_;

            // auto robust_weight = GM_robust_kernel(error.norm());
            auto robust_weight = huberWeight(error.norm());

            if (H)
            {
                // Jacobian: ∂error/∂pose = ∂p_world/∂pose
                *H = H_point * robust_weight;
            }

            return robust_weight * error;

            // if (H)
            // {
            //     // // Compute Jacobian if requested
            //     // Matrix36 H_point_wrt_pose;
            //     // pose.transformFrom(measured_point_, H_point_wrt_pose);
            //     // (*H) = H_point_wrt_pose * weight_;
            // }

            /*
            // Transform landmark from world frame to LiDAR frame
            gtsam::Point3 predicted = pose.transformTo(target_point_, H);

            // Compute residual: measured - predicted
            return measured_point_ - predicted;
            */

            // return error;
        }

        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToPointFactor>(
                this->key(),
                measured_point_,
                target_point_,
                this->noiseModel());
        }
    };

    // Custom factor for point-to-plane constraints
    class PointToPlaneFactor : public NoiseModelFactor1<Pose3>
    {
    private:
        Point3 measured_point_; // Point in sensor frame
        Point3 plane_normal_;   // Plane normal (normalized)
        Point3 target_point_;   // A point on the plane in world frame
        double d_;
        bool use_alternative_method_; // Flag to choose between calculation methods

        double huber_delta_ = 1.0; // .1;  maybe take this as parameter

        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToPlaneFactor(Key pose_key,
                           const Point3 &measured_point,
                           const Point3 &plane_norm,
                           const Point3 &target_point,
                           double d,
                           bool use_alternative_method,
                           const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                            measured_point_(measured_point),
                                                            plane_normal_(plane_norm),
                                                            target_point_(target_point),
                                                            d_(d),
                                                            use_alternative_method_(use_alternative_method) {}

        Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
        {
            Matrix36 H_point_wrt_pose;
            Point3 p_transformed = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0); // Transform measured point to world frame

            double error = 0.0;
            if (use_alternative_method_)
            {
                error = (p_transformed - target_point_).dot(plane_normal_);
            }
            else
            {
                error = plane_normal_.dot(p_transformed) + d_;
            }

            // auto vec = p_transformed - target_point_; //p2p
            // auto dist = vec.dot(plane_normal_);
            // auto projected_point = p_transformed - dist * plane_normal_;
            // error = (p_transformed - projected_point).dot(plane_normal_);

            double robust_weight = huberWeight(fabs(error)); //  Apply robust weighting
            // double robust_weight = 1.0;

            if (H)
            {
                *H = (plane_normal_.transpose() * H_point_wrt_pose) * robust_weight;
            }

            return (Vector(1) << error * robust_weight).finished();
        }

        // Clone method for deep copy
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToPlaneFactor>(
                this->key(),
                measured_point_,
                plane_normal_,
                target_point_,
                d_,
                use_alternative_method_,
                this->noiseModel());
        }
    };

    class PointToLineFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
    {
        using Base = gtsam::NoiseModelFactor1<gtsam::Pose3>;

        Point3 measured_point_; // Point in sensor frame
        Point3 line_dir_;       // direction (normalized) should be unit
        Point3 target_point_;   // A point on the plane in world frame

        double huber_delta_ = .1;
        // Huber weight calculation
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToLineFactor(gtsam::Key poseKey,
                          const Point3 &measured_point,
                          const Point3 &target_point,
                          const Point3 &line_dir,
                          const gtsam::SharedNoiseModel &model)
            : Base(model, poseKey),
              measured_point_(measured_point),
              target_point_(target_point),
              line_dir_(line_dir.normalized()) {}

        gtsam::Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
        {
            Matrix36 H_point_wrt_pose;
            // Transform the measured point to world frame
            Point3 world_point = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0);

            // Vector from line center to point
            Point3 point_to_center = world_point - target_point_;

            gtsam::Vector3 cross_err = point_to_center.cross(line_dir_);
            double error_norm = cross_err.norm();

            double robust_weight = huberWeight(error_norm);
            // double robust_weight = 1.0;

            // if (H)
            // {
            //     // ∂e/∂world_point = skew(line_dir_)^T / ||point_to_center × dir||
            //     gtsam::Matrix33 de_dworld_point;
            //     if (error_norm < 1e-8)
            //     {
            //         de_dworld_point.setZero();
            //     }
            //     else
            //     {
            //         de_dworld_point = gtsam::skewSymmetric(line_dir_).transpose() * (1.0 / error_norm);
            //     }

            //     *H = de_dworld_point * H_point_wrt_pose * robust_weight;
            // }

            if (H)
            {
                // Compute Jacobian for scalar error
                gtsam::Matrix13 de_dworld_point;
                if (error_norm < 1e-8)
                {
                    de_dworld_point.setZero();
                }
                else
                {
                    // ∂(‖a×b‖)/∂a = (a×b)^T/‖a×b‖ * [b]×
                    de_dworld_point = cross_err.transpose() / error_norm * gtsam::skewSymmetric(line_dir_);
                }

                // Chain rule: ∂e/∂pose = ∂e/∂world_point * ∂world_point/∂pose
                *H = de_dworld_point * H_point_wrt_pose * robust_weight;
            }

            return (Vector(1) << error_norm * robust_weight).finished();
        }

        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToLineFactor>(
                this->key(),
                measured_point_,
                target_point_,
                line_dir_,
                this->noiseModel());
        }
    };

    class PointToMeshFactor : public NoiseModelFactor1<Pose3>
    {
    private:
        Point3 measured_point_; // Point in sensor frame
        Point3 plane_normal_;   // Plane normal (normalized)
        Point3 target_point_;   // A point on the plane in world frame

        double huber_delta_ = 1.0; // .1;  maybe take this as parameter
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToMeshFactor(Key pose_key,
                          const Point3 &measured_point,
                          const Point3 &plane_norm,
                          const Point3 &target_point,
                          const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                           measured_point_(measured_point),
                                                           plane_normal_(plane_norm),
                                                           target_point_(target_point) {}

        Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
        {
            Matrix36 H_point_wrt_pose;
            Point3 p_transformed = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0); // Transform measured point to world frame

            // Project point to plane
            auto p2p = p_transformed - target_point_; // p2p
            // double dist = p2p.dot(plane_normal_); //distance to the plane
            // target_point = p_transformed - dist * plane_normal_; //projected_point

            auto projected_point = p_transformed - (p2p.dot(plane_normal_) * plane_normal_);

            // Calculate error vector
            Vector3 error = p_transformed - projected_point;

            auto robust_weight = huberWeight(error.norm());
            // double robust_weight = 1.0;

            if (H)
            {
                // Jacobian: ∂error/∂pose = ∂p_world/∂pose
                *H = H_point_wrt_pose * robust_weight;
            }

            return robust_weight * error;
        }

        virtual gtsam::NonlinearFactor::shared_ptr clone() const override
        {
            return std::make_shared<PointToMeshFactor>(
                this->key(),
                measured_point_,
                plane_normal_,
                target_point_,
                this->noiseModel());
        }
    };
};

using namespace custom_factor;

auto sigma_point = 1; // 100cm standard deviation (3σ ≈ 3m allowed)

struct landmark_new
{
    int map_point_index; // index of the point from the reference map
    V3D norm;            // the normal of the plane in global frame
    V3D landmark_point;  // point_on_the_plane;      // point from the map
    double d;            // d parameter of the plane

    double re_proj_error;
    std::vector<int> line_idx; // from which line is seen
    std::vector<int> scan_idx; // from which which point index of the line_idx it is seen
    int landmark_key = 0;

    V3D center;
    M3D covariance;

    V3D key;

    bool is_plane = false;
    bool is_edge = false;
    V3D edge_direction;

    double sigma;
};

#include <Eigen/Dense>
#include <functional>
#include <unordered_map>

struct Vector3dHash
{
    std::size_t operator()(const V3D &point) const
    {
        std::size_t h1 = std::hash<double>()(point.x());
        std::size_t h2 = std::hash<double>()(point.y());
        std::size_t h3 = std::hash<double>()(point.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Vector3dEqual
{
    bool operator()(const V3D &a, const V3D &b) const
    {
        return a.isApprox(b, 1e-6); // Adjust tolerance as needed
    }
};

double robust_kernel = .1;

double landmarks_sigma = 1; // this was used for all the tests so far

// bool use_artificial_uncertainty = false;
bool use_artificial_uncertainty = true;

auto plane_noise_cauchy = gtsam::noiseModel::Robust::Create(
    // gtsam::noiseModel::mEstimator::Cauchy::Create(.2), // Less aggressive than Tukey
    gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // 2cm
    gtsam::noiseModel::Isotropic::Sigma(1, landmarks_sigma));

auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
    gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> global_seen_landmarks;

std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> prev_seen_landmarks;

void debugPoint(const V3D &t, ros::Publisher &pub)
{
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->points.emplace_back(t.x(), t.y(), t.z());
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "world";

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header = header;
    pub.publish(msg);
}

void publishPose(ros::Publisher &pose_pub, const Sophus::SE3 &pose, double cloud_time,
                 const V3D &trans_var, // variance of x, y, z
                 const V3D &rot_var)
{
    /*
        trans_var and rot_var should be in the 'frame_id = "world";' frame 
    */
    std::cout << "\n publishPose :" << cloud_time << ", t:" << pose.translation().transpose() << std::endl;
    nav_msgs::Odometry pose_msg;
    pose_msg.header.frame_id = "world"; 
    //pose_msg.header.stamp = ros::Time().fromSec(cloud_time);
    pose_msg.header.stamp = ros::Time(cloud_time); 

    // pose_msg.header.stamp = ros::Time::now();

    Eigen::Vector3d trans = pose.translation();
    pose_msg.pose.pose.position.x = trans.x();
    pose_msg.pose.pose.position.y = trans.y();
    pose_msg.pose.pose.position.z = trans.z();

    Eigen::Quaterniond q(pose.so3().matrix());
    pose_msg.pose.pose.orientation.x = q.x();
    pose_msg.pose.pose.orientation.y = q.y();
    pose_msg.pose.pose.orientation.z = q.z();
    pose_msg.pose.pose.orientation.w = q.w();

    // Fill diagonal covariance [x, y, z, roll, pitch, yaw]
    for (int i = 0; i < 3; i++)
    {
        pose_msg.pose.covariance[i * 6 + i] = trans_var(i);           // x, y, z
        pose_msg.pose.covariance[(i + 3) * 6 + (i + 3)] = rot_var(i); // roll, pitch, yaw
    }

    // pose_msg.pose.covariance = {
    //     trans_var.x(), 0, 0, 0, 0, 0,
    //     0, trans_var.y(), 0, 0, 0, 0,
    //     0, 0, trans_var.z(), 0, 0, 0,
    //     0, 0, 0, rot_var.x(), 0, 0,
    //     0, 0, 0, 0, rot_var.y(), 0,
    //     0, 0, 0, 0, 0, rot_var.z()
    // };

    pose_pub.publish(pose_msg);
}

std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> get_Landmarks(
    pcl::PointCloud<VUX_PointType>::Ptr &scan,                       // scan in sensor frame
    const Sophus::SE3 &T,                                            // init guess
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,       // reference kdtree
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud, // reference cloud
    double threshold_nn = 1.0, bool radius_based = false, bool weighted_mean = false)
{

    // std::unordered_map<int, landmark_new> landmarks_map; // key is index of the point from the map
    std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> landmarks_map;

    // //return landmarks_map; // no landmarks TEST WITHOUT PLANES

    double uncertainty_scale = 15;

    uncertainty_scale = 1.;
    uncertainty_scale = 10;

    // this can be done in parallel BTW-----------------------------
    for (int i = 0; i < scan->size(); i++)
    {
        // scan->points[i].reflectance = 0; // no error yet;

        // scan->points[i].reflectance = i;

        V3D p_src(scan->points[i].x, scan->points[i].y, scan->points[i].z);
        V3D p_transformed = T * p_src; // transform the point with the initial guess pose

        // if(p_transformed.z() > 0)
        // {
        //     continue;
        // }

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

        if (radius_based)
        {
            // std::cout<<"\n Radius based NN search ..."<<std::endl;

            std::vector<int> point_idx; //(neighbours);
            std::vector<float> point_dist;

            if (refference_kdtree->radiusSearch(search_point, threshold_nn, point_idx, point_dist) >= 15) // at least 15 neighbours
            {
                int neighbours = point_idx.size();

                // Compute the centroid
                V3D centroid(0, 0, 0);
                // Compute covariance matrix
                M3D covariance;
                covariance.setZero();

                // Regularize
                //  double lambda_reg = 1e-6;
                //  covariance = covariance + lambda_reg * Eye3d;

                if (weighted_mean)
                {
                    double weight_sum = 0.0;

                    // Compute weighted centroid
                    for (int j = 0; j < neighbours; ++j)
                    {
                        const double &w = reference_localMap_cloud->points[point_idx[j]].intensity;
                        const auto &p = reference_localMap_cloud->points[point_idx[j]];
                        centroid += w * V3D(p.x, p.y, p.z);
                        weight_sum += w;
                    }
                    centroid /= weight_sum;

                    // Compute weighted covariance matrix
                    for (int j = 0; j < neighbours; ++j)
                    {
                        const double &w = reference_localMap_cloud->points[point_idx[j]].intensity;
                        const auto &p = reference_localMap_cloud->points[point_idx[j]];
                        V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                        covariance += w * diff * diff.transpose();
                    }
                    covariance /= weight_sum;
                }
                else
                {
                    for (int j = 0; j < neighbours; j++)
                    {
                        centroid(0) += reference_localMap_cloud->points[point_idx[j]].x;
                        centroid(1) += reference_localMap_cloud->points[point_idx[j]].y;
                        centroid(2) += reference_localMap_cloud->points[point_idx[j]].z;
                    }
                    centroid /= neighbours;

                    for (int j = 0; j < neighbours; j++)
                    {
                        const auto &p = reference_localMap_cloud->points[point_idx[j]];
                        V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                        covariance += diff * diff.transpose();
                    }
                    covariance /= neighbours;
                }

                // Compute Eigenvalues and Eigenvectors
                Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
                V3D norm = solver.eigenvectors().col(0); // Smallest eigenvector
                norm.normalize();

                // Compute plane offset: d = - (n * centroid)
                double d = -norm.dot(centroid);

                // Compute eigenvalue ratios to assess planarity
                const auto &eigenvalues = solver.eigenvalues();
                double lambda0 = eigenvalues(0); // smallest
                double lambda1 = eigenvalues(1);
                double lambda2 = eigenvalues(2);

                double curvature = lambda0 / (lambda0 + lambda1 + lambda2);

                // Check for invalid or degenerate cases
                if (lambda0 < 0 || (lambda0 + lambda1 + lambda2) < 1e-6)
                {
                    std::cerr << "Degenerate covariance matrix (maybe zero variation). Skipping...\n";
                    std::cerr << "curvature is : " << curvature << std::endl;
                    std::cerr << "lambda0:" << lambda0 << ", lambda1:" << lambda1 << ", lambda2:" << lambda2 << std::endl;

                    throw std::runtime_error("invalid or degenerate cases...");

                    continue;
                }

                // Colinear: if the two smallest eigenvalues are close to zero
                // if ((lambda1 / lambda2) < 1e-3)
                // {
                //     std::cerr << "Colinear structure detected. Skipping...\n";
                //     std::cout<<"eigenvalues:"<<eigenvalues<<std::endl;
                //     throw std::runtime_error("Colinear structure detected. Skipping ...");
                //     continue;
                // }

                // this can be done in the visualization
                //  Flip normal to point consistently toward viewpoint
                //  V3D point_on_the_plane(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                //  if (norm.dot(T.translation() - point_on_the_plane) < 0)
                //  {
                //      norm = -norm;
                //  }

                if (curvature > 0)
                {
                    double linearity = eigenvalues(2) / eigenvalues.sum();
                    // good plane
                    if (curvature <= .01)
                    { //.04 - all tests with .04
                        // Compute sigma (standard deviation along normal)
                        double sigma_plane = uncertainty_scale * std::sqrt(lambda0); // scale_factor = 1.0 -> 1-sigma confidence, 3 and so on

                        // neighbours = 5; //test to take 5 neighbours

                        for (int j = 0; j < neighbours; j++) // all the points share the same normal
                        {
                            V3D point_3d(reference_localMap_cloud->points[point_idx[j]].x, reference_localMap_cloud->points[point_idx[j]].y, reference_localMap_cloud->points[point_idx[j]].z);
                            auto planes_iterator = landmarks_map.find(point_3d); // Search by 3D point instead of index
                            // auto planes_iterator = landmarks_map.find(point_idx[j]);

                            // scan->points[i].reflectance = fabs((p_transformed - point_3d).dot(norm));

                            if (planes_iterator == landmarks_map.end()) // Does not exist -> add new landmark
                            {
                                landmark_new tgt;
                                tgt.is_plane = true;
                                tgt.map_point_index = point_idx[j];
                                tgt.norm = norm;
                                tgt.center = centroid;
                                tgt.covariance = covariance;

                                // tgt.landmark_point = centroid; // centroid

                                tgt.landmark_point = point_3d; // first point

                                tgt.key = point_3d;
                                tgt.sigma = sigma_plane;
                                tgt.d = d;
                                tgt.re_proj_error = curvature;
                                tgt.line_idx.push_back(0); // Seen from line l
                                tgt.scan_idx.push_back(i); // Seen at point i in line l

                                landmarks_map.emplace(point_3d, tgt); // Insert with 3D point as key
                                // landmarks_map[point_idx[j]] = tgt;
                            }
                            // else if (planes_iterator->second.is_plane) // Already exists → update if better
                            // {
                            //     if (planes_iterator->second.re_proj_error > curvature)
                            //     {
                            //         planes_iterator->second.norm = norm;
                            //         planes_iterator->second.d = d;
                            //         planes_iterator->second.re_proj_error = curvature; // Fixed: Use curvature (not point_dist[j])
                            //     }
                            // }

                            break; // to keep only the closest neighbour
                        }
                    }
                    else if (linearity > .8) // edge like
                    {
                        // if (lambda2 > 3 * lambda1) -> good line aloam

                        V3D point_3d(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                        auto edge_iterator = landmarks_map.find(point_3d);

                        double sigma_edge = uncertainty_scale * std::sqrt(lambda1 + lambda2); // perturbation of the edge

                        if (edge_iterator == landmarks_map.end()) // Does not exist -> add new landmark
                        {
                            landmark_new tgt;
                            tgt.is_edge = true;
                            tgt.map_point_index = point_idx[0]; // Optional: Keep original index if needed
                            tgt.norm = norm;

                            tgt.landmark_point = centroid; // point_3d;
                            tgt.key = point_3d;
                            tgt.center = centroid;
                            tgt.covariance = covariance;

                            tgt.d = d;
                            tgt.sigma = sigma_edge;

                            tgt.re_proj_error = linearity; // keep linearity here, bigger -> better

                            tgt.line_idx.push_back(0); // Seen from line l
                            tgt.scan_idx.push_back(i); // Seen at point i in line l

                            V3D line_direction = solver.eigenvectors().col(2);
                            line_direction.normalize();
                            tgt.edge_direction = line_direction; // biggest eigen vector
                            landmarks_map.emplace(point_3d, tgt);
                        }
                        // else if (edge_iterator->second.is_edge) // Already exists → update if better
                        // {
                        //     if (edge_iterator->second.re_proj_error < linearity) // found a better
                        //     {
                        //         V3D line_direction = solver.eigenvectors().col(2);
                        //         line_direction.normalize();

                        //         edge_iterator->second.edge_direction = line_direction;
                        //         edge_iterator->second.d = d;
                        //         edge_iterator->second.re_proj_error = linearity;
                        //     }
                        // }
                    }
                }
            }

            continue; // do not go to the next, kNN
        }

        // std::cout<<"kNN based NN search ..."<<std::endl;
        std::vector<int> point_idx(BA_NEIGH);
        std::vector<float> point_dist(BA_NEIGH);

        if (refference_kdtree->nearestKSearch(search_point, BA_NEIGH, point_idx, point_dist) > 0) // search for neighbour points
        {
            if (point_dist[BA_NEIGH - 1] < threshold_nn)
            {
                // Compute the centroid
                V3D centroid(0, 0, 0);
                for (int j = 0; j < BA_NEIGH; j++)
                {
                    centroid(0) += reference_localMap_cloud->points[point_idx[j]].x;
                    centroid(1) += reference_localMap_cloud->points[point_idx[j]].y;
                    centroid(2) += reference_localMap_cloud->points[point_idx[j]].z;
                }
                centroid /= BA_NEIGH;

                // Compute covariance matrix
                M3D covariance;
                covariance.setZero();
                for (int j = 0; j < BA_NEIGH; j++)
                {
                    const auto &p = reference_localMap_cloud->points[point_idx[j]];
                    V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                    covariance += diff * diff.transpose();
                }
                covariance /= BA_NEIGH;

                // Compute Eigenvalues and Eigenvectors
                Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
                V3D norm = solver.eigenvectors().col(0); // Smallest eigenvector
                norm.normalize();

                // Compute plane offset: d = - (n * centroid)
                double d = -norm.dot(centroid);

                // Compute eigenvalue ratios to assess planarity
                const auto &eigenvalues = solver.eigenvalues();
                double lambda0 = eigenvalues(0); // smallest
                double lambda1 = eigenvalues(1);
                double lambda2 = eigenvalues(2);

                double curvature = lambda0 / (lambda0 + lambda1 + lambda2);

                // Check for invalid or degenerate cases
                if (lambda0 < 0 || (lambda0 + lambda1 + lambda2) < 1e-6)
                {
                    std::cerr << "Degenerate covariance matrix (maybe zero variation). Skipping...\n";
                    std::cerr << "curvature is : " << curvature << std::endl;
                    std::cerr << "lambda0:" << lambda0 << ", lambda1:" << lambda1 << ", lambda2:" << lambda2 << std::endl;
                    // throw std::runtime_error("Handle this...");
                    continue;
                }

                // Colinear: if the two smallest eigenvalues are close to zero
                if ((lambda1 / lambda2) < 1e-3)
                {
                    std::cerr << "Colinear structure detected. Skipping...\n";
                    continue;
                }

                // Flat/Planar Region: curvature ≈ 0.001 - 0.05
                // Edge/Corner: curvature ≈ 0.1 - 0.3
                // Noisy/Irregular: curvature > 0.3

                // Flip normal to point consistently toward viewpoint
                // V3D point_on_the_plane(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                // if (norm.dot(T.translation() - point_on_the_plane) < 0)
                // {
                //     norm = -norm;
                // }

                if (curvature > 0)
                {
                    double linearity = eigenvalues(2) / eigenvalues.sum();
                    // good plane
                    if (curvature <= .02)
                    { //.04 - all tests with .04
                        // Compute sigma (standard deviation along normal)
                        double sigma_plane = std::sqrt(lambda0); // scale_factor = 1.0 -> 1-sigma confidence, 3 and so on

                        for (int j = 0; j < BA_NEIGH; j++) // all the points share the same normal
                        {
                            V3D point_3d(reference_localMap_cloud->points[point_idx[j]].x, reference_localMap_cloud->points[point_idx[j]].y, reference_localMap_cloud->points[point_idx[j]].z);
                            auto planes_iterator = landmarks_map.find(point_3d); // Search by 3D point instead of index
                            // auto planes_iterator = landmarks_map.find(point_idx[j]);

                            if (planes_iterator == landmarks_map.end()) // Does not exist -> add new landmark
                            {
                                landmark_new tgt;
                                tgt.is_plane = true;
                                tgt.map_point_index = point_idx[j];
                                tgt.norm = norm;
                                tgt.center = centroid;
                                tgt.covariance = covariance;

                                tgt.landmark_point = centroid; // centroid
                                // tgt.landmark_point = point_3d;  //first point
                                tgt.key = point_3d;
                                tgt.sigma = sigma_plane;
                                tgt.d = d;
                                tgt.re_proj_error = curvature;
                                tgt.line_idx.push_back(0); // Seen from line l
                                tgt.scan_idx.push_back(i); // Seen at point i in line l

                                landmarks_map.emplace(point_3d, tgt); // Insert with 3D point as key
                                // landmarks_map[point_idx[j]] = tgt;
                            }
                            else if (planes_iterator->second.is_plane) // Already exists → update if better
                            {
                                if (planes_iterator->second.re_proj_error > curvature)
                                {
                                    planes_iterator->second.norm = norm;
                                    planes_iterator->second.d = d;
                                    planes_iterator->second.re_proj_error = curvature; // Fixed: Use curvature (not point_dist[j])
                                }
                            }

                            break; // to keep only the closest neighbour
                        }
                    }
                    else if (linearity > 0.8) // edge like
                    {
                        //
                        // if (lambda2 > 3 * lambda1) -> good line aloam

                        V3D point_3d(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                        auto edge_iterator = landmarks_map.find(point_3d);

                        if (edge_iterator == landmarks_map.end()) // Does not exist -> add new landmark
                        {
                            landmark_new tgt;
                            tgt.is_edge = true;
                            tgt.map_point_index = point_idx[0]; // Optional: Keep original index if needed
                            tgt.norm = norm;

                            tgt.landmark_point = centroid; // point_3d;
                            tgt.key = point_3d;
                            tgt.center = centroid;
                            tgt.covariance = covariance;

                            tgt.d = d;

                            tgt.re_proj_error = linearity; // keep linearity here, bigger -> better

                            tgt.line_idx.push_back(0); // Seen from line l
                            tgt.scan_idx.push_back(i); // Seen at point i in line l

                            V3D line_direction = solver.eigenvectors().col(2);
                            line_direction.normalize();
                            tgt.edge_direction = line_direction; // biggest eigen vector
                            landmarks_map.emplace(point_3d, tgt);
                        }
                        else if (edge_iterator->second.is_edge) // Already exists → update if better
                        {
                            if (edge_iterator->second.re_proj_error < linearity) // found a better
                            {
                                V3D line_direction = solver.eigenvectors().col(2);
                                line_direction.normalize();

                                edge_iterator->second.edge_direction = line_direction;
                                edge_iterator->second.d = d;
                                edge_iterator->second.re_proj_error = linearity;
                            }
                        }
                    }
                }
            }
        }
    }

    return landmarks_map;
}

std::default_random_engine generator(42);
namespace latest_code
{
    Sophus::SE3 addNoiseToPose(const Sophus::SE3 &T,
                               const double &trans_noise_std,
                               const double &rot_noise_std,
                               std::default_random_engine &generator)
    {
        auto sampleGaussian = [&](double std_dev)
        {
            std::normal_distribution<double> dist(0.0, std_dev);
            return dist(generator);
        };

        V3D delta_t(0, 0, 0);

        // works for
        // delta_t[0] = sampleGaussian(trans_noise_std);  // errors on x axis of vux - which is z of the world

        // not good for
        //  delta_t[1] = sampleGaussian(trans_noise_std);
        //  delta_t[2] = sampleGaussian(trans_noise_std);

        V3D delta_rpy(0, 0, 0);
        // delta_rpy[2] = sampleGaussian(rot_noise_std);

        // Convert RPY noise to rotation matrix
        Eigen::AngleAxisd rx(delta_rpy[0], Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd ry(delta_rpy[1], Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rz(delta_rpy[2], Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d delta_R = (rz * ry * rx).toRotationMatrix();

        // Sophus::SE3 noise(delta_R, delta_t);

        // translation and then rotation
        Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();

        delta[3] = sampleGaussian(rot_noise_std); // rotation around local x
        // delta[5] = sampleGaussian(rot_noise_std); // rotation around local Z

        // delta[0] = sampleGaussian(trans_noise_std); // translation on local X
        auto noise = Sophus::SE3::exp(delta);

        std::cout << "noise:\n"
                  << noise.matrix() << std::endl;

        Sophus::SE3 T_noisy = T * noise;
        return T_noisy;
    }

    void updateDataAssociation(ros::Publisher &_pub_debug, int pose_key,
                               const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
                               pcl::PointCloud<VUX_PointType>::Ptr &scan, // scan in sensor frame
                               const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                               const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                               const Sophus::SE3 &nn_init_guess_T, NonlinearFactorGraph &this_Graph, bool use_prev_scans_landmarks = false)
    {

        double threshold_nn = 1.0;
        bool radius_based = true; // false;

        const std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> &landmarks_map = get_Landmarks(
            scan, nn_init_guess_T, refference_kdtree, reference_localMap_cloud,
            threshold_nn, radius_based);

        if (use_prev_scans_landmarks)
            std::cout << "Number of items in landmarks_map: " << landmarks_map.size() << ", global_seen_landmarks: " << global_seen_landmarks.size() << std::endl;

        int added_constraints = 0;
        if (use_prev_scans_landmarks)
        {
            std::cout << "use_prev_scans_landmarks is true,  use prev pose landmarks ..." << std::endl;
        }
        if (landmarks_map.size() > 2) //
        {
            prev_seen_landmarks.clear();
            for (const auto &[landmark_id, land] : landmarks_map)
            {
                for (int i = 0; i < land.scan_idx.size(); i++)
                {
                    const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                    const auto &raw_point = scan->points[p_idx];

                    Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                    Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());

                    if (land.is_plane)
                    {
                        Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                        // if (use_alternative_method_)
                        //     error = (p_transformed - target_point_).dot(plane_normal_);
                        // else
                        //     error = plane_normal_.dot(p_transformed) + d_;

                        bool use_alternative_method = true;
                        // use_alternative_method = false;

                        auto robust_noise = gtsam::noiseModel::Robust::Create(
                            gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel),
                            gtsam::noiseModel::Isotropic::Sigma(1, 3 * land.sigma));

                        if (use_artificial_uncertainty)
                        {
                            robust_noise = plane_noise_cauchy; // use the artificial one
                        }

                        this_Graph.emplace_shared<PointToPlaneFactor>(X(pose_key), measured_point, plane_norm, target_point, land.d,
                                                                      use_alternative_method, robust_noise);

                        // p2plane is more stable than this
                        //  auto point_noise = gtsam::noiseModel::Robust::Create(
                        //  gtsam::noiseModel::mEstimator::Cauchy::Create(.1), // Robust kernel for outliers 10cm
                        //  gtsam::noiseModel::Isotropic::Sigma(3, 1.));
                        //  this_Graph.emplace_shared<PointToMeshFactor>(X(pose_key), measured_point, plane_norm, target_point,
                        //                                               point_noise);

                        added_constraints++;

                        //--------check consistency with prev scans-----------------------
                        if (use_prev_scans_landmarks)
                        {
                            auto it = global_seen_landmarks.find(landmark_id); // check if its seen in prev scan
                            if (it != global_seen_landmarks.end())             // same plane seen in prev scans too
                            {
                                // Point3 target_point_prev(it->second.landmark_point.x(), it->second.landmark_point.y(), it->second.landmark_point.z());
                                // Point3 plane_norm_prev(it->second.norm.x(), it->second.norm.y(), it->second.norm.z());

                                // auto prev_robust_noise = gtsam::noiseModel::Robust::Create(
                                //     gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel),
                                //     gtsam::noiseModel::Isotropic::Sigma(1, 3 * it->second.sigma));

                                // if (use_artificial_uncertainty)
                                // {
                                //     prev_robust_noise = plane_noise_cauchy; // use the artificial one
                                // }

                                // //this_Graph.emplace_shared<PointToPlaneFactor>(X(pose_key), measured_point, plane_norm_prev, target_point_prev, land.d,
                                // //                                              use_alternative_method, prev_robust_noise);

                                // prev_seen_landmarks[land.key] = land;
                            }
                            else
                            {
                                // this landmarks is has not been seet yet,  add it to buffer
                                global_seen_landmarks[land.key] = land;
                            }
                        }
                    }
                    else if (land.is_edge && false)
                    {
                        Point3 line_dir(land.edge_direction.x(), land.edge_direction.y(), land.edge_direction.z());
                        auto robust_noise = gtsam::noiseModel::Robust::Create(
                            gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel),
                            gtsam::noiseModel::Isotropic::Sigma(1, 3 * land.sigma));

                        if (use_artificial_uncertainty)
                        {
                            robust_noise = plane_noise_cauchy; // use the artificial one
                        }
                        this_Graph.emplace_shared<PointToLineFactor>(X(pose_key), measured_point, target_point, line_dir, robust_noise);

                        added_constraints++;
                    }

                    // p2p
                    // auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
                    // gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
                    // gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

                    // auto point_noise = gtsam::noiseModel::Robust::Create(
                    //     gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers 10cm
                    //     gtsam::noiseModel::Isotropic::Sigma(3, .5));

                    // this_Graph.emplace_shared<PointToPointFactor>(X(pose_key), measured_point, target_point, point_noise);

                    break; // means only once
                }
            }
        }

        if (use_prev_scans_landmarks)
            std::cout << "added_constraints:" << added_constraints << std::endl;
    }

    // auto r_sigma = V3D(.01, .01, .01);
    auto t_sigma = V3D(.005, .005, .005);

    // auto t_sigma = V3D(.5, .005, .005); //handles errors on x axis of vux - which is z of the world
    // auto r_sigma = V3D(.01, .01, .1); //roll

    auto r_sigma = V3D(.2, .01, .01); // roll

    auto odom_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector6() << gtsam::Vector3(r_sigma), // rotation stddev (radians): roll, pitch, yaw
         gtsam::Vector3(t_sigma)                      //. translation stddev (m): x, y, z
         )
            .finished());
    // the issues is that is should be // rad,rad,rad,m, m, m

    Vector6 sigmas_prior = (Vector6() << gtsam::Vector3(r_sigma), gtsam::Vector3(t_sigma)).finished();
    // auto prior_noise_model_loose_world = noiseModel::Diagonal::Sigmas(sigmas_prior);
    auto prior_noise_model_loose_world = noiseModel::Gaussian::Covariance(sigmas_prior.array().square().matrix().asDiagonal());

    // Vector6 sigmas_anchor = (Vector6() << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished();
    Vector6 sigmas_anchor = (Vector6() << gtsam::Vector3(r_sigma), gtsam::Vector3(t_sigma)).finished();
    // auto anchor_noise_model_world = noiseModel::Diagonal::Sigmas(sigmas_anchor);
    auto anchor_noise_model_world = noiseModel::Gaussian::Covariance(sigmas_anchor.array().square().matrix().asDiagonal());

    std::deque<Sophus::SE3> pose_buffer;
    std::deque<pcl::PointCloud<VUX_PointType>::Ptr> scan_buffer;

    const size_t max_buffer_size = 50; // 25; // 3;
    const size_t step_size = 25;       // 10;  // Number of poses to slide each time

    bool has_prev_solution = false;

    Pose3 anchor_pose, relative_anchor;
    pcl::PointCloud<VUX_PointType>::Ptr anchor_scan(new pcl::PointCloud<VUX_PointType>);

    double prev_error = 9999., current_error;  // std::numeric_limits<double>::max();
    const double convergence_threshold = .001; // .05; // 5cm  // when to stop

    Sophus::SE3 updateSimple(
        ros::Publisher &_pub_debug, volatile bool &flag,
        ros::Publisher &_pub_prev, ros::Publisher &_pub_curr,
        const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
        const pcl::PointCloud<VUX_PointType>::Ptr &scan,           // scan in sensor frame
        const Sophus::SE3 &initial_pose_clean, Sophus::SE3 &rel_T, // absolute T, odometry
        const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
        const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
        ros::Publisher &pubOptimizedVUX, ros::Publisher &pose_pub, double time = 0)
    {
        Sophus::SE3 initial_pose = initial_pose_clean;

        bool add_noise = true; // false;
        if (add_noise)
        {
            double translation_std = .1; // .3;// .05; // 5cm meters
            double rotation_std = .2;// .05;   // radians - 5.7 degrees

            Sophus::SE3 T_noisy = addNoiseToPose(initial_pose, translation_std, rotation_std, generator);

            initial_pose = T_noisy;
        }


        {
            /*
                some TODOs

                GTSAM : [rot,  tran]
                SOPHUS: [tran, rot]

                -PriorFactor<Pose3>: noise is absolute in the world frame
                -BetweenFactor<Pose3>: noise is in the frame of the first pose 

                // T is a gtsam::Pose3
                gtsam::Vector6 noise;
                gtsam::Pose3 noisy_pose = T.retract(noise); // or Pose3::Expmap(noise) * T

                transforming a noise vector
                    Vector6 noise_in_sensor;
                    Matrix66 Ad_T = T.AdjointMap(); // T = Pose3(sensor_in_world)
                    Vector6 noise_in_world = Ad_T * noise_in_sensor;

                transforming the noise covariance
                    Matrix66 cov_in_sensor;
                    Matrix66 cov_in_world = Ad_T * cov_in_sensor * Ad_T.transpose();


                gtsam::Matrix6 Adjoint_world_to_body = T.inverse().Adjoint();
                gtsam::Matrix6 body_cov = Adjoint_world_to_body * world_cov * Adjoint_world_to_body.transpose();

                ////////////////////////////////////////////////////////////////////////////////

                Pose3 addNoiseToPose(const Pose3& true_pose, const Vector6& noise_local) {
                    // noise_local: 6D vector [rx ry rz tx ty tz] in the local frame
                    // Step 1: Convert to gtsam::Pose3 via exponential map (on tangent space)
                    Pose3 noise = Pose3::Expmap(noise_local);

                    // Step 2: Compose to get noisy pose
                    Pose3 noisy_pose = true_pose.compose(noise);
                    return noisy_pose;
                }

                Pose3 T_true = Pose3(Rot3::RzRyRx(0.1, 0.2, 0.3), Point3(1.0, 2.0, 3.0));
                // Step 1: Define known noise
                Vector6 noise_local;
                noise_local << 0.01, -0.02, 0.005, 0.1, -0.05, 0.02;  // rotation then translation
                // Step 2: Simulate noisy measurement
                Pose3 T_noisy = addNoiseToPose(T_true, noise_local);

                // Assume the noise is in the local (body) frame, so identity covariance
                Vector6 sigmas = noise_local.cwiseAbs();  // std deviations
                auto noiseModel = noiseModel::Diagonal::Sigmas(sigmas);

                // Step 5: Add prior factor (or BetweenFactor) using the known noise
                graph.add(PriorFactor<Pose3>(X(0), T_true, noiseModel));

                Matrix66 Ad_T = T_true.inverse().AdjointMap();  // transforms from world to local
                Vector6 noise_local = Ad_T * noise_world;

                //-----------------------------------------------
                Pose3 pose_i = values.at<Pose3>(X(i));
                Matrix66 local_cov = marginals.marginalCovariance(X(i)); is for the absolute pose of node X(i) in its own frame 
                Matrix66 Ad_T = pose_i.AdjointMap();
                Matrix66 world_cov = Ad_T * local_cov * Ad_T.transpose();  // Now in world frame

                //--------------------------------------------------------------------
                for relative use

                // Step 1: Get joint marginal covariance
                JointMarginal jointMarg = marginals.jointMarginalCovariance(X(i), X(j));
                Eigen::Matrix<double, 12, 12> joint = jointMarg.fullMatrix();

                // Split into blocks
                Matrix66 Sigma_ii = joint.block<6,6>(0,0);
                Matrix66 Sigma_jj = joint.block<6,6>(6,6);
                Matrix66 Sigma_ij = joint.block<6,6>(0,6);  // Cross-covariance

                // Step 2: Compute relative transform T_ij = Ti⁻¹ * Tj
                Pose3 Tij = Ti.between(Tj);

                // Step 3: Get adjoint of T_ij⁻¹
                Matrix66 Ad_Tij_inv = Tij.inverse().AdjointMap();

                // Step 4: Apply covariance propagation rule:
                // Cov(Tij) = Ad_Tij⁻¹ * (Sigma_ii + Sigma_jj - Sigma_ij - Sigma_ijᵗ) * Ad_Tij⁻¹ᵗ
                Matrix66 Sigma_rel = Sigma_ii + Sigma_jj - Sigma_ij - Sigma_ij.transpose();
                Matrix66 Sigma_Tij = Ad_Tij_inv * Sigma_rel * Ad_Tij_inv.transpose();

                return Sigma_Tij;  // 6x6: relative covariance of T_i⁻¹ T_j in local frame of Tij
            */
        }





        pose_buffer.push_back(initial_pose);
        scan_buffer.push_back(scan);

        // rotate translation sigma to curr frame
        auto _sigma_translation = t_sigma; // initial_pose.so3().matrix() * t_sigma;

        // Eigen::Matrix3d Sigma = t_sigma.array().square().matrix().asDiagonal();
        // auto R = initial_pose.so3().matrix();
        // Eigen::Matrix3d rotated_Sigma = R * Sigma * R.transpose();
        // //_sigma_translation = rotated_Sigma.diagonal().cwiseSqrt();

        // std::cout<<"\ninitial t_sigma:"<<t_sigma.transpose()<<std::endl;
        // std::cout<<"rotated _sigma_translation:"<<_sigma_translation.transpose()<<std::endl;

        // do the same for rotation

        odom_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector6() << gtsam::Vector3(r_sigma), // rotation stddev (radians): roll, pitch, yaw
             gtsam::Vector3(_sigma_translation)           //. translation stddev (m): x, y, z
             )
                .finished());

        debugPoint(initial_pose.translation(), _pub_prev);
        if (pose_pub.getNumSubscribers() != 0)
        {
            publishPose(pose_pub, initial_pose, time, _sigma_translation, r_sigma);
        }

        if (pose_buffer.size() < max_buffer_size) // not enough poses
        {
            return initial_pose; // the optimization will start when pose_buffer.size() is equal to max_buffer_size
        }


        
        
        // Initialize values from raw odometry
        Values current_values;
        if (has_prev_solution) // add anchor
        {
            current_values.insert(A(0), anchor_pose);
        }

        std::cout << "updateSimple pose_buffer size:" << pose_buffer.size() << std::endl;
        for (size_t i = 0; i < pose_buffer.size(); i++)
        {
            Pose3 pose = sophusToGtsam(pose_buffer[i]);
            current_values.insert(X(i), pose);
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr buffer_clouds(new pcl::PointCloud<pcl::PointXYZI>());

        // optimization refinement
        bool debug = true; // false;

        NonlinearFactorGraph graph;
        for (int iter = 0; iter < 50; ++iter)
        {
            ros::spinOnce();
            if (flag || !ros::ok())
                break;

            buffer_clouds->clear();
            global_seen_landmarks.clear();

            graph = gtsam::NonlinearFactorGraph(); // .resize(0); // everytime a new graph

            // process the buffer
            for (size_t i = 0; i < pose_buffer.size(); i++)
            {
                // add odometry data from current_values-------------------------------------------------
                Pose3 curr_pose = current_values.at<Pose3>(X(i));

                if (i == 0)
                {
                    graph.add(PriorFactor<Pose3>(X(i), curr_pose, prior_noise_model_loose_world));
                }
                else
                {
                    Pose3 prev_pose = current_values.at<Pose3>(X(i - 1));
                    Pose3 rel_pose = prev_pose.between(curr_pose);
                    // graph.add(BetweenFactor<Pose3>(X(i - 1), X(i), rel_pose, odom_noise_model));
                    graph.add(PriorFactor<Pose3>(X(i), curr_pose, odom_noise_model));
                }

                // Add point-to-plane factors using updated poses--------------------------------------
                Eigen::Matrix4d curr_pose_i = curr_pose.matrix();
                Sophus::SE3 nn_init_guess_T(curr_pose_i.block<3, 3>(0, 0), curr_pose_i.block<3, 1>(0, 3));
                bool use_prev_scans_landmarks = (normals_pub.getNumSubscribers() != 0 || cloud_pub.getNumSubscribers() != 0); // true;
                updateDataAssociation(_pub_debug, i,                                                                          // add planes for graph from pose i
                                      cloud_pub, normals_pub,
                                      scan_buffer[i], // scan in sensor frame
                                      refference_kdtree, reference_localMap_cloud,
                                      nn_init_guess_T, graph, use_prev_scans_landmarks);

                if (_pub_debug.getNumSubscribers() != 0 || normals_pub.getNumSubscribers() != 0 || cloud_pub.getNumSubscribers() != 0)
                {
                    if (_pub_debug.getNumSubscribers() != 0)
                    {
                        for (int j = 0; j < scan_buffer[i]->size(); j++)
                        {
                            V3D p_src(scan_buffer[i]->points[j].x, scan_buffer[i]->points[j].y, scan_buffer[i]->points[j].z);
                            V3D p_transformed = nn_init_guess_T * p_src;

                            pcl::PointXYZI p;
                            p.x = p_transformed.x();
                            p.y = p_transformed.y();
                            p.z = p_transformed.z();
                            p.intensity = i; // scan_buffer[i]->points[j].reflectance;

                            buffer_clouds->push_back(p);
                        }

                        if (has_prev_solution)
                        {
                            auto anchor_T = GtsamToSophus(anchor_pose);
                            for (int j = 0; j < anchor_scan->size(); j++)
                            {
                                V3D p_src(anchor_scan->points[j].x, anchor_scan->points[j].y, anchor_scan->points[j].z);
                                V3D p_transformed = anchor_T * p_src;

                                pcl::PointXYZI p;
                                p.x = p_transformed.x();
                                p.y = p_transformed.y();
                                p.z = p_transformed.z();
                                p.intensity = 0; // anchor_scan->points[j].reflectance;

                                buffer_clouds->push_back(p);
                            }
                        }

                        sensor_msgs::PointCloud2 cloud_msg;
                        pcl::toROSMsg(*buffer_clouds, cloud_msg);
                        cloud_msg.header.frame_id = "world";
                        _pub_debug.publish(cloud_msg);
                    }

                    if (normals_pub.getNumSubscribers() != 0 || cloud_pub.getNumSubscribers() != 0)
                    {
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>());
                        // add the map landmarks
                        for (const auto &[index, land] : global_seen_landmarks)
                        // for (const auto &[index, land] : prev_seen_landmarks)
                        {
                            // break; // just for now do not show them
                            pcl::PointXYZINormal pt;

                            if (land.is_edge)
                            {
                                pt.curvature = -2; // edges
                            }
                            else
                            {
                                // pt.curvature = -1; //prev planes
                                pt.curvature = 1; // <0 plotted as blue
                            }

                            pt.x = land.landmark_point.x();
                            pt.y = land.landmark_point.y();
                            pt.z = land.landmark_point.z();

                            pt.normal_x = land.norm.x();
                            pt.normal_y = land.norm.y();
                            pt.normal_z = land.norm.z();

                            pt.intensity = land.sigma; // just for test

                            cloud_with_normals->push_back(pt);
                        }
                        // // add the new landmarks
                        // for (const auto &[index, land] : landmarks_map)
                        // {
                        //     if (land.is_edge)
                        //     {
                        //         pcl::PointXYZINormal pt;
                        //         pt.x = land.landmark_point.x();
                        //         pt.y = land.landmark_point.y();
                        //         pt.z = land.landmark_point.z();

                        //         pt.normal_x = land.edge_direction.x();
                        //         pt.normal_y = land.edge_direction.y();
                        //         pt.normal_z = land.edge_direction.z();

                        //         pt.curvature = -2;

                        //         cloud_with_normals->push_back(pt);
                        //     }
                        //     else if (land.is_plane)
                        //     {
                        //         pcl::PointXYZINormal pt;
                        //         pt.x = land.landmark_point.x();
                        //         pt.y = land.landmark_point.y();
                        //         pt.z = land.landmark_point.z();

                        //         pt.normal_x = land.norm.x();
                        //         pt.normal_y = land.norm.y();
                        //         pt.normal_z = land.norm.z();

                        //         pt.curvature = -1;

                        //         pt.intensity = land.sigma; // just for test

                        //         cloud_with_normals->push_back(pt);
                        //     }
                        // }

                        debug_CloudWithNormals2(cloud_with_normals, cloud_pub, normals_pub);
                    }
                }
            }

            if (has_prev_solution) // add anchor
            {
                // Re-insert anchor pose factor every time
                graph.add(PriorFactor<Pose3>(A(0), anchor_pose, anchor_noise_model_world));
                // Tie anchor to first pose
                graph.add(BetweenFactor<Pose3>(A(0), X(0), relative_anchor, odom_noise_model));
            }

            LevenbergMarquardtOptimizer optimizer(graph, current_values);
            current_values = optimizer.optimize(); // re-optimize the current values

            current_error = optimizer.error();
            auto d_error = std::abs(prev_error - current_error);

            if (debug)
            {
                std::cout << "\nIteration " << iter << ", error = " << current_error << std::endl;
                std::cout << "Number of factors in graph: " << graph.size() << ", d_error:" << d_error << std::endl;
            }

            if (d_error < convergence_threshold)
            {
                std::cout << "Converged after " << iter << " iterations.\n";
                break;
            }
            prev_error = current_error;

            if (debug)
            {
                std::cout << "Finished one iteration, press enter..." << std::endl;
                std::cin.get();
            }

            // break;
        }

        std::cout << "Accepted press enter... pose_buffer.size():" << pose_buffer.size() << std::endl;
        std::cin.get();

        //has_prev_solution = true; //do not use anchor for now, untill figure how noise works which frame 

        // Estimate new uncertainty for prior
        gtsam::Marginals marginals(graph, current_values);

        gtsam::Matrix anchor_covariance;
        gtsam::Matrix next_prior_covariance;
        if (pose_buffer.size() >= max_buffer_size)
        {
            buffer_clouds->clear();
            for (size_t i = 0; i < step_size; i++) // remove first step_size poses
            {
                if (!pose_buffer.empty())
                {
                    anchor_scan = scan_buffer.front();

                    pose_buffer.pop_front();
                    scan_buffer.pop_front();

                    anchor_pose = current_values.at<Pose3>(X(i));
                    relative_anchor = anchor_pose.between(current_values.at<Pose3>(X(i + 1)));

                    anchor_covariance = marginals.marginalCovariance(X(i)); //the covariances are in frame X(i)
                    next_prior_covariance = marginals.marginalCovariance(X(i + 1));

                    if (pubOptimizedVUX.getNumSubscribers() != 0)
                    {
                        debugPoint(GtsamToSophus(current_values.at<Pose3>(X(i))).translation(), _pub_curr);

                        Sophus::SE3 anchor_T = GtsamToSophus(anchor_pose);
                        for (int j = 0; j < anchor_scan->size(); j++)
                        {
                            V3D p_src(anchor_scan->points[j].x, anchor_scan->points[j].y, anchor_scan->points[j].z);
                            V3D p_transformed = anchor_T * p_src;

                            pcl::PointXYZI p;
                            p.x = p_transformed.x();
                            p.y = p_transformed.y();
                            p.z = p_transformed.z();
                            p.intensity = j; // anchor_scan->points[j].reflectance;

                            buffer_clouds->push_back(p);
                        }
                    }
                }
            }
            if (pubOptimizedVUX.getNumSubscribers() != 0)
            {
                sensor_msgs::PointCloud2 cloud_msg;
                pcl::toROSMsg(*buffer_clouds, cloud_msg);
                cloud_msg.header.frame_id = "world";
                pubOptimizedVUX.publish(cloud_msg);
            }

            prior_noise_model_loose_world = gtsam::noiseModel::Gaussian::Covariance(next_prior_covariance);
            anchor_noise_model_world = gtsam::noiseModel::Gaussian::Covariance(anchor_covariance);
        }

        

        return GtsamToSophus(anchor_pose);
    }
};