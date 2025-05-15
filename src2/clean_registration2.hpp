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

// try now see if it still drifts

// #define BA_NEIGH (20.0) no landmarks found for sparse data

// this will rewure tests with radius based landmarks

// #define BA_NEIGH (30.0) too many enighbours does not fit the 1m threshold

namespace Eigen
{
    const int state_size_ = 6;
    using Matrix6d = Eigen::Matrix<double, state_size_, state_size_>;
    using Matrix3_6d = Eigen::Matrix<double, 3, state_size_>;
    using Vector6d = Eigen::Matrix<double, state_size_, 1>;
} // namespace Eigen

typedef Eigen::Quaterniond Q4D;

namespace registration
{
    struct P2Plane_global
    {
        P2Plane_global(const V3D &curr_point_, const V3D &plane_unit_norm_,
                       double negative_OA_dot_norm_)
            : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_) {}

        template <typename T>
        bool operator()(const T *q, const T *t, T *residual) const
        {
            Eigen::Quaternion<T> q_w(q[3], q[0], q[1], q[2]);
            Eigen::Matrix<T, 3, 1> t_w(t[0], t[1], t[2]);

            // Transform raw scanner point to world frame
            Eigen::Matrix<T, 3, 1> point_world = q_w * curr_point.template cast<T>() + t_w;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double negative_OA_dot_norm_)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_global, 1, 4, 3>(
                new P2Plane_global(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double negative_OA_dot_norm;
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
                                        double negative_OA_dot_norm_, const Sophus::SE3 &fixed_pose_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                                                                        negative_OA_dot_norm(negative_OA_dot_norm_), fixed_pose(fixed_pose_) {}

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
            residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

            return true;
        }

        static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                           const double negative_OA_dot_norm_, const Sophus::SE3 &fixed_pose_)
        {
            return (new ceres::AutoDiffCostFunction<
                    LidarPlaneNormFactor_extrinsics, 1, 4, 3>(
                new LidarPlaneNormFactor_extrinsics(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, fixed_pose_)));
        }

        V3D curr_point;
        V3D plane_unit_norm;
        double negative_OA_dot_norm;
        Sophus::SE3 fixed_pose;
    };
};

using namespace registration;

double square(const double &x)
{
    return x * x;
}

double scan2map_ceres(pcl::PointCloud<PointType>::Ptr &src,
                      const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                      const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                      Eigen::Quaterniond &q, V3D &t,
                      const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                      bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    return 0; // implemented in clean_Registration.hpp
}

double scan2map_GN_omp(pcl::PointCloud<PointType>::Ptr &src,
                       const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                       const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                       Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                       const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                       bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
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

    std::cout << "Run GN omp ..." << std::endl;
    if (p2plane && p2p)
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

            if (p2p && p2plane) // both
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
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
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

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
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
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                     norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
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

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
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

double scan2map_planes_test(pcl::PointCloud<PointType>::Ptr &src,
                            const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                            const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                            Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                            double threshold_nn = 1.0)
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

    std::cout << "Run scan2map_test ..." << std::endl;

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

            {
                std::vector<int> point_idx(BA_NEIGH);
                std::vector<float> point_dist(BA_NEIGH);
                if (refference_kdtree->nearestKSearch(search_point, BA_NEIGH, point_idx, point_dist) > 0)
                {
                    if (point_dist[BA_NEIGH - 1] < threshold_nn) // not too far
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
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        // Compute eigenvalue ratios to assess planarity
                        const auto &eigenvalues = solver.eigenvalues();
                        double lambda0 = eigenvalues(0); // smallest
                        double lambda1 = eigenvalues(1);
                        double lambda2 = eigenvalues(2);

                        // Planarity filter: reject unstable normals
                        // double th = .3;//allow noisy data
                        // double th = .7;                                                             // good ones
                        // bool planeValid = (lambda2 > 1e-6) && ((lambda1 - lambda0) / lambda2 > th); // Tunable thresholds

                        double curvature = lambda0 / (lambda0 + lambda1 + lambda2);
                        bool planeValid = false;
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

                        if (curvature > 0 && curvature <= .01) //.01
                        {
                            planeValid = true;
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

                            // double residual = norm(0) * p_transformed.x() + norm(1) * p_transformed.y() + norm(2) * p_transformed.z() + negative_OA_dot_norm;
                            double residual = (p_transformed - target_point).dot(norm);

                            double w = Weight(residual * residual);
                            JTJ_private.noalias() += J_r * w * J_r.transpose();
                            JTr_private.noalias() += J_r * w * residual;

                            cost_private += w * residual * residual; // Always non-negative
                        }
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

void debug_CloudWithNormals(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals,
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

void debug_CloudWithNormals2(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals,
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
    double normal_length = 2.;     // 5.;     // Length of normal lines
    double opacity = 1.0;
    for (const auto &point : cloud_with_normals->points)
    {
        geometry_msgs::Point p1, p2;

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
            color.r = 0.3;
            color.g = 0.8;
            color.b = 1.0;
            opacity = 1.0;
        }
        else if (point.curvature == -1)
        {
            color.r = 0.0;
            color.g = 1.0; // seen once
            color.b = 0.0;
            opacity = .8;
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

using gtsam::symbol_shorthand::L; // Landmark symbols
using gtsam::symbol_shorthand::X; // Pose symbols
using gtsam::symbol_shorthand::Z; // prev Pose symbols

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

void debugGraphs(const gtsam::Values &prev_values, const gtsam::Values &curr_values,
                 ros::Publisher &prev_pub, ros::Publisher &curr_pub)
{
    // ros::Publisher prev_pub = nh.advertise<sensor_msgs::PointCloud2>("prev_graph_cloud", 1);
    // ros::Publisher curr_pub = nh.advertise<sensor_msgs::PointCloud2>("curr_graph_cloud", 1);

    auto prev_cloud = extractTranslations(prev_values);
    auto curr_cloud = extractTranslations(curr_values);

    std::cout << "prev_cloud:" << prev_cloud->size() << ", curr_cloud:" << curr_cloud->size() << std::endl;

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "world";

    publishBACloud(prev_pub, prev_cloud, header);
    publishBACloud(curr_pub, curr_cloud, header);
}

struct landmark_
{
    int map_point_index;         // index of the point from the reference map
    V3D norm;                    // the normal of the plane in global frame
    V3D point_on_the_plane;      // point from the map
    double negative_OA_dot_norm; // d parameter of the plane
    int seen;                    // how many times it has been seen
    double re_proj_error;
    std::vector<int> line_idx; // from which line is seen
    std::vector<int> scan_idx; // from which which point index of the line_idx it is seen
    int landmark_key;

    bool is_plane = false;
    bool is_edge = false;
    V3D edge_direction;
};

// only planes for now
std::unordered_map<int, landmark_> get_Correspondences(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines, // a list of lines
    const std::deque<Sophus::SE3> &line_poses_,                         // with their initial guesses
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,          // reference kdtree
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,    // reference cloud
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment, double threshold_nn = 1.0, double good_plane_threshold = .1)
{
    // Containers
    std::unordered_map<int, landmark_> landmarks_map; // key is index of the point from the map

    // return landmarks_map; // no landmarks TEST WITHOUT PLANES

#define pca_norms

#ifdef pca_norms
    std::cout << "we're using PCA for planes" << std::endl;
#endif
    //  Loop through each scan/line
    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        auto &T = line_poses_[l]; // get the pose of current line
        for (int i = 0; i < lidar_lines[l]->size(); i++)
        {
            V3D p_src(lidar_lines[l]->points[i].x, lidar_lines[l]->points[i].y, lidar_lines[l]->points[i].z);
            V3D p_transformed = T * p_src; // transform the point with the initial guess pose

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            std::vector<int> point_idx(BA_NEIGH);
            std::vector<float> point_dist(BA_NEIGH);

            if (refference_kdtree->nearestKSearch(search_point, BA_NEIGH, point_idx, point_dist) > 0) // search for 5 neighbour points
            {
                if (point_dist[BA_NEIGH - 1] < threshold_nn)
                {
#ifndef pca_norms
                    // this will do the LS plane fitting

                    Eigen::Matrix<double, BA_NEIGH, 3> matA0;
                    Eigen::Matrix<double, BA_NEIGH, 1> matB0 = -1 * Eigen::Matrix<double, BA_NEIGH, 1>::Ones();

                    for (int j = 0; j < BA_NEIGH; j++)
                    {
                        matA0(j, 0) = reference_localMap_cloud->points[point_idx[j]].x;
                        matA0(j, 1) = reference_localMap_cloud->points[point_idx[j]].y;
                        matA0(j, 2) = reference_localMap_cloud->points[point_idx[j]].z;
                    }

                    // find the norm of plane
                    V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    bool planeValid = true;
                    for (int j = 0; j < BA_NEIGH; j++)
                    {
                        double d = fabs(norm(0) * reference_localMap_cloud->points[point_idx[j]].x +
                                        norm(1) * reference_localMap_cloud->points[point_idx[j]].y +
                                        norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm);
                        if (d > good_plane_threshold)
                        {
                            planeValid = false;
                            break;
                        }
                        point_dist[j] = d;
                    }

#else
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
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    // Compute eigenvalue ratios to assess planarity
                    const auto &eigenvalues = solver.eigenvalues();
                    double lambda0 = eigenvalues(0); // smallest
                    double lambda1 = eigenvalues(1);
                    double lambda2 = eigenvalues(2);

                    // Planarity filter: reject unstable normals
                    // double th = .3;//allow noisy data
                    // double th = .7;                                                             // good ones
                    // bool planeValid = (lambda2 > 1e-6) && ((lambda1 - lambda0) / lambda2 > th); // Tunable thresholds

                    // for (int j = 0; j < BA_NEIGH; j++)
                    //     point_dist[j] = lambda2;

                    double curvature = lambda0 / (lambda0 + lambda1 + lambda2);
                    bool planeValid = false;
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

                    if (curvature > 0 && curvature <= .005) //.01
                    {
                        planeValid = true;
                        for (int j = 0; j < BA_NEIGH; j++)
                            point_dist[j] = curvature;
                    }

#endif

                    if (planeValid) // found a good plane
                    {
                        // Flip normal to point consistently toward viewpoint
                        // V3D point_on_the_plane(reference_localMap_cloud->points[point_idx[0]].x, reference_localMap_cloud->points[point_idx[0]].y, reference_localMap_cloud->points[point_idx[0]].z);
                        // if (norm.dot(T.translation() - point_on_the_plane) < 0)
                        // {
                        //     norm = -norm;
                        // }

                        for (int j = 0; j < BA_NEIGH; j++) // all the points share the same normal
                        {
                            auto planes_iterator = landmarks_map.find(point_idx[j]);
                            if (planes_iterator == landmarks_map.end()) // does not exist - add
                            {
                                landmark_ tgt;
                                tgt.map_point_index = point_idx[j];              // index of the point from the map
                                tgt.norm = norm;                                 // plane normal
                                tgt.negative_OA_dot_norm = negative_OA_dot_norm; // d of the plane
                                tgt.seen = 1;                                    // first time seen
                                tgt.re_proj_error = point_dist[j];
                                tgt.line_idx.push_back(l); // seen from line l
                                tgt.scan_idx.push_back(i); // and point i from this line l

                                landmarks_map[point_idx[j]] = tgt;
                            }
                            else // Already exists - increment "seen"
                            {
                                // // a modification was added here - increment seen only if seen from a different line
                                // //TODO here
                                // planes_iterator->second.seen++;                // Increment the seen count
                                // planes_iterator->second.line_idx.push_back(l); // seen from line l
                                // planes_iterator->second.scan_idx.push_back(i); // and point i from this line
                                // // if a better normal was found - replace the old one
                                // if (planes_iterator->second.re_proj_error > point_dist[j])
                                // {
                                //     planes_iterator->second.norm = norm;
                                //     planes_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                //     planes_iterator->second.re_proj_error = point_dist[j];
                                // }

                                { // added new

                                    // Check if the current line l is already recorded
                                    // check if this plane already has measurement from line l
                                    bool already_seen_from_this_line =
                                        std::find(planes_iterator->second.line_idx.begin(), planes_iterator->second.line_idx.end(), l) != planes_iterator->second.line_idx.end();

                                    // already_seen_from_this_line = false; //just a test remove this

                                    // Only increment seen if this landmark is seen from a new line
                                    if (already_seen_from_this_line == false)
                                    {
                                        planes_iterator->second.seen++;                // Increment the seen count
                                        planes_iterator->second.line_idx.push_back(l); // seen from line l
                                        planes_iterator->second.scan_idx.push_back(i); // and point i from this line

                                        // Update normal if new reprojection error is better
                                        // if (planes_iterator->second.re_proj_error > point_dist[j])
                                        // {
                                        //     planes_iterator->second.norm = norm;
                                        //     planes_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                        //     planes_iterator->second.re_proj_error = point_dist[j];
                                        // }
                                    }
                                    else
                                    {
                                        // Same line as before, do not increment seen
                                        // Only update the reprojection data if better

                                        // if (planes_iterator->second.re_proj_error > point_dist[j])
                                        // {
                                        //     planes_iterator->second.norm = norm;
                                        //     planes_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                        //     planes_iterator->second.re_proj_error = point_dist[j];
                                        // }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return landmarks_map;
}

double GM_robust_kernel(const double &residual2)
{
    double kernel_ = 1.0;
    return square(kernel_) / square(kernel_ + residual2);
}

class PlaneFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>
{
private:
    gtsam::Point3 sensor_point_; // Observed point (sensor frame)
    gtsam::Unit3 plane_normal_;  // Plane normal (global frame, as Unit3)

public:
    PlaneFactor(
        gtsam::Key poseKey,
        gtsam::Key landmarkKey,
        const gtsam::Point3 &sensor_point,
        const gtsam::Unit3 &plane_normal,
        const gtsam::SharedNoiseModel &noise_model) : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(noise_model, poseKey, landmarkKey),
                                                      sensor_point_(sensor_point),
                                                      plane_normal_(plane_normal) {}

    gtsam::Vector evaluateError(
        const gtsam::Pose3 &pose,
        const gtsam::Point3 &landmark_point,
        OptionalMatrixType H1,
        OptionalMatrixType H2) const override
    {
        // Transform sensor point to global frame: p_global = pose * p_sensor
        gtsam::Point3 p_global = pose.transformFrom(sensor_point_);

        // Point-to-plane error: (p_global - landmark_point) · plane_normal_
        double error = gtsam::dot(p_global - landmark_point, plane_normal_.unitVector());

        // Jacobians (if requested)
        if (H1)
        {
            // Derivative w.r.t. pose (6D)
            gtsam::Matrix36 D_p_global_wrt_pose;
            p_global = pose.transformFrom(sensor_point_, D_p_global_wrt_pose);
            *H1 = plane_normal_.unitVector().transpose() * D_p_global_wrt_pose;
        }
        if (H2)
        {
            // Derivative w.r.t. landmark point (3D)
            *H2 = -plane_normal_.unitVector().transpose();
        }

        return (gtsam::Vector(1) << error).finished();
    }
};

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
        Point3 measured_point_;       // Point in sensor frame
        Point3 plane_normal_;         // Plane normal (normalized)
        Point3 target_point_;         // A point on the plane in world frame
        double negative_OA_dot_norm_; // = 1 / norm.norm()
        bool use_alternative_method_; // Flag to choose between calculation methods

        double huber_delta_ = .1; // 10cm
        // Huber weight calculation
        double huberWeight(double error_norm) const
        {
            return (error_norm <= huber_delta_) ? 1.0 : huber_delta_ / error_norm;
        }

    public:
        PointToPlaneFactor(Key pose_key,
                           const Point3 &measured_point,
                           const Point3 &plane_norm,
                           const Point3 &target_point,
                           double negative_OA_dot_norm,
                           bool use_alternative_method,
                           const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                            measured_point_(measured_point),
                                                            plane_normal_(plane_norm),
                                                            target_point_(target_point),
                                                            negative_OA_dot_norm_(negative_OA_dot_norm),
                                                            use_alternative_method_(use_alternative_method) {}

        Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
        {
            // Transform measured point to world frame
            // Point3 p_transformed = pose.transformFrom(measured_point_, H);
            Matrix36 H_point_wrt_pose;
            Point3 p_transformed = pose.transformFrom(measured_point_, H ? &H_point_wrt_pose : 0);

            double error = 0.0;
            if (use_alternative_method_)
            {
                error = (p_transformed - target_point_).dot(plane_normal_);
            }
            else
            {
                error = plane_normal_.dot(p_transformed) + negative_OA_dot_norm_;
            }

            //  Apply robust weighting
            // auto robust_weight = GM_robust_kernel(error * error);

            double robust_weight = huberWeight(fabs(error));
            // double robust_weight = 1.0;

            if (H)
            {
                // Apply same weight to Jacobian
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
                negative_OA_dot_norm_,
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

    class PointToDistributionFactor : public NoiseModelFactor1<Pose3> {
        using Base = NoiseModelFactor1<Pose3>;
        Point3 point_;                // LiDAR point in local frame
        Point3 mean_;                 // Mean of nearby map points in world frame
        Matrix3 invCov_;              // Inverse covariance (Σ⁻¹)
      
      public:
        PointToDistributionFactor(Key poseKey,
                                  const Point3& point,
                                  const Point3& mean,
                                  const Matrix3& invCov)
          : Base(noiseModel::Unit::Create(3), poseKey),
            point_(point), mean_(mean), invCov_(invCov) {}
      
        Vector evaluateError(const Pose3& pose, OptionalMatrixType H) const override {
          Point3 p_world = pose.transformFrom(point_, H);
      
          Vector3 diff = p_world - mean_;
          if (H) {
            *H = invCov_ * (*H); // Chain rule: dMahalanobis/dPose = Σ⁻¹ * dP/dPose
          }
          return invCov_ * diff; // Mahalanobis residual vector
        }
      
        virtual ~PointToDistributionFactor() {}
      };
};

using namespace custom_factor;

NonlinearFactorGraph prev_graph;
Values prev_optimized_values;
bool prev_graph_exist = false;

auto sigma_point = 1; // 100cm standard deviation (3σ ≈ 3m allowed)
auto sigma_plane = 1; // 1; // 100cm standard deviation (3σ ≈ 30cm allowed)

// auto sigma_odom = .02; //for relative odometry - tested with skip one scan
auto sigma_odom = .01;  // 1cm .005; for relative odometry no scan skip
auto sigma_prior = 1.0; // not sure about the prior - let it change it

auto point_noise = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(0.5), // Robust kernel for outliers
    // gtsam::noiseModel::mEstimator::Huber(.5),
    gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

auto plane_noise = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Tukey::Create(0.1), // Very aggressive outlier rejection
    // gtsam::noiseModel::mEstimator::Cauchy::Create(0.5),
    gtsam::noiseModel::Isotropic::Sigma(1, sigma_plane));

auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3::Constant(sigma_odom), gtsam::Vector3::Constant(sigma_odom)).finished());

// auto sigma_trans = .05;      // .05 cm for translation (x,y,z)
// auto sigma_rot_deg = 5.0;     // 5 degrees for rotation
// auto sigma_rot_rad = sigma_rot_deg * (M_PI / 180.0);  // Convert to radians

// // Create a 6D diagonal noise model (first 3: translation, next 3: rotation)
// auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
//     (gtsam::Vector(6) << sigma_trans, sigma_trans, sigma_trans,
//                           sigma_rot_rad, sigma_rot_rad, sigma_rot_rad).finished());

auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3::Constant(sigma_prior), gtsam::Vector3::Constant(sigma_prior)).finished());

auto tight_noise = noiseModel::Diagonal::Sigmas(
    (Vector6() << .0001, .0001, .0001, .0001, .0001, .0001).finished());

void updatePriorUncertainty(gtsam::NonlinearFactorGraph &graph,
                            gtsam::Key prior_key,
                            const gtsam::SharedNoiseModel &new_prior_noise)
{
    // Find and remove the existing prior factor
    for (size_t i = 0; i < graph.size(); ++i)
    {
        if (auto prior_factor = dynamic_cast<gtsam::PriorFactor<gtsam::Pose3> *>(graph[i].get()))
        {
            if (prior_factor->key() == prior_key)
            {
                // // Remove the old factor
                // graph.erase(graph.begin() + i);

                // // Add new factor with updated noise model
                // graph.addPrior(prior_key, prior_factor->prior(), new_prior_noise);

                // Remove the old factor
                std::cout << "\n[DEBUG] Replacing prior factor for key: "
                          << gtsam::DefaultKeyFormatter(prior_key) << std::endl;
                std::cout << " - Old noise model sigmas: ";
                if (prior_factor->noiseModel())
                {
                    auto diagonal = dynamic_cast<const gtsam::noiseModel::Diagonal *>(prior_factor->noiseModel().get());
                    if (diagonal)
                    {
                        std::cout << diagonal->sigmas().transpose() << std::endl;
                    }
                    else
                    {
                        std::cout << "(non-diagonal noise model)" << std::endl;
                    }
                }
                else
                {
                    std::cout << "(no noise model)" << std::endl;
                }

                graph.erase(graph.begin() + i);
                std::cout << " - Removed old prior factor at index " << i << std::endl;

                // Add new factor with updated noise model
                std::cout << " - New noise model sigmas: ";
                if (auto diagonal = dynamic_cast<const gtsam::noiseModel::Diagonal *>(new_prior_noise.get()))
                {
                    std::cout << diagonal->sigmas().transpose() << std::endl;
                }
                else
                {
                    std::cout << "(non-diagonal noise model)" << std::endl;
                }

                graph.addPrior(prior_key, prior_factor->prior(), new_prior_noise);
                std::cout << " - Successfully added new prior factor with tighter uncertainty\n"
                          << std::endl;

                return;
            }
        }
    }

    throw std::runtime_error("Prior factor not found in graph");
}

void mergeXandZGraphs(NonlinearFactorGraph &merged_graph, Values &merged_values,
                      const NonlinearFactorGraph &prev_graph, const Values &prev_values,
                      const NonlinearFactorGraph &curr_graph, const Values &curr_values,
                      const bool curr_uses_x,
                      int overlap_size = 50, int total_poses = 100)
{
    /*
    X-graph: x0 ─── x1 ─── ... ─── x49 ─── x50 ─── ... ─── x99
                             │        │            │
    Identity constraints:              ≈ z0     ≈ z1         ≈ z49
    Z-graph:                          z0 ─── z1 ─── ... ─── z49 ─── z50 ─── ... ─── z99
    */

    bool prev_uses_x = !curr_uses_x;

    // Helper function to get the correct symbol
    auto symbol = [](bool use_x, int i)
    {
        return use_x ? X(i) : Z(i);
    };

    // 1. Add all previous graph components
    // std::cout << "\nAdd all previous graph components" << std::endl;
    merged_values.insert(prev_values);
    for (const auto &factor : prev_graph)
    {
        merged_graph.push_back(factor);
    }

    if (true)
    {
        // (x, y, z, roll, pitch, yaw)
        gtsam::Vector6 new_prior_sigmas;
        new_prior_sigmas << .01, .01, .01, .01, .01, .01;

        auto new_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(new_prior_sigmas);

        // Update the prior uncertainty
        //  X(0) or Z(0)
        Key prev_prior_key = symbol(prev_uses_x, 0);
        std::cout << "\n updatePriorUncertainty prev_prior_key: " << prev_prior_key << std::endl;
        // Decode and print
        gtsam::Symbol s(prev_prior_key);
        std::cout << "Prior node: Key: " << s.chr() << s.index() << std::endl;

        updatePriorUncertainty(merged_graph, prev_prior_key, new_prior_noise);

        // updatePriorNoise(merged_graph, merged_values,prev_prior_key, new_prior_noise);
    }

    // 2. Add current graph components
    // std::cout << "\nAdd current graph components" << std::endl;
    merged_values.insert(curr_values);
    for (const auto &factor : curr_graph)
    {
        merged_graph.push_back(factor);
    }

    // 3. Add identity constraints between overlapping poses
    // std::cout << "\nAdd identity constraints between overlapping poses" << std::endl;
    for (int i = 0; i < overlap_size; i++)
    {
        // Last 'overlap_size' poses of previous graph
        Key prev_key = symbol(prev_uses_x, total_poses - overlap_size + i);
        // First 'overlap_size' poses of current graph
        Key curr_key = symbol(curr_uses_x, i);

        // if(i==0)
        {
            // Use Symbol to get the symbol corresponding to the Key
            // Symbol symbol_prev(prev_key), symbol_curr(curr_key);
            // std::cout << "\nsymbol_prev: " << symbol_prev << " symbol_curr: " << symbol_curr << std::endl;
        }

        merged_graph.emplace_shared<BetweenFactor<Pose3>>(prev_key, curr_key, Pose3(), tight_noise);
    }

    // 4. Add odometry bridge between last non-overlapping poses
    // std::cout << "\nAdd odometry bridge between last non-overlapping poses" << std::endl;
    /*
    Previous segment poses: [0 ... 49     50 ... 99]
                  (non-overlap) (overlap)
    Current segment poses:                [0 ... 49  50 ... 99]
                                (overlap) (non-overlap)

    Connecting:
    - Previous segment's last non-overlap pose: 49
    - Current segment's first non-overlap pose: 50
    */
    if (overlap_size < total_poses && false)
    {
        auto odometry_noise = noiseModel::Diagonal::Sigmas(
            (Vector6() << 0.1, 0.1, 0.1, 0.05, 0.05, 0.05).finished());

        // Last non-overlap pose of previous graph (using correct symbol type)
        // Key last_prev = prev_uses_x ? X(total_poses - overlap_size - 1) : Z(total_poses - overlap_size - 1);

        // First non-overlap pose of current graph (using correct symbol type)
        // Key first_curr = curr_uses_x ? X(overlap_size) : Z(overlap_size);

        // std::cout<<"prev_uses_x:"<<prev_uses_x<<", curr_uses_x:"<<curr_uses_x<<std::endl;

        Key last_prev = prev_uses_x ? X(total_poses - overlap_size - 1) : Z(total_poses - overlap_size - 1);
        Key first_curr = curr_uses_x ? X(overlap_size) : Z(overlap_size);

        // this is only for debugging
        //  std::cout << "Connecting:\n";
        //  std::cout << "  Last prev pose: " << gtsam::DefaultKeyFormatter(last_prev) << "\n";
        //  std::cout << "  First curr pose: " << gtsam::DefaultKeyFormatter(first_curr) << "\n";

        // auto first_key_prev = Symbol(prev_values.begin()->key);
        // std::cout << "\nfirst_key_prev: " << first_key_prev << std::endl;

        // auto first_key_curr = Symbol(curr_values.begin()->key);
        // std::cout << "first_key_curr: " << first_key_curr << std::endl;

        // Verify keys exist before accessing
        // if (!prev_values.exists(last_prev))
        // {
        //     std::cerr << "Error: Previous pose " << gtsam::DefaultKeyFormatter(last_prev)
        //               << " does not exist\n";
        //     //continue;
        // }
        // if (!curr_values.exists(first_curr))
        // {
        //     std::cerr << "Error: Current pose " << gtsam::DefaultKeyFormatter(first_curr)
        //               << " does not exist\n";
        //     //continue;
        // }

        Pose3 last_prev_pose = prev_values.at<Pose3>(last_prev);
        Pose3 first_curr_pose = curr_values.at<Pose3>(first_curr);

        // Calculate relative transform
        Pose3 relative = last_prev_pose.inverse() * first_curr_pose;

        // Add the between factor
        merged_graph.emplace_shared<BetweenFactor<Pose3>>(last_prev, first_curr, relative, odometry_noise);
    }

    // 5. Add anchor - anchor_pose (fixed prior) ──(anchor_delta)──▶ first_node
    // std::cout << "anchor_pose:" << anchor_pose.log().transpose() << std::endl;
    // std::cout << "anchor_delta:" << anchor_delta.log().transpose() << std::endl;
    // if(anchor_delta.log().norm() > 0){
    //     std::cout<<"Add first anchor..."<<std::endl;
    //     Pose3 anchor_pose(anchor_pose.matrix());
    //     Pose3 anchor_delta(anchor_delta.matrix());

    //     gtsam::Key anchor_key = A(0);  // Arbitrary high index for anchor

    //     Key first_prev = prev_uses_x ? X(0) : Z(0);

    //     std::cout << "Connecting A(0) with First prev pose: " << gtsam::DefaultKeyFormatter(first_prev) << "\n";

    //     auto anchor_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished()); // tight
    //     merged_graph.add(PriorFactor<Pose3>(anchor_key, anchor_pose, anchor_noise));
    //     merged_values.insert(anchor_key, anchor_pose);
    //     merged_graph.add(BetweenFactor<Pose3>(anchor_key, first_prev, anchor_delta, anchor_noise));
    // }
}

bool useX = true; // Set this flag to false to use Z instead of X

void buildGraph(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    NonlinearFactorGraph &graph, Values &initial_values,
    int overlap_size = 50,
    double threshold_nn = 1.0, bool p2p = true, bool p2plane = false)
{
    std::cout << "Run BA_refinement_merge_graph..." << std::endl;

    double planarity = .05; // 5cm allowed
    // double planarity = .02;   //2 cm allowed

    const std::unordered_map<int, landmark_> &landmarks_map = get_Correspondences(
        lidar_lines,
        line_poses,
        refference_kdtree,
        reference_localMap_cloud,
        prev_segment_init,
        prev_segment,
        kdtree_prev_segment, threshold_nn, planarity);

    if (normals_pub.getNumSubscribers() != 0)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        for (const auto &[index, land] : landmarks_map)
        {
            pcl::PointNormal pt;
            pt.x = reference_localMap_cloud->points[land.map_point_index].x;
            pt.y = reference_localMap_cloud->points[land.map_point_index].y;
            pt.z = reference_localMap_cloud->points[land.map_point_index].z;

            pt.normal_x = land.norm.x();
            pt.normal_y = land.norm.y();
            pt.normal_z = land.norm.z();

            // pt.curvature = land.seen;
            pt.curvature = land.re_proj_error;
            // pt.curvature = land.line_idx.size();

            cloud_with_normals->push_back(pt);
        }

        debug_CloudWithNormals(cloud_with_normals, cloud_pub, normals_pub);
    }

    // 1. Create nodes from initial odometry poses
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Pose3 gtsam_pose(line_poses[i].matrix());

        // Add to initial values
        initial_values.insert(useX ? X(i) : Z(i), gtsam_pose);

        // Add prior on the first pose to fix the coordinate frame
        if (i == 0)
        {
            graph.addPrior(useX ? X(0) : Z(0), gtsam_pose, prior_noise);
        }

        // Add odometry constraints between consecutive poses
        if (i > 0)
        {
            Sophus::SE3 relative_pose = line_poses[i - 1].inverse() * line_poses[i];
            Pose3 gtsam_relative(relative_pose.matrix());

            graph.emplace_shared<BetweenFactor<Pose3>>(useX ? X(i - 1) : Z(i - 1), useX ? X(i) : Z(i), gtsam_relative, odometry_noise);
        }
    }

    // 2. Add constraints
    int added_constraints = 0;
    for (const auto &[landmark_id, land] : landmarks_map)
    {
        if (land.seen > 1) // if the landmark is seen from multiple measurements
        {
            for (int i = 0; i < land.seen; i++)
            {
                const auto &pose_idx = land.line_idx[i]; // point from line pose_idx
                const auto &p_idx = land.scan_idx[i];    // at index p_idx from that scan
                const auto &raw_point = lidar_lines[pose_idx]->points[p_idx];
                // measured_landmar_in_sensor_frame
                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z);

                Point3 target_point;

                target_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                      reference_localMap_cloud->points[land.map_point_index].y,
                                      reference_localMap_cloud->points[land.map_point_index].z);

                // auto weighted_plane_noise = noiseModel::Isotropic::Sigma(1, 0.1/land.weight);
                Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());
                bool use_alternative_method = false; // true;

                use_alternative_method = true;

                graph.emplace_shared<PointToPlaneFactor>(useX ? X(pose_idx) : Z(pose_idx), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                         use_alternative_method, plane_noise);

                added_constraints++;
            }
        }
    }

    std::cout << "added_constraints:" << added_constraints << std::endl;
}

double BA_refinement_merge_graph(
    ros::Publisher &prev_pub, ros::Publisher &curr_pub,

    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    int run_iterations = 1, bool flg_exit = false,
    int overlap_size = 50,
    double threshold_nn = 1.0, bool p2p = true, bool p2plane = false,
    Sophus::SE3 anchor_pose = Sophus::SE3(), Sophus::SE3 anchor_delta = Sophus::SE3())
{
    useX = !useX;
    std::cout << "For current graph useX:" << useX << " overlap_size:" << overlap_size << std::endl;
    std::cout << " reference_localMap_cloud:" << reference_localMap_cloud->size() << std::endl;
    if (reference_localMap_cloud->size() == 0)
    {
        throw std::runtime_error("reference_localMap_cloud not init - no points");
    }
    // Set optimization parameters
    LevenbergMarquardtParams params;
    params.setMaxIterations(100);     // 100
    params.setRelativeErrorTol(1e-3); // 1e-3
    params.setVerbosity("ERROR");     // TERMINATION  ERROR

    bool rv = 0.;

    if (!prev_graph_exist)
    {
        NonlinearFactorGraph graph;
        Values initial_values;

        buildGraph(lidar_lines, line_poses, refference_kdtree, reference_localMap_cloud,
                   prev_segment_init, prev_segment, kdtree_prev_segment,
                   cloud_pub, normals_pub,
                   graph, initial_values,
                   overlap_size, threshold_nn, p2p, p2plane);

        // Optimize the graph
        // LevenbergMarquardtOptimizer optimizer(graph, initial_values, params); //
        // Values optimized_values = optimizer.optimize();

        // // optimized_values.print("Optimized Results:\n");
        // //  Retrieve optimized poses
        // //  std::vector<Sophus::SE3> optimized_poses;
        // for (size_t i = 0; i < line_poses.size(); i++)
        // {
        //     gtsam::Key pose_key = useX ? X(i) : Z(i);
        //     Pose3 optimized_pose = optimized_values.at<Pose3>(pose_key);
        //     M3D R = optimized_pose.rotation().matrix();
        //     V3D t = optimized_pose.translation();
        //     line_poses[i] = Sophus::SE3(R, t);
        //     // optimized_poses.emplace_back(Sophus::SE3(R, t));
        // }

        prev_graph_exist = true;
        prev_optimized_values = initial_values; // keep the raw data
        // prev_optimized_values = optimized_values; // keep the optimized values
        prev_graph = graph;

        // rv = optimizer.error();

        return rv;
    }
    else
    {
        // int prev_last_idx = line_poses.size();
        // std::cout << "Use the prev values with size:" << overlap_size << ", prev_last_idx:" << prev_last_idx << std::endl;

        NonlinearFactorGraph graph;
        Values initial_values;

        for (int iter = 0; iter < run_iterations; iter++)
        {
            if (flg_exit)
                break;

            std::cout << "Run iteration:" << iter << std::endl;
            graph.resize(0);        // Clears all factors from the graph
            initial_values.clear(); // Clears all values

            buildGraph(lidar_lines, line_poses, refference_kdtree, reference_localMap_cloud,
                       prev_segment_init, prev_segment, kdtree_prev_segment,
                       cloud_pub, normals_pub,
                       graph, initial_values,
                       overlap_size, threshold_nn, p2p, p2plane);

            NonlinearFactorGraph merged_graph;
            Values merged_values;

            // std::cout << "mergeXandZGraphs" << std::endl;
            int total_poses = line_poses.size();
            mergeXandZGraphs(merged_graph, merged_values,
                             prev_graph, prev_optimized_values,
                             graph, initial_values,
                             useX, overlap_size, total_poses);

            // for debugging
            debugGraphs(prev_optimized_values, initial_values, prev_pub, curr_pub);

            // Optimize the graph
            LevenbergMarquardtOptimizer optimizer(merged_graph, merged_values, params); //
            Values optimized_values = optimizer.optimize();

            // optimized_values.print("Optimized Results:\n");
            //  Retrieve optimized poses

            Values re_optimized_prev;
            for (size_t i = 0; i < total_poses; i++)
            {
                gtsam::Key curr_pose_key = useX ? X(i) : Z(i);
                if (optimized_values.exists(curr_pose_key))
                {
                    Pose3 optimized_pose = optimized_values.at<Pose3>(curr_pose_key);
                    line_poses[i] = Sophus::SE3(optimized_pose.rotation().matrix(), optimized_pose.translation());
                }
                else
                {
                    Symbol symbol_key(curr_pose_key);
                    std::cerr << "Optimized pose not found for key: " << symbol_key << std::endl;
                }

                // for the prev segment use the reoptimized values------------------------------
                gtsam::Key prev_pose_key = useX ? Z(i) : X(i);
                if (optimized_values.exists(curr_pose_key))
                {
                    Pose3 prev_optimized_pose = optimized_values.at<Pose3>(prev_pose_key);
                    re_optimized_prev.insert(prev_pose_key, prev_optimized_pose);
                }
                else
                {
                    Symbol symbol_key(prev_pose_key);
                    std::cerr << "Re-Optimized pose not found for key: " << symbol_key << std::endl;
                }
            }
            prev_optimized_values = re_optimized_prev;

            // std::cout << "\bIteraion:" << iter << ", error:" << optimizer.error() << std::endl;
            rv = optimizer.error();
        }

        prev_graph_exist = true;
        prev_optimized_values = initial_values; // keep the raw data
        prev_graph = graph;

        return rv;
    }
}

// #define use_spline

#ifdef use_spline

// Spline basis matrix for uniform cubic B-spline
const Eigen::Matrix4d C = (Eigen::Matrix4d() << 6. / 6., 0., 0., 0.,
                           5. / 6., 3. / 6., -3. / 6., 1. / 6.,
                           1. / 6., 3. / 6., 3. / 6., -2. / 6.,
                           0., 0., 0., 1. / 6.)
                              .finished();

class CubicSpline
{
public:
    using SE3Type = Sophus::SE3;
    using SE3DerivType = Eigen::Matrix4d;

    // Constructor to initialize time step and initial time
    CubicSpline(double dt = 1.0, double t0 = 0.0) : dt_(dt), t0_(t0) {}

    // Add a knot (pose) to the spline
    void add_knot(const SE3Type &pose)
    {
        knots_.push_back(pose);
    }

    // Get a knot (pose) from the spline
    SE3Type get_knot(size_t k) const
    {
        return knots_[k];
    }

    // Evaluate the spline at time t and compute the pose and its derivatives
    void evaluate(double t, SE3Type &P) const; //, SE3DerivType &P_prim, SE3DerivType &P_bis

private:
    static int floor_(double x)
    {
        return static_cast<int>(std::floor(x));
    }

    double dt_;                  // Time step
    double t0_;                  // Initial time
    std::vector<SE3Type> knots_; // Vector of SE3 poses as knots
};

void CubicSpline::evaluate(double t, SE3Type &P) const //, SE3DerivType &P_prim, SE3DerivType &P_bis
{
    using Mat4 = Eigen::Matrix4d;
    using Vec4 = Eigen::Matrix<double, 4, 1>;

    assert(dt_ > 0.0 && "CubicSpline::evaluate: Time step (dt) must be greater than zero.");
    assert(knots_.size() >= 4 && "CubicSpline::evaluate: There must be at least 4 knots for spline interpolation.");
    assert(t >= t0_ + dt_ && "CubicSpline::evaluate: Time t must be greater than or equal to the time of the first knot.");
    assert(t < t0_ + dt_ * (knots_.size() - 2) && "CubicSpline::evaluate: Time t must be less than the time of the last knot.");

    // Compute normalized time (offset-aware)
    double s = (t - t0_) / dt_;
    int i = floor_(s);
    double u = s - i;
    int i0 = i - 1;

    double u2 = u * u, u3 = u2 * u, dt_inv = 1.0 / dt_;
    Vec4 B = C * Vec4{1.0, u, u2, u3};
    Vec4 Bd1 = C * Vec4{0.0, 1.0, 2.0 * u, 3.0 * u2} * dt_inv;
    Vec4 Bd2 = C * Vec4{0.0, 0.0, 2.0, 6.0 * u} * dt_inv * dt_inv;

    SE3Type P0 = knots_[i0]; // First knot pose
    P = P0;

    Mat4 A[3], Ad1[3], Ad2[3];

    for (int j : {1, 2, 3})
    {
        SE3Type knot1 = knots_[i0 + j - 1];
        SE3Type knot2 = knots_[i0 + j];
        auto omega = (knot1.inverse() * knot2).log();
        Mat4 omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(B[j] * omega);
        P = P * Aj;
        Mat4 Ajm = Aj.matrix();
        Mat4 Ajd1 = Ajm * omega_hat * Bd1[j];
        Mat4 Ajd2 = Ajd1 * omega_hat * Bd1[j] + Ajm * omega_hat * Bd2[j];
        A[j - 1] = Ajm;
        Ad1[j - 1] = Ajd1;
        Ad2[j - 1] = Ajd2;
    }

    // // Compute the derivatives
    // Mat4 M1 = Ad1[0] * A[1] * A[2] + A[0] * Ad1[1] * A[2] + A[0] * A[1] * Ad1[2];
    // Mat4 M2 = Ad2[0] * A[1] * A[2] + A[0] * Ad2[1] * A[2] + A[0] * A[1] * Ad2[2] +
    //           2.0 * Ad1[0] * Ad1[1] * A[2] + 2.0 * Ad1[0] * A[1] * Ad1[2] +
    //           2.0 * A[0] * Ad1[1] * Ad1[2];

    // P_prim = P0.matrix() * M1;
    // P_bis = P0.matrix() * M2;
}

#endif

//----------------------------------------------------------------------------------------------

struct landmark_new
{
    int map_point_index;         // index of the point from the reference map
    V3D norm;                    // the normal of the plane in global frame
    V3D landmark_point;          // point_on_the_plane;      // point from the map
    double negative_OA_dot_norm; // d parameter of the plane

    double re_proj_error;
    std::vector<int> line_idx; // from which line is seen
    std::vector<int> scan_idx; // from which which point index of the line_idx it is seen
    int landmark_key = 0;

    V3D center;
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

// float robust_kernel = .01; // 1cm used so far

float robust_kernel = .03;

auto landmarks_sigma = 1; // this was used for all the tests so far

bool use_artificial_uncertainty = false;

auto plane_noise_cauchy = gtsam::noiseModel::Robust::Create(
    // gtsam::noiseModel::mEstimator::Cauchy::Create(.2), // Less aggressive than Tukey
    gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // 2cm
    gtsam::noiseModel::Isotropic::Sigma(1, landmarks_sigma));

auto point_noise_cauchy = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(robust_kernel), // Robust kernel for outliers
    gtsam::noiseModel::Isotropic::Sigma(3, sigma_point));

auto plane_noise_cauchy_for_prev_segment = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(.005), //  .005 = .5 cm
    gtsam::noiseModel::Isotropic::Sigma(1, landmarks_sigma));

// used so far
auto odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3(.01, .01, .05), //.01, .01, .02 translation stddev (m): x, y, z
     gtsam::Vector3(.01, .01, .01)                      // rotation stddev (radians): roll, pitch, yaw
     )
        .finished());

auto loose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
    (gtsam::Vector6() << gtsam::Vector3(5., 5., 10.), // translation stddev (m): x, y, z
     gtsam::Vector3(.5, .5, 1.)                       //                      // rotation stddev (radians): roll, pitch, yaw ~5 degrees std in rpy
     )
        .finished());

NonlinearFactorGraph reference_graph;
Values reference_values;
int key = 0, landmark_id_counter = 0, max_size = 200;
bool doneFirstOpt = false;
bool systemInitialized = false;
bool optimize_landmarks = false; // true;

bool batch_optimization = false;   // true;    // batch after every buffer requires re-init
NonlinearFactorGraph global_graph; // keep all factors here for batch
Values global_values;

bool use_local_submap = false; // true;
int history_steps = 20;        // last 20 scans
std::deque<pcl::PointCloud<VUX_PointType>::Ptr> history_scans;
std::deque<std::vector<V3D>> history_scans_ref_nn;
std::deque<Sophus::SE3> history_poses;
pcl::KdTreeFLANN<PointType>::Ptr history_kdtree(new pcl::KdTreeFLANN<PointType>);
bool have_enough_ref_map = false;
int counter = 0;

Pose3 prevPose_;
Sophus::SE3 prevOptimized_pose = Sophus::SE3();
gtsam::ISAM2 isam;

// persistent variable
// Map: V3D (point) -> int (landmark key)
std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> global_seen_landmarks;

void debugGraph(const gtsam::Values &_values, ros::Publisher &pub)
{
    auto _cloud = extractTranslations(_values);

    std::cout << "_cloud:" << _cloud->size() << std::endl;

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = "world";

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(*_cloud, msg);
    msg.header = header;
    pub.publish(msg);
}

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

void resetOptimization()
{
    gtsam::ISAM2Params optParameters;
    // optParameters.relinearizeThreshold = 0.1; //0.1 means if the change in a variable (like pose or velocity) exceeds 0.1 in norm, then it will be relinearized.
    optParameters.relinearizeThreshold = 0.01;

    optParameters.relinearizeSkip = 1;
    isam = gtsam::ISAM2(optParameters);

    gtsam::NonlinearFactorGraph newGraphFactors;
    reference_graph = newGraphFactors;

    gtsam::Values NewGraphValues;
    reference_values = NewGraphValues;

    global_seen_landmarks.clear();
    landmark_id_counter = 0; // Reset to initial value (e.g., 0 or 1)
}

std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> get_Landmarks(
    const pcl::PointCloud<VUX_PointType>::Ptr &scan,                 // scan in sensor frame
    const Sophus::SE3 &T,                                            // init guess
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,       // reference kdtree
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud, // reference cloud
    double threshold_nn = 1.0, bool radius_based = false, bool weighted_mean = false)
{

    // std::unordered_map<int, landmark_new> landmarks_map; // key is index of the point from the map
    std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> landmarks_map;

    // //return landmarks_map; // no landmarks TEST WITHOUT PLANES

    double uncertainty_scale = 15;

    // this can be done in parallel BTW-----------------------------
    for (int i = 0; i < scan->size(); i++)
    {
        V3D p_src(scan->points[i].x, scan->points[i].y, scan->points[i].z);
        V3D p_transformed = T * p_src; // transform the point with the initial guess pose

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
                double negative_OA_dot_norm = -norm.dot(centroid);

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

                        for (int j = 0; j < neighbours; j++) // all the points share the same normal
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

                                tgt.landmark_point = centroid; // centroid
                                // tgt.landmark_point = point_3d;  //first point
                                tgt.key = point_3d;
                                tgt.sigma = sigma_plane;
                                tgt.negative_OA_dot_norm = negative_OA_dot_norm;
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
                                    planes_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                    planes_iterator->second.re_proj_error = curvature; // Fixed: Use curvature (not point_dist[j])
                                }
                            }

                            break; // to keep only the closest neighbour
                        }
                    }
                    else if (linearity > .7) // edge like
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

                            tgt.negative_OA_dot_norm = negative_OA_dot_norm;
                            tgt.sigma = sigma_edge;

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
                                edge_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                edge_iterator->second.re_proj_error = linearity;
                            }
                        }
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
                double negative_OA_dot_norm = -norm.dot(centroid);

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

                                tgt.landmark_point = centroid; // centroid
                                // tgt.landmark_point = point_3d;  //first point
                                tgt.key = point_3d;
                                tgt.sigma = sigma_plane;
                                tgt.negative_OA_dot_norm = negative_OA_dot_norm;
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
                                    planes_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
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

                            tgt.negative_OA_dot_norm = negative_OA_dot_norm;

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
                                edge_iterator->second.negative_OA_dot_norm = negative_OA_dot_norm;
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

void buildReferenceGraph(
    ros::Publisher &_pub,
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_scans, // init segment scans
    std::deque<Sophus::SE3> &line_poses,                                // init segment odometry
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud)
{
    resetOptimization(); // this is only when we use isam

    // 1. Create nodes from initial odometry poses
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Pose3 gtsam_pose(line_poses[i].matrix());

        // Add to initial values
        reference_values.insert(X(i), gtsam_pose);

        // Add prior on the first pose to fix the coordinate frame
        if (i == 0)
        {
            reference_graph.addPrior(X(0), gtsam_pose, prior_noise);
        }

        // Add odometry constraints between consecutive poses
        if (i > 0)
        {
            Sophus::SE3 relative_pose = line_poses[i - 1].inverse() * line_poses[i];
            Pose3 gtsam_relative(relative_pose.matrix());

            reference_graph.emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), gtsam_relative, odometry_noise);
        }

        key = i + 1; // the last key
    }
    prevOptimized_pose = line_poses.back();

    debugGraph(reference_values, _pub);

    if (batch_optimization)
    {
        // Add to global graph and estimate
        global_graph.add(reference_graph);
        global_values.insert(reference_values);
    }

    // Do the initial batch optimization
    isam.update(reference_graph, reference_values);
    // Clear graph and values to avoid reusing same keys
    reference_graph.resize(0);
    reference_values.clear();

    doneFirstOpt = true;
    systemInitialized = true;
}

Sophus::SE3 updateReferenceGraph(
    ros::Publisher &_pub_debug,
    ros::Publisher &_pub_prev, ros::Publisher &_pub_curr,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    pcl::PointCloud<VUX_PointType>::Ptr &scan, // scan in sensor frame
    const Sophus::SE3 &T, Sophus::SE3 &rel_T,  // absolute T, odometry
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud)
{
    if (doneFirstOpt)
    {
        std::cout << "updateReferenceGraph key:" << key << std::endl;

        debugPoint(T.translation(), _pub_prev);

        // this is only in fusion with the prev relative T
        Pose3 _absolute_guess = Pose3(T.matrix());
        reference_graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(key), _absolute_guess, loose_prior_noise));

        auto T_init = T; // set the absolute pose
        // auto T_init = prevOptimized_pose * rel_T; // this added to start from the prev optimized value
        //   it works better without prevOptimized_pose * rel_T for
        //   if we use it it drifts

        Pose3 absolute_gtsam_pose(T_init.matrix());
        Pose3 gtsam_relative(rel_T.matrix());

        reference_values.insert(X(key), absolute_gtsam_pose);
        reference_graph.emplace_shared<BetweenFactor<Pose3>>(X(key - 1), X(key), gtsam_relative, odometry_noise_);

        std::vector<V3D> curr_scan_nn;
        if (true)
        {
            double threshold_nn = 1.0;
            bool radius_based = true; // false;

            // auto nn_init_guess_T = T_init;
            auto nn_init_guess_T = prevOptimized_pose * rel_T;

            // add the landmarks
            // const std::unordered_map<int, landmark_new> &landmarks_map
            const std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> &landmarks_map = get_Landmarks(
                scan, nn_init_guess_T, refference_kdtree, reference_localMap_cloud,
                threshold_nn, radius_based);

            std::cout << "optimize_landmarks:" << optimize_landmarks << std::endl;
            std::cout << "Number of items in landmarks_map: " << landmarks_map.size() << ", global_seen_landmarks: " << global_seen_landmarks.size() << std::endl;

            int added_constraints = 0;

            if (landmarks_map.size() > 5) // at least 5 planes
            {
                if (optimize_landmarks)
                {
                    int constraints_existing = 0, constraints_new = 0;
                    // 2. Add plane landmarks ( as optimizable variables!)
                    for (const auto &[point, land] : landmarks_map)
                    {
                        // Check if the landmark is already tracked
                        auto it = global_seen_landmarks.find(point);
                        if (it != global_seen_landmarks.end())
                        {
                            // Existing landmark: retrieve its key and add constraints
                            constraints_existing++;
                            int l_key = it->second.landmark_key;

                            // std::cout << "Existing landmark L(" << l_key << ") " << std::endl;

                            for (int i = 0; i < land.scan_idx.size(); i++)
                            {
                                added_constraints++;

                                const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                const auto &raw_point = scan->points[p_idx];

                                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                // // Initial guess for landmark position
                                // Point3 landmark_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                //                                reference_localMap_cloud->points[land.map_point_index].y,
                                //                                reference_localMap_cloud->points[land.map_point_index].z);
                                gtsam::Unit3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());

                                // reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm, landmark_point, land.negative_OA_dot_norm,
                                //                                                      true, plane_noise_cauchy);

                                // --- Existing Landmark ---
                                // Retrieve its current estimate from GTSAM's values
                                // gtsam::Point3 existing_landmark = reference_values.at<gtsam::Point3>(L(l_key));

                                // Add constraints(e.g., point - to - plane factors)
                                reference_graph.emplace_shared<PlaneFactor>(
                                    X(key),            // Current pose key
                                    L(l_key),          // Existing landmark key
                                    measured_point,    // Measured point in sensor frame
                                    plane_norm,        // Plane normal
                                    plane_noise_cauchy // Noise model
                                );

                                break;
                            }
                        }
                        else
                        {
                            // New landmark: assign key and add to graph
                            // std::cout << "New landmark at: " << point.transpose() << std::endl;
                            int l_key = landmark_id_counter++;
                            constraints_new++;
                            // std::cout << "New landmark L(" << l_key << ") " << std::endl;
                            auto new_land = land;
                            new_land.landmark_key = l_key;
                            global_seen_landmarks[point] = new_land;

                            for (int i = 0; i < land.scan_idx.size(); i++)
                            {
                                added_constraints++;
                                const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                const auto &raw_point = scan->points[p_idx];

                                Point3 sensor_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                // Initial guess for landmark position
                                Point3 landmark_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                                               reference_localMap_cloud->points[land.map_point_index].y,
                                                               reference_localMap_cloud->points[land.map_point_index].z);

                                gtsam::Unit3 landmark_normal(land.norm.x(), land.norm.y(), land.norm.z()); // Plane normal (fixed or optimized)

                                // std::cout << "Adding L(" << l_key << ") landmark\n"<< std::endl;
                                //  Add prior on landmark (optional, to anchor the first landmark)
                                reference_graph.addPrior(L(l_key), landmark_point, gtsam::noiseModel::Isotropic::Sigma(3, 1));

                                // Add point-to-plane factors (linking poses and landmarks)
                                reference_graph.emplace_shared<PlaneFactor>(
                                    X(key),   // Pose key
                                    L(l_key), // Landmark key
                                    sensor_point,
                                    landmark_normal,
                                    plane_noise_cauchy);

                                // Initialize values (poses + landmarks)
                                reference_values.insert(L(l_key), landmark_point); // Landmark is now part of optimization!

                                break;
                            }
                        }
                    }
                    std::cout << "constraints_existing:" << constraints_existing << ", constraints_new:" << constraints_new << std::endl;
                    debugPoint(T.translation(), _pub_prev);
                }
                else
                {
                    // 2. Add constraints
                    for (const auto &[landmark_id, land] : landmarks_map)
                    {
                        // if (land.seen > 1) // if the landmark is seen from multiple measurements
                        {
                            for (int i = 0; i < land.scan_idx.size(); i++)
                            {
                                if (land.is_plane)
                                {
                                    const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                    const auto &raw_point = scan->points[p_idx];

                                    Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                    // Point3 target_point = Point3(reference_localMap_cloud->points[land.map_point_index].x,
                                    //                              reference_localMap_cloud->points[land.map_point_index].y,
                                    //                              reference_localMap_cloud->points[land.map_point_index].z);
                                    Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
                                    Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());

                                    // if (use_alternative_method_)
                                    //     error = (p_transformed - target_point_).dot(plane_normal_);
                                    // else
                                    //     error = plane_normal_.dot(p_transformed) + negative_OA_dot_norm_;

                                    bool use_alternative_method = true; // this works - the tests was done with this true

                                    // THE BUG WAS SOLVED: ---- THERE IS A BUG HERE SOMEWHERE - SOLVE IT
                                    use_alternative_method = false; // just check if this solves the issue of xy drift

                                    // reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                    //                                                    use_alternative_method, plane_noise_cauchy);

                                    // Robust kernel scale: threshold for downweighting begins
                                    // double cauchy_param = 1.5 * land.sigma;
                                    double cauchy_param = robust_kernel; //.05;

                                    // std::cout<<"plane sigma:"<<land.sigma<<", cauchy_param:"<<cauchy_param<<std::endl;

                                    auto robust_noise = gtsam::noiseModel::Robust::Create(
                                        gtsam::noiseModel::mEstimator::Cauchy::Create(cauchy_param),
                                        gtsam::noiseModel::Isotropic::Sigma(1, 3 * land.sigma));

                                    if (use_artificial_uncertainty)
                                    {
                                        robust_noise = plane_noise_cauchy; // use the artificial one
                                    }

                                    reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                                                       use_alternative_method, robust_noise);

                                    added_constraints++;

                                    //--------check consistency with prev scans-----------------------
                                    auto it = global_seen_landmarks.find(landmark_id); // check if its seen in prev scan
                                    if (it != global_seen_landmarks.end())             // same plane seen in prev scans too
                                    {
                                        // add the connection from curr state to prev scans' landmark
                                        Point3 target_point_prev(it->second.landmark_point.x(), it->second.landmark_point.y(), it->second.landmark_point.z());
                                        Point3 plane_norm_prev(it->second.norm.x(), it->second.norm.y(), it->second.norm.z());
                                        // reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm_prev, target_point_prev, land.negative_OA_dot_norm,
                                        //                                                    use_alternative_method, plane_noise_cauchy);

                                        // cauchy_param = 1.5. * it->second.sigma;
                                        auto prev_robust_noise = gtsam::noiseModel::Robust::Create(
                                            gtsam::noiseModel::mEstimator::Cauchy::Create(cauchy_param),
                                            gtsam::noiseModel::Isotropic::Sigma(1, 3 * it->second.sigma));

                                        if (use_artificial_uncertainty)
                                        {
                                            prev_robust_noise = plane_noise_cauchy; // use the artificial one
                                        }

                                        reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm_prev, target_point_prev, land.negative_OA_dot_norm,
                                                                                           use_alternative_method, prev_robust_noise);
                                    }
                                    else
                                    {
                                        // this landmarks is has not been seet yet,  add it to buffer
                                        global_seen_landmarks[land.key] = land;
                                    }
                                }

                                else if (land.is_edge)
                                {
                                    // std::cout << "found an edge" << std::endl;
                                    const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                    const auto &raw_point = scan->points[p_idx];

                                    Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                    Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
                                    Point3 line_dir(land.edge_direction.x(), land.edge_direction.y(), land.edge_direction.z());

                                    // reference_graph.emplace_shared<PointToLineFactor>(X(key), measured_point, target_point, line_dir, plane_noise_cauchy);

                                    double cauchy_param = robust_kernel; //.05
                                    // std::cout<<"edge sigma:"<<land.sigma<<std::endl;

                                    auto robust_noise = gtsam::noiseModel::Robust::Create(
                                        gtsam::noiseModel::mEstimator::Cauchy::Create(cauchy_param),
                                        gtsam::noiseModel::Isotropic::Sigma(1, 3 * land.sigma));

                                    if (use_artificial_uncertainty)
                                    {
                                        robust_noise = plane_noise_cauchy; // use the artificial one
                                    }
                                    reference_graph.emplace_shared<PointToLineFactor>(X(key), measured_point, target_point, line_dir, robust_noise);

                                    //--------check consistency with prev scans-----------------------
                                    // auto it = global_seen_landmarks.find(landmark_id); // check if its seen in prev scan
                                    // if (it != global_seen_landmarks.end())             // same plane seen in prev scans too
                                    // {
                                    //     // add the connection from curr state to prev scans' landmark
                                    //     Point3 target_point_prev(it->second.landmark_point.x(), it->second.landmark_point.y(), it->second.landmark_point.z());
                                    //     Point3 line_dir_prev(it->second.edge_direction.x(), it->second.edge_direction.y(), it->second.edge_direction.z());

                                    //     reference_graph.emplace_shared<PointToLineFactor>(X(key), measured_point, target_point_prev, line_dir_prev, plane_noise_cauchy);
                                    // }
                                    // else
                                    // {
                                    //     // this landmarks is has not been seet yet,  add it to buffer
                                    //     global_seen_landmarks[land.key] = land;
                                    // }

                                    added_constraints++;
                                }

                                const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                const auto &raw_point = scan->points[p_idx];
                                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
                                reference_graph.emplace_shared<PointToPointFactor>(X(key), measured_point, target_point, point_noise_cauchy);

                                curr_scan_nn.push_back(land.key);

                                break; // means only once
                            }
                        }
                    }

                    // debugGraph(reference_values, _pub_prev);

                    // landmarks_map is the landmarks from the prev iteration/scan
                    // global_seen_landmarks = landmarks_map; //keep only the prev scan
                }
            }

            std::cout << "added_constraints:" << added_constraints << std::endl;

            counter++;
            if (use_local_submap && have_enough_ref_map && counter % 1 == 0) // every 5th scan
            {
                // search the planes and edges landmarks in that scan
                // add them to the graph -

                if (history_scans.size() >= history_steps) // add a condition here so that there is enigh distance from curr to prev
                {
                    // merge prev scans into a cloud
                    pcl::PointCloud<PointType>::Ptr combined_scans(new pcl::PointCloud<PointType>);
                    for (int l = 0; l < history_scans.size(); l++) // for each scan line
                    {
                        const auto &initial_guess = history_poses[l];      // copy of the refined pose
                        for (int i = 0; i < history_scans[l]->size(); i++) // for each point in the scan line
                        {
                            V3D p_src(history_scans[l]->points[i].x, history_scans[l]->points[i].y, history_scans[l]->points[i].z);
                            if (p_src.norm() < 30)
                            {
                                V3D p_transformed = initial_guess * p_src; // transform to world frame

                                PointType p;
                                p.x = p_transformed.x();
                                p.y = p_transformed.y();
                                p.z = p_transformed.z();
                                p.intensity = 1; // weight 1

                                combined_scans->push_back(p);
                            }
                        }
                    }
                    for (int l = 0; l < history_scans_ref_nn.size(); l++) // for each scan line ref nn
                    {
                        for (int i = 0; i < history_scans_ref_nn[l].size(); i++) // for each point in the scan line ref nn
                        {
                            const V3D &p_transformed = history_scans_ref_nn[l][i]; // already in global frame
                            PointType p;
                            p.x = p_transformed.x();
                            p.y = p_transformed.y();
                            p.z = p_transformed.z();
                            p.intensity = 100; // weight 100

                            combined_scans->push_back(p);
                        }
                    }
                    for (int i = 0; i < curr_scan_nn.size(); i++)
                    {
                        const V3D &p_transformed = curr_scan_nn[i]; // already in global frame
                        PointType p;
                        p.x = p_transformed.x();
                        p.y = p_transformed.y();
                        p.z = p_transformed.z();
                        p.intensity = 100; // weight 100

                        combined_scans->push_back(p);
                    }

                    std::cout << "combined_scans:" << combined_scans->size() << std::endl;
                    history_kdtree->setInputCloud(combined_scans);

                    // plot the combined scans
                    if (normals_pub.getNumSubscribers() != 0)
                    {
                        std_msgs::Header header;
                        header.stamp = ros::Time::now();
                        header.frame_id = "world";

                        sensor_msgs::PointCloud2 msg;
                        pcl::toROSMsg(*combined_scans, msg);
                        msg.header = header;
                        _pub_debug.publish(msg);
                    }

                    auto nn_init_guess_T = prevOptimized_pose * rel_T;

                    double threshold_nn = 1.0;
                    bool radius_based = true; // false;

                    bool weighted_mean = true;

                    const std::unordered_map<V3D, landmark_new, Vector3dHash, Vector3dEqual> &history_landmarks_map = get_Landmarks(
                        scan, nn_init_guess_T, history_kdtree, combined_scans,
                        threshold_nn, radius_based, weighted_mean);

                    std::cout << "history_landmarks_map: " << history_landmarks_map.size() << std::endl;

                    // check how many history_landmarks_map  and how many landmarks_map
                    // balance them properly

                    for (const auto &[landmark_id, land] : history_landmarks_map)
                    {
                        for (int i = 0; i < land.scan_idx.size(); i++)
                        {
                            if (land.is_plane)
                            {
                                const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                const auto &raw_point = scan->points[p_idx];

                                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
                                Point3 plane_norm(land.norm.x(), land.norm.y(), land.norm.z());

                                // if (use_alternative_method_)
                                //     error = (p_transformed - target_point_).dot(plane_normal_);
                                // else
                                //     error = plane_normal_.dot(p_transformed) + negative_OA_dot_norm_;

                                bool use_alternative_method = true; // this works - the tests was done with this true

                                // THE BUG WAS SOLVED: ---- THERE IS A BUG HERE SOMEWHERE - SOLVE IT
                                // use_alternative_method = false; // just check if this solves the issue of xy drift

                                reference_graph.emplace_shared<PointToPlaneFactor>(X(key), measured_point, plane_norm, target_point, land.negative_OA_dot_norm,
                                                                                   use_alternative_method, plane_noise_cauchy_for_prev_segment);

                                added_constraints++;
                            }

                            else if (land.is_edge)
                            {
                                // std::cout << "found an edge" << std::endl;
                                const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
                                const auto &raw_point = scan->points[p_idx];

                                Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
                                Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
                                Point3 line_dir(land.edge_direction.x(), land.edge_direction.y(), land.edge_direction.z());

                                reference_graph.emplace_shared<PointToLineFactor>(X(key), measured_point, target_point, line_dir, plane_noise_cauchy_for_prev_segment);
                                added_constraints++;
                            }

                            break; // means only once
                        }
                    }

                    // plot the cloud and the landmakrs
                    if (normals_pub.getNumSubscribers() != 0)
                    {
                        // just a test
                        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
                        for (const auto &[index, land] : history_landmarks_map)
                        {
                            if (land.is_edge)
                            {
                                pcl::PointNormal pt;
                                pt.x = land.landmark_point.x();
                                pt.y = land.landmark_point.y();
                                pt.z = land.landmark_point.z();

                                pt.normal_x = land.edge_direction.x();
                                pt.normal_y = land.edge_direction.y();
                                pt.normal_z = land.edge_direction.z();

                                pt.curvature = -2;

                                cloud_with_normals->push_back(pt);
                            }
                            else if (land.is_plane)
                            {
                                pcl::PointNormal pt;
                                pt.x = land.landmark_point.x();
                                pt.y = land.landmark_point.y();
                                pt.z = land.landmark_point.z();

                                pt.normal_x = land.norm.x();
                                pt.normal_y = land.norm.y();
                                pt.normal_z = land.norm.z();

                                pt.curvature = -1;

                                cloud_with_normals->push_back(pt);
                            }
                        }

                        debug_CloudWithNormals2(cloud_with_normals, cloud_pub, normals_pub);
                    }
                }
            }

            if (normals_pub.getNumSubscribers() != 0 || cloud_pub.getNumSubscribers() != 0)
            {
                pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
                // add the map landmarks
                for (const auto &[index, land] : global_seen_landmarks)
                {
                    // break; // just for now do not show them
                    pcl::PointNormal pt;

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

                    cloud_with_normals->push_back(pt);
                }
                // add the new landmarks
                for (const auto &[index, land] : landmarks_map)
                {
                    if (land.is_edge)
                    {
                        pcl::PointNormal pt;
                        pt.x = land.landmark_point.x();
                        pt.y = land.landmark_point.y();
                        pt.z = land.landmark_point.z();

                        pt.normal_x = land.edge_direction.x();
                        pt.normal_y = land.edge_direction.y();
                        pt.normal_z = land.edge_direction.z();

                        pt.curvature = -2;

                        cloud_with_normals->push_back(pt);
                    }
                    else if (land.is_plane)
                    {
                        pcl::PointNormal pt;
                        pt.x = land.landmark_point.x();
                        pt.y = land.landmark_point.y();
                        pt.z = land.landmark_point.z();

                        pt.normal_x = land.norm.x();
                        pt.normal_y = land.norm.y();
                        pt.normal_z = land.norm.z();

                        pt.curvature = -1;

                        cloud_with_normals->push_back(pt);
                    }
                }

                debug_CloudWithNormals2(cloud_with_normals, cloud_pub, normals_pub);
            }

            if (counter > 1000) // has been updated 300 times
            {
                have_enough_ref_map = true;
            }
        }

        if (batch_optimization)
        {
            // Add to global graph and estimate
            global_graph.add(reference_graph);
            global_values.insert(reference_values);
        }

        // Update ISAM2 with just new data
        isam.update(reference_graph, reference_values);

        isam.update(); // Repeatedly relinearizes and refines

        reference_graph.resize(0);
        reference_values.clear(); // this is required to clean the graph, isam has its copy
        gtsam::Values current_estimate = isam.calculateEstimate();
        prevPose_ = current_estimate.at<gtsam::Pose3>(X(key));

        Eigen::Matrix4d T_last = prevPose_.matrix();
        Sophus::SE3 optimized_pose(T_last.block<3, 3>(0, 0), T_last.block<3, 1>(0, 3));
        prevOptimized_pose = optimized_pose;

        debugPoint(T_last.block<3, 1>(0, 3), _pub_curr);

        if (use_local_submap)
        {
            // keep track of prev data
            if (counter % 5 == 0)
            {
                history_scans.push_back(scan);
                history_poses.push_back(prevOptimized_pose);
                history_scans_ref_nn.push_back(curr_scan_nn);

                if (history_scans.size() > history_steps) // use latest 20 scans
                {
                    history_scans.pop_front();
                    history_poses.pop_front();
                    history_scans_ref_nn.pop_front();
                }
            }
        }

        key++;
        if (key >= max_size)
        {
            if (batch_optimization)
            {
                std::cout << "Performing batch optimization before reset..." << std::endl;

                // Run optimization on the accumulated graph
                std::cout << "Performing batch optimization with LM before reset..." << std::endl;
                gtsam::LevenbergMarquardtParams lmParams;
                // lmParams.setVerbosityLM("SUMMARY");
                lmParams.setVerbosity("ERROR");
                lmParams.maxIterations = 10; // its pretty close already 100;
                lmParams.setRelativeErrorTol(1e-3);
                gtsam::LevenbergMarquardtOptimizer batch_optimizer(global_graph, global_values, lmParams);
                gtsam::Values batch_result = batch_optimizer.optimize();

                // Extract the final pose from batch result
                gtsam::Pose3 last_optimized_pose = batch_result.at<gtsam::Pose3>(X(key - 1));
                Eigen::Matrix4d T_last = last_optimized_pose.matrix();
                prevPose_ = last_optimized_pose;
                prevOptimized_pose = Sophus::SE3(T_last.block<3, 3>(0, 0), T_last.block<3, 1>(0, 3));
                optimized_pose = prevOptimized_pose;

                // Estimate new uncertainty for prior
                gtsam::Marginals marginals(global_graph, batch_result);
                gtsam::Matrix covariance = marginals.marginalCovariance(X(key - 1));
                gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(covariance);

                // Reset iSAM2 and restart from batch result
                resetOptimization();
                gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
                reference_graph.add(priorPose);
                reference_values.insert(X(0), prevPose_);

                isam.update(reference_graph, reference_values);
                reference_graph.resize(0);
                reference_values.clear();

                // Clear the global containers and reset the key
                global_graph.resize(0);
                global_values.clear();

                // Add to global graph and estimate
                global_graph.add(priorPose);
                global_values.insert(X(0), prevPose_);

                key = 1;
            }
            else
            {
                std::cout << "reset graph =============================" << std::endl;

                // get updated noise before reset
                gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(isam.marginalCovariance(X(key - 1)));

                // reset graph
                resetOptimization();
                // add pose
                gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
                reference_graph.add(priorPose);

                // add values
                reference_values.insert(X(0), prevPose_);

                // optimize once
                isam.update(reference_graph, reference_values);
                reference_graph.resize(0);
                reference_values.clear();

                key = 1;
            }
        }

        return optimized_pose;
    }
    else
    {
        throw std::runtime_error("Reference graph not inited...");
    }
}