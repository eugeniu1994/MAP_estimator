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
        P2Plane_local(const Eigen::Vector3d &src_point,
                      const Eigen::Vector3d &target_point,
                      const Eigen::Vector3d &normal)
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

        static ceres::CostFunction *Create(const Eigen::Vector3d &src_point,
                                           const Eigen::Vector3d &target_point,
                                           const Eigen::Vector3d &normal)
        {
            return (new ceres::AutoDiffCostFunction<
                    P2Plane_local, 1, 4, 3>(
                new P2Plane_local(src_point, target_point, normal)));
        }

        Eigen::Vector3d src_point;
        Eigen::Vector3d target_point;
        Eigen::Vector3d normal;
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

double scan2map_ceres(pcl::PointCloud<PointType>::Ptr &src,
                      const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                      const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                      Eigen::Quaterniond &q, V3D &t,
                      const bool &prev_segment_init, const pcl::PointCloud<PointType>::Ptr &prev_segment, const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
                      bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    using namespace registration;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    int points_used_for_registration = 0;

    // Add parameter blocks (with quaternion parameterization)
    problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
    problem.AddParameterBlock(t.data(), 3);

    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane && !p2p)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p && !p2plane)
    {
        std::cout << "Perform p2p" << std::endl;
    }
    for (const auto &raw_point : src->points)
    {
        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
        V3D p_transformed = q * p_src + t;

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

        if (p2plane && p2p)
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
                                 norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    if (planeValid)
                    {
                        if (local_error)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }
                        else
                        {
                            ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }

                        points_used_for_registration++;
                    }
                    else
                    {
                        V3D target_point(
                            reference_localMap_cloud->points[point_idx[0]].x,
                            reference_localMap_cloud->points[point_idx[0]].y,
                            reference_localMap_cloud->points[point_idx[0]].z);

                        // Add residuals to Ceres problem
                        ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        points_used_for_registration++;
                    }
                }
            }
        }
        else if (p2plane && !p2p)
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
                                 norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    if (planeValid)
                    {
                        if (local_error)
                        {
                            V3D target_point(
                                reference_localMap_cloud->points[point_idx[0]].x,
                                reference_localMap_cloud->points[point_idx[0]].y,
                                reference_localMap_cloud->points[point_idx[0]].z);

                            ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }
                        else
                        {
                            ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        }

                        points_used_for_registration++;
                    }
                }
            }
        }
        else if (p2p && !p2plane)
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

                    // Add residuals to Ceres problem
                    ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                    problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                    points_used_for_registration++;
                }
            }
        }
    }

    if (prev_segment_init) // we have the prev segment too
    {
        std::cout << "add the prev segment too" << std::endl;
        for (const auto &raw_point : src->points)
        {
            V3D p_src(raw_point.x, raw_point.y, raw_point.z);
            V3D p_transformed = q * p_src + t;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            if (p2plane && p2p)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = prev_segment->points[point_idx[j]].x;
                            matA0(j, 1) = prev_segment->points[point_idx[j]].y;
                            matA0(j, 2) = prev_segment->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                     norm(1) * prev_segment->points[point_idx[j]].y +
                                     norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            if (local_error)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }
                            else
                            {
                                ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }

                            points_used_for_registration++;
                        }
                        else
                        {
                            V3D target_point(
                                prev_segment->points[point_idx[0]].x,
                                prev_segment->points[point_idx[0]].y,
                                prev_segment->points[point_idx[0]].z);

                            // Add residuals to Ceres problem
                            ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                            problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            points_used_for_registration++;
                        }
                    }
                }
            }
            else if (p2plane && !p2p)
            {
                std::vector<int> point_idx(5);
                std::vector<float> point_dist(5);
                if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[4] < threshold_nn) // not too far
                    {
                        Eigen::Matrix<double, 5, 3> matA0;
                        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                        for (int j = 0; j < 5; j++)
                        {
                            matA0(j, 0) = prev_segment->points[point_idx[j]].x;
                            matA0(j, 1) = prev_segment->points[point_idx[j]].y;
                            matA0(j, 2) = prev_segment->points[point_idx[j]].z;
                        }

                        // find the norm of plane
                        V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                        double negative_OA_dot_norm = 1 / norm.norm();
                        norm.normalize();

                        bool planeValid = true;
                        for (int j = 0; j < 5; j++)
                        {
                            if (fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                     norm(1) * prev_segment->points[point_idx[j]].y +
                                     norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .2)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid)
                        {
                            if (local_error)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

                                ceres::CostFunction *cost_function = P2Plane_local::Create(p_src, target_point, norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }
                            else
                            {
                                ceres::CostFunction *cost_function = P2Plane_global::Create(p_src, norm, negative_OA_dot_norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }

                            points_used_for_registration++;
                        }
                    }
                }
            }
            else if (p2p && !p2plane)
            {
                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (kdtree_prev_segment->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
                {
                    if (point_dist[0] < threshold_nn)
                    {
                        V3D target_point(
                            prev_segment->points[point_idx[0]].x,
                            prev_segment->points[point_idx[0]].y,
                            prev_segment->points[point_idx[0]].z);

                        // Add residuals to Ceres problem
                        ceres::CostFunction *cost_function = P2Point::Create(p_src, target_point);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        points_used_for_registration++;
                    }
                }
            }
        }
    }
    // Solve the problem
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization complete. Points used: " << points_used_for_registration << "/" << src->size() << std::endl;
    return summary.final_cost;
}

double scan2map_GN(pcl::PointCloud<PointType>::Ptr &src,
                   const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                   const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                   Eigen::Quaterniond &q, V3D &t, Sophus::SE3 &T_icp,
                   bool p2p = false, bool p2plane = true, bool local_error = true, double threshold_nn = 1.0)
{
    using namespace registration;
    int points_used_for_registration = 0;

    std::cout << "Run ceres solver..." << std::endl;
    if (p2plane && p2p)
    {
        std::cout << "Perform both p2p and p2plane" << std::endl;
    }
    else if (p2plane && !p2p)
    {
        std::cout << "Perform p2plane" << std::endl;
    }
    else if (p2p && !p2plane)
    {
        std::cout << "Perform p2p" << std::endl;
    }

    // Eigen::MatrixXd H;
    // Eigen::VectorXd g;
    // int num_scans = 1;
    // H = Eigen::MatrixXd::Zero(6 * num_scans, 6 * num_scans);
    // g = Eigen::VectorXd::Zero(6 * num_scans);

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };

    Eigen::Matrix6d JTJ_private; // state_size x state_size  (6x6)
    Eigen::Vector6d JTr_private; // state_size x 1           (6x1)
    JTJ_private.setZero();
    JTr_private.setZero();

    for (const auto &raw_point : src->points)
    {
        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
        V3D p_transformed = q * p_src + t;

        // Nearest neighbor search
        PointType search_point;
        search_point.x = p_transformed.x();
        search_point.y = p_transformed.y();
        search_point.z = p_transformed.z();

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

                const V3D residual = p_transformed - target_point;
                Eigen::Matrix3_6d J_r;
                J_r.block<3, 3>(0, 0) = Eye3d;                                  // df/dt
                J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // df/dR

                double w = Weight(residual.squaredNorm());

                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;

                // points_used_for_registration++;
            }
        }
    }

    // Insert into global Hessian
    // H.block<6, 6>(0, 0) = JTJ_private;
    // g.segment<6>(0) = JTr_private;

    // Solve the system H * deltaT = -g
    // Eigen::VectorXd deltaT = H.ldlt().solve(-g); // 6*num_scans   X   1
    // Eigen::VectorXd dx = deltaT.segment<6>(0);   // Eigen::Vector6d

    const Eigen::Vector6d dx = JTJ_private.ldlt().solve(-JTr_private); // translation and rotation perturbations

    const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
    T_icp = estimation * T_icp;

    q = Eigen::Quaterniond(T_icp.so3().matrix());
    t = T_icp.translation();

    return dx.norm();
}

// pcl::KdTreeFLANN<PointType>::Ptr kdtree_prev_segment(new pcl::KdTreeFLANN<PointType>());
// pcl::PointCloud<PointType>::Ptr prev_segment(new pcl::PointCloud<PointType>);
// bool prev_segment_init = false;

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

    std::cout << "Run GN..." << std::endl;
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

    // maybe integrate this in the prev section
    if (prev_segment_init) // we have the prev segment too
    {
        std::cout << "add the prev segment too" << std::endl;

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
                }
                else if (p2plane)
                {
                    std::vector<int> point_idx(5);
                    std::vector<float> point_dist(5);
                    if (kdtree_prev_segment->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                    {
                        if (point_dist[4] < threshold_nn) // not too far
                        {
                            Eigen::Matrix<double, 5, 3> matA0;
                            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                            for (int j = 0; j < 5; j++)
                            {
                                matA0(j, 0) = prev_segment->points[point_idx[j]].x;
                                matA0(j, 1) = prev_segment->points[point_idx[j]].y;
                                matA0(j, 2) = prev_segment->points[point_idx[j]].z;
                            }

                            // find the norm of plane
                            V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                            double negative_OA_dot_norm = 1 / norm.norm();
                            norm.normalize();

                            bool planeValid = true;
                            for (int j = 0; j < 5; j++)
                            {
                                if (fabs(norm(0) * prev_segment->points[point_idx[j]].x +
                                         norm(1) * prev_segment->points[point_idx[j]].y +
                                         norm(2) * prev_segment->points[point_idx[j]].z + negative_OA_dot_norm) > .1)
                                {
                                    planeValid = false;
                                    break;
                                }
                            }

                            if (planeValid)
                            {
                                V3D target_point(
                                    prev_segment->points[point_idx[0]].x,
                                    prev_segment->points[point_idx[0]].y,
                                    prev_segment->points[point_idx[0]].z);

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
                    if (kdtree_prev_segment->nearestKSearch(search_point, 1, point_idx, point_dist) > 0)
                    {
                        if (point_dist[0] < threshold_nn)
                        {
                            V3D target_point(
                                prev_segment->points[point_idx[0]].x,
                                prev_segment->points[point_idx[0]].y,
                                prev_segment->points[point_idx[0]].z);

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
    double normal_length = 5.;     // Length of normal lines

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
        if (point.curvature < 10) // seen less than 3 times
        {
            color.r = 1.0; // not seen enough
            color.g = 0.0;
            color.b = 0.0;
        }
        else
        {
            color.r = 0.0;
            color.g = 1.0; // Green for high curvature
            color.b = 0.0;
        }
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

struct landmark_
{
    int map_point_index;
    V3D norm;
    double negative_OA_dot_norm;
    int seen;
    double re_proj_error;
    std::vector<int> line_idx;
    std::vector<int> scan_idx;
};

std::unordered_map<int, landmark_> get_Correspondences(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    const std::deque<Sophus::SE3> &line_poses_,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment, double threshold_nn = 1.0)
{
    // Containers
    std::unordered_map<int, landmark_> landmarks_map; // key is index of the point from the map
    double good_plane_threshold = .1;

    // Loop through each scan/line
    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        auto &T = line_poses_[l];
        for (int i = 0; i < lidar_lines[l]->size(); i++)
        {
            V3D p_src(lidar_lines[l]->points[i].x, lidar_lines[l]->points[i].y, lidar_lines[l]->points[i].z);
            V3D p_transformed = T * p_src;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            std::vector<int> point_idx(5);
            std::vector<float> point_dist(5);
            if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
            {
                if (point_dist[4] < threshold_nn)
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

                    if (planeValid) // found a good plane
                    {
                        for (int j = 0; j < 5; j++) // all the points share the same normal
                        {
                            auto it = landmarks_map.find(point_idx[j]);
                            if (it == landmarks_map.end()) // does not exist - add
                            {
                                landmark_ tgt;
                                tgt.map_point_index = point_idx[j];
                                tgt.norm = norm;
                                tgt.negative_OA_dot_norm = negative_OA_dot_norm;
                                tgt.seen = 1;
                                tgt.re_proj_error = point_dist[j];
                                tgt.line_idx.push_back(l); // seen from line l
                                tgt.scan_idx.push_back(i); // and point i from this line

                                landmarks_map[point_idx[j]] = tgt;
                            }
                            else // Already exists - increment "seen"
                            {
                                it->second.seen++;                // Increment the seen count
                                it->second.line_idx.push_back(l); // seen from line l
                                it->second.scan_idx.push_back(i); // and point i from this line
                                // if a better normal was found - replace the old one
                                if (it->second.re_proj_error > point_dist[j])
                                {
                                    it->second.norm = norm;
                                    it->second.negative_OA_dot_norm = negative_OA_dot_norm;
                                    it->second.re_proj_error = point_dist[j];
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

using namespace gtsam;
using gtsam::symbol_shorthand::X; // Pose symbols

class LiDARMeasurementFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
    gtsam::Point3 measured_;
    gtsam::Point3 landmark_;

    LiDARMeasurementFactor(gtsam::Key poseKey,
                           const gtsam::Point3 &measured,
                           const gtsam::Point3 &landmark,
                           const gtsam::SharedNoiseModel &model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, poseKey),
          measured_(measured),
          landmark_(landmark) {}

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override
    // gtsam::Vector evaluateError(const gtsam::Pose3 &pose, boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
        // Transform landmark from world frame to LiDAR frame
        gtsam::Point3 predicted = pose.transformTo(landmark_, H);

        // Compute residual: measured - predicted
        return measured_ - predicted;
    }
};

// Custom factor for point-to-point constraints
class PointToPointFactor : public NoiseModelFactor1<Pose3>
{
private:
    Point3 measured_point_; // Point in sensor frame
    Point3 target_point_;   // Corresponding point in world frame
    double weight_;

public:
    PointToPointFactor(Key pose_key,
                       const Point3 &measured_point,
                       const Point3 &target_point,
                       double weight,
                       const SharedNoiseModel &model) : NoiseModelFactor1<Pose3>(model, pose_key),
                                                        measured_point_(measured_point), target_point_(target_point), weight_(weight) {}
    
    //Vector evaluateError(const Pose3 &pose, boost::optional<Matrix &> H = boost::none) const override
    Vector evaluateError(const gtsam::Pose3 &pose, OptionalMatrixType H) const override

    {
        // Transform measured point to world frame
        Point3 world_point = pose.transformFrom(measured_point_, H);

        // Calculate error vector
        Vector3 error = world_point - target_point_;
        //error *= weight_;

        if (H)
        {
            // // Compute Jacobian if requested
            // Matrix36 H_point_wrt_pose;
            // pose.transformFrom(measured_point_, H_point_wrt_pose);
            // (*H) = H_point_wrt_pose * weight_;
        }

        return error;
    }
};

double BA_refinement(
    const std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &lidar_lines,
    std::deque<Sophus::SE3> &line_poses,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
    const bool &prev_segment_init,
    const pcl::PointCloud<PointType>::Ptr &prev_segment,
    const pcl::KdTreeFLANN<PointType>::Ptr &kdtree_prev_segment,
    const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub,
    double threshold_nn = 1.0)
{

    std::unordered_map<int, landmark_> landmarks_map = get_Correspondences(
        lidar_lines,
        line_poses,
        refference_kdtree,
        reference_localMap_cloud,
        prev_segment_init,
        prev_segment,
        kdtree_prev_segment, threshold_nn);

    if (false)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        // for (const auto &land : landmarks_map)
        for (const auto &[index, land] : landmarks_map)
        {
            pcl::PointNormal pt;
            pt.x = reference_localMap_cloud->points[land.map_point_index].x;
            pt.y = reference_localMap_cloud->points[land.map_point_index].y;
            pt.z = reference_localMap_cloud->points[land.map_point_index].z;

            pt.normal_x = land.norm.x();
            pt.normal_y = land.norm.y();
            pt.normal_z = land.norm.z();

            pt.curvature = land.seen;
            // pt.curvature = land.line_idx.size();

            cloud_with_normals->push_back(pt);
        }

        debug_CloudWithNormals(cloud_with_normals, cloud_pub, normals_pub);
    }

    NonlinearFactorGraph graph;
    Values initial_values;

    // 1. Create nodes from initial odometry poses
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Sophus::SE3 pose = line_poses[i];
        Pose3 gtsam_pose(pose.matrix());

        // Add to initial values
        initial_values.insert(X(i), gtsam_pose);

        // Add prior on the first pose to fix the coordinate frame
        if (i == 0)
        {
            auto prior_noise = noiseModel::Diagonal::Sigmas((Vector6() << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
            graph.addPrior(X(0), gtsam_pose, prior_noise);
        }

        // Add odometry constraints between consecutive poses
        if (i > 0)
        {
            Sophus::SE3 relative_pose = line_poses[i - 1].inverse() * line_poses[i];
            Pose3 gtsam_relative(relative_pose.matrix());

            auto odometry_noise = noiseModel::Diagonal::Sigmas((Vector6() << Vector3::Constant(0.05), Vector3::Constant(0.05)).finished());
            graph.emplace_shared<BetweenFactor<Pose3>>(X(i - 1), X(i), gtsam_relative, odometry_noise);
        }
    }

    // //improve from here tomorrow
    //     // Simulated measurements (3D points in the LiDAR frame)
    //     std::vector<std::vector<gtsam::Point3>>
    //         measurements = {
    //             {gtsam::Point3(1.1, 2.1, 3.0), gtsam::Point3(4.0, 5.0, 6.0)}, // Pose 0
    //             {gtsam::Point3(7.0, 8.0, 9.1), gtsam::Point3(1.1, 2.0, 3.2)}  // Pose 1
    //         };

    // // Add measurement factors
    // // Add measurement factors to the graph
    // for (size_t i = 0; i < lidar_poses.size(); ++i)
    // {
    //     for (size_t j = 0; j < measurements[i].size(); ++j)
    //     {
    //         graph.emplace_shared<LiDARMeasurementFactor>(X(i + 1), measurements[i][j], landmarks[j], meas_noise);
    //     }
    // }

    // Set optimization parameters
    // LevenbergMarquardtParams params;
    // params.setMaxIterations(100);
    // params.setRelativeErrorTol(1e-5);
    // params.setVerbosity("ERROR");

    // Optimize the graph
    LevenbergMarquardtOptimizer optimizer(graph, initial_values); //,params
    Values optimized_values = optimizer.optimize();

    // optimized_values.print("Optimized Results:\n");
    //  Retrieve optimized poses
    //  std::vector<Sophus::SE3> optimized_poses;
    for (size_t i = 0; i < line_poses.size(); i++)
    {
        Pose3 optimized_pose = optimized_values.at<Pose3>(X(i));
        M3D R = optimized_pose.rotation().matrix();
        V3D t = optimized_pose.translation();
        line_poses[i] = Sophus::SE3(R, t);
        // optimized_poses.emplace_back(Sophus::SE3(R, t));
    }

    return 0;
}