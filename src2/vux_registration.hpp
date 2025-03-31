#pragma once

#include "DataHandler_vux.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_set>

// #include <pcl/features/normal_estimation_omp.h>
// //#include <pcl/features/normal_estimation.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h> // OpenMP version!
#include <pcl/kdtree/kdtree_flann.h>

namespace Eigen
{
    const int state_size = 6;
    using Matrix6d = Eigen::Matrix<double, state_size, state_size>;
    using Matrix3_6d = Eigen::Matrix<double, 3, state_size>;
    using Vector6d = Eigen::Matrix<double, state_size, 1>;
} // namespace Eigen



void publishPointCloudWithNormals(const pcl::PointCloud<PointType>::Ptr &cloud_with_normals,
                                  const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                  const ros::Publisher &normals_pub)
{
    // --- 2. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.05; // Line width
    normals_marker.color.r = 0.0;
    normals_marker.color.g = 1.0; // full green
    normals_marker.color.b = 0.0;
    normals_marker.color.a = 1.0;

    const double curvature_threshold = .02;
    double normal_length = 5.; // Length of normal lines
    std::cout << "cloud_with_normals->size():" << cloud_with_normals->size() << std::endl;
    for (int i = 0; i < cloud_with_normals->size(); i++)
    {
        const auto &normal = normals->points[i];
        // if (normal.curvature > curvature_threshold)
        // {
        //     continue;
        // }
        const auto &point = cloud_with_normals->points[i];

        geometry_msgs::Point p1, p2;

        // Start of normal (point)
        p1.x = point.x;
        p1.y = point.y;
        p1.z = point.z;
        normals_marker.points.push_back(p1);

        // End of normal (point + normal * length)
        p2.x = point.x + normal_length * normal.normal_x;
        p2.y = point.y + normal_length * normal.normal_y;
        p2.z = point.z + normal_length * normal.normal_z;
        normals_marker.points.push_back(p2);
    }

    normals_pub.publish(normals_marker);
}

// Function to publish point cloud and normals
void publishPointCloudWithNormals(const pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals,
                                  const ros::Publisher &cloud_pub, const ros::Publisher &normals_pub)
{

    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_with_normals, cloud_msg);
    cloud_msg.header.frame_id = "world";
    cloud_pub.publish(cloud_msg);

    // --- 2. Publish Normals as Markers ---
    visualization_msgs::Marker normals_marker;
    normals_marker.header.frame_id = "world";
    normals_marker.type = visualization_msgs::Marker::LINE_LIST;
    normals_marker.action = visualization_msgs::Marker::ADD;
    normals_marker.scale.x = 0.05; // Line width
    normals_marker.color.r = 0.0;
    normals_marker.color.g = 1.0; // full green
    normals_marker.color.b = 0.0;
    normals_marker.color.a = 1.0;

    double normal_length = 7.; // Length of normal lines

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
    }

    normals_pub.publish(normals_marker);
}

void publishJustPoints(const pcl::PointCloud<PointType>::Ptr &cloud_,const ros::Publisher &cloud_pub)
{
    // --- 1. Publish Point Cloud ---
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_, cloud_msg);
    cloud_msg.header.frame_id = "world";
    cloud_pub.publish(cloud_msg);
}

struct LidarDistanceFactor
{
    LidarDistanceFactor(const V3D &curr_point_, const V3D &closest_point_, const Sophus::SE3 &gnss_pose_)
        : curr_point(curr_point_), closest_point(closest_point_), gnss_pose(gnss_pose_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // Convert extrinsic transformation (Scanner to IMU)
        Eigen::Quaternion<T> q_scanner_to_imu(q[3], q[0], q[1], q[2]);
        Eigen::Matrix<T, 3, 1> t_scanner_to_imu(t[0], t[1], t[2]);

        // Transform raw scanner point to GNSS-IMU frame
        Eigen::Matrix<T, 3, 1> point_in_imu = q_scanner_to_imu * curr_point.template cast<T>() + t_scanner_to_imu;

        // Convert GNSS-IMU pose to appropriate type
        Eigen::Matrix<T, 3, 3> R_gnss_imu = gnss_pose.rotation_matrix().template cast<T>();
        Eigen::Matrix<T, 3, 1> t_gnss_imu = gnss_pose.translation().template cast<T>();

        // Georeference point
        Eigen::Matrix<T, 3, 1> point_world = R_gnss_imu * point_in_imu + t_gnss_imu;

        // Compute residual as difference to the closest map point
        residual[0] = point_world.x() - T(closest_point.x());
        residual[1] = point_world.y() - T(closest_point.y());
        residual[2] = point_world.z() - T(closest_point.z());

        return true;
    }

    static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &closest_point_, const Sophus::SE3 &gnss_pose_)
    {
        return new ceres::AutoDiffCostFunction<LidarDistanceFactor, 3, 4, 3>(
            new LidarDistanceFactor(curr_point_, closest_point_, gnss_pose_));
    }

    V3D curr_point;
    V3D closest_point;
    Sophus::SE3 gnss_pose;
};

struct LidarPlaneNormFactor
{

    LidarPlaneNormFactor(const V3D &curr_point_, const V3D &plane_unit_norm_,
                         double negative_OA_dot_norm_, const Sophus::SE3 &gnss_pose_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                                                        negative_OA_dot_norm(negative_OA_dot_norm_), gnss_pose(gnss_pose_) {}

    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // Convert extrinsic transformation (Scanner to IMU)
        Eigen::Quaternion<T> q_scanner_to_imu(q[3], q[0], q[1], q[2]);
        Eigen::Matrix<T, 3, 1> t_scanner_to_imu(t[0], t[1], t[2]);

        // Convert GNSS-IMU pose to appropriate type
        Eigen::Matrix<T, 3, 3> R_gnss_imu = gnss_pose.rotation_matrix().template cast<T>();
        Eigen::Matrix<T, 3, 1> t_gnss_imu = gnss_pose.translation().template cast<T>();

        // Transform raw scanner point to GNSS-IMU frame
        Eigen::Matrix<T, 3, 1> point_in_imu = q_scanner_to_imu * curr_point.template cast<T>() + t_scanner_to_imu;

        // Georeference point
        Eigen::Matrix<T, 3, 1> point_world = R_gnss_imu * point_in_imu + t_gnss_imu;

        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        residual[0] = norm.dot(point_world) + T(negative_OA_dot_norm);

        return true;
    }

    static ceres::CostFunction *Create(const V3D &curr_point_, const V3D &plane_unit_norm_,
                                       const double negative_OA_dot_norm_, const Sophus::SE3 &gnss_pose_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneNormFactor, 1, 4, 3>(
            new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, gnss_pose_)));
    }

    V3D curr_point;
    V3D plane_unit_norm;
    double negative_OA_dot_norm;
    Sophus::SE3 gnss_pose;
};

int computeScanDerivatives(
    const pcl::PointCloud<VUX_PointType>::Ptr &scan,
    const pcl::PointCloud<PointType>::Ptr &ref_map_cloud,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    double threshold_nn,
    Eigen::Matrix6d &H,
    Eigen::Vector6d &g)
{
    Eigen::Matrix6d JTJ_private; // state_size x state_size  (6x6)
    Eigen::Vector6d JTr_private; // state_size x 1           (6x1)
    JTJ_private.setZero();
    JTr_private.setZero();

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };

    int nn_found = 0;
    for (const auto &raw_point : scan->points)
    {
        PointType search_point;
        search_point.x = raw_point.x;
        search_point.y = raw_point.y;
        search_point.z = raw_point.z;

        bool p2plane = false;
        if (p2plane)
        {
            // to be implemented
        }
        else
        {
            std::vector<int> point_idx(1);
            std::vector<float> point_dist(1);
            if (refference_kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
            {
                if (point_dist[0] < threshold_nn) // 1
                {
                    V3D src(raw_point.x, raw_point.y, raw_point.z);

                    V3D tgt(ref_map_cloud->points[point_idx[0]].x, ref_map_cloud->points[point_idx[0]].y, ref_map_cloud->points[point_idx[0]].z);

                    const V3D residual = src - tgt;
                    Eigen::Matrix3_6d J_r;
                    J_r.block<3, 3>(0, 0) = Eye3d;                        // df/dt
                    J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(src); // df/dR

                    // double w = 1.; // to be handled later
                    double w = Weight(residual.squaredNorm());

                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;

                    nn_found++;
                }
            }
        }
    }
    std::cout << "Found " << nn_found << "/" << scan->size() << " neighbours" << std::endl;

    H = JTJ_private;
    g = JTr_private;

    return nn_found;
}

struct VuxPlanes
{
    VuxPlanes(const V3D &curr_point_, const V3D &plane_unit_norm_,
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
                VuxPlanes, 1, 4, 3>(
            new VuxPlanes(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
    }

    V3D curr_point;
    V3D plane_unit_norm;
    double negative_OA_dot_norm;
};

struct VuxPlanes_local
{
    VuxPlanes_local(const Eigen::Vector3d &src_point,
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
                VuxPlanes_local, 1, 4, 3>(
            new VuxPlanes_local(src_point, target_point, normal)));
    }

    Eigen::Vector3d src_point;
    Eigen::Vector3d target_point;
    Eigen::Vector3d normal;
};

// Struct to compute point-to-point residuals
struct VuxP2P_local
{
    VuxP2P_local(const V3D &curr_point_, const V3D &closest_point_)
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
                VuxP2P_local, 3, 4, 3>(
            new VuxP2P_local(curr_point_, closest_point_)));
    }

    V3D curr_point;
    V3D closest_point;
};

// This enforces smoothness by penalizing large changes in pose between consecutive scans.
struct MotionRegularizationCost
{
    MotionRegularizationCost(const Eigen::Quaterniond &q_init, const Eigen::Vector3d &t_init)
        : q_init(q_init), t_init(t_init) {}

    template <typename T>
    bool operator()(const T *const q_i, const T *const t_i,
                    const T *const q_j, const T *const t_j,
                    T *residuals) const
    {

        // Map quaternion and translation parameters
        Eigen::Quaternion<T> q_i_(q_i[3], q_i[0], q_i[1], q_i[2]);
        Eigen::Quaternion<T> q_j_(q_j[3], q_j[0], q_j[1], q_j[2]);
        Eigen::Matrix<T, 3, 1> t_i_(t_i[0], t_i[1], t_i[2]);
        Eigen::Matrix<T, 3, 1> t_j_(t_j[0], t_j[1], t_j[2]);

        // Relative transformation between scans i and j
        Eigen::Quaternion<T> dq = q_i_.conjugate() * q_j_;
        Eigen::Matrix<T, 3, 1> dt = q_i_.conjugate() * (t_j_ - t_i_);

        // Expected relative pose from the initial guess
        Eigen::Quaternion<T> q_rel = q_init.cast<T>();
        Eigen::Matrix<T, 3, 1> t_rel = t_init.cast<T>();

        // --- Rotation Error (Use SO(3) Logarithm) ---
        Eigen::AngleAxis<T> angle_axis(dq * q_rel.inverse());
        Eigen::Matrix<T, 3, 1> rotation_error = angle_axis.angle() * angle_axis.axis();

        // --- Translation Error (Direct Subtraction) ---
        Eigen::Matrix<T, 3, 1> translation_error = dt - t_rel;

        // Populate residuals: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        residuals[0] = rotation_error.x();
        residuals[1] = rotation_error.y();
        residuals[2] = rotation_error.z();
        residuals[3] = translation_error.x();
        residuals[4] = translation_error.y();
        residuals[5] = translation_error.z();

        return true;
    }

    // Factory method to create the cost function
    static ceres::CostFunction *Create(const Eigen::Quaterniond &q_init, const Eigen::Vector3d &t_init)
    {
        return (new ceres::AutoDiffCostFunction<
                MotionRegularizationCost, 6, 4, 3, 4, 3>(
            new MotionRegularizationCost(q_init, t_init)));
    }

    Eigen::Quaterniond q_init;
    Eigen::Vector3d t_init;
};

double joint_registration(const std::vector<std::vector<V3D>> &lidar_lines,
                          std::vector<Eigen::Quaterniond> &q_params,
                          std::vector<Eigen::Vector3d> &t_params,
                          const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                          const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                          const double threshold_nn, bool p2p = true, bool local_error = true)
{

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);

    int points_used_for_registration = 0;

    // Loop through each scan/line
    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        Eigen::Quaterniond &q = q_params[l];
        Eigen::Vector3d &t = t_params[l];

        // Add parameter blocks (with quaternion parameterization)
        problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
        problem.AddParameterBlock(t.data(), 3);

        // Transform all points once with initial guess to avoid repetitive transformations
        for (const auto &raw_point : lidar_lines[l])
        {
            V3D p_transformed = q * raw_point + t;

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            if (p2p) // point to point
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
                        ceres::CostFunction *cost_function = VuxP2P_local::Create(raw_point, target_point);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                        points_used_for_registration++;
                    }
                }
            }
            else // point to plane,
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
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
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

                                ceres::CostFunction *cost_function = VuxPlanes_local::Create(raw_point, target_point, norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }
                            else
                            {
                                ceres::CostFunction *cost_function = VuxPlanes::Create(raw_point, norm, negative_OA_dot_norm);
                                problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                            }

                            points_used_for_registration++;
                        }
                    }
                }
            }
        }

        // if (l < lidar_lines.size() - 1)
        // {
        //     // Compute relative pose using initial guess
        //     Eigen::Quaterniond q_rel = q_params[l].inverse() * q_params[l + 1];
        //     V3D t_rel = q_params[l].inverse() * (t_params[l + 1] - t_params[l]);

        //     ceres::CostFunction *motion_cost = MotionRegularizationCost::Create(q_rel, t_rel);
        //     problem.AddResidualBlock(motion_cost, loss_function,
        //                              q_params[l].coeffs().data(), t_params[l].data(),
        //                              q_params[l + 1].coeffs().data(), t_params[l + 1].data());
        // }
    }

    // Solve the problem
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_SCHUR;
    options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "Optimization complete. Points used: " << points_used_for_registration << std::endl;

    return summary.final_cost;
}

struct valid_tgt
{
    int map_point_index;
    V3D norm;
    int seen;
    double negative_OA_dot_norm;
    std::vector<int> line_idx;
    std::vector<int> scan_idx;
};

// Bundle Adjustment: Joint Scan Registration
double BA(const std::vector<std::vector<V3D>> &lidar_lines,
          std::vector<Eigen::Quaterniond> &q_params,
          std::vector<Eigen::Vector3d> &t_params,
          const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
          const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
          std::vector<valid_tgt> &landmarks,
          const double threshold_nn, bool p2p = false, bool local_error = true)
{

    int points_used_for_registration = 0;

    /*
        we need to enforce that the same plane is seen from multiple scans
        iterate the scans - and search the NN for each point
        map landmarks (point, normal) that are found as NN - good tgt points with normal

        if the same landmakr is seen multiple times
        define a cost function to adjust the sensor pose - so that to minimize the error


        BA should be don in the following way
        when I compute the normals - I also have the src points to that
    */

    // map landmarks (point, normal) that are found as NN
    // std::vector<valid_tgt> landmarks;

    // Containers
    std::unordered_map<int, valid_tgt> landmarks_map;

    double good_plane_threshold = .02; // 2cm

    std::cout << "\nStart Landmark association..." << std::endl;
    // Loop through each scan/line
    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        Eigen::Quaterniond &q = q_params[l];
        Eigen::Vector3d &t = t_params[l];

        // Transform all points once with initial guess to avoid repetitive transformations
        // for (const auto &raw_point : lidar_lines[l])
        for (int p_idx = 0; p_idx < lidar_lines[l].size(); p_idx++)
        {
            const auto &raw_point = lidar_lines[l][p_idx];
            V3D p_transformed = q * raw_point + t; // transfrom with init guess to search for NN

            // Nearest neighbor search
            PointType search_point;
            search_point.x = p_transformed.x();
            search_point.y = p_transformed.y();
            search_point.z = p_transformed.z();

            if (p2p)
            {
                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (refference_kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0)
                {
                    if (point_dist[0] < threshold_nn) // not too far
                    {
                        auto it = landmarks_map.find(point_idx[0]);
                        if (it == landmarks_map.end()) // does not exist → add
                        {
                            valid_tgt tgt;
                            tgt.map_point_index = point_idx[0];
                            tgt.norm = V3D(0, 0, 1);
                            tgt.seen = 1;
                            tgt.negative_OA_dot_norm = 0;
                            tgt.line_idx.push_back(l);     // seen from line l
                            tgt.scan_idx.push_back(p_idx); // and point p_idx from this line

                            landmarks_map[point_idx[0]] = tgt; // Directly use the target object to insert
                        }
                        else // Already exists → increment "seen"
                        {
                            it->second.seen++;                    // Increment the seen count
                            it->second.line_idx.push_back(l);     // seen from line l
                            it->second.scan_idx.push_back(p_idx); // and point p_idx from this line
                        }
                    }
                }
            }
            else
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
                                     norm(2) * reference_localMap_cloud->points[point_idx[j]].z + negative_OA_dot_norm) > good_plane_threshold)
                            {
                                planeValid = false;
                                break;
                            }
                        }

                        if (planeValid) // found a good plane
                        {
                            for (int j = 0; j < 1; j++) // 5
                            {
                                auto it = landmarks_map.find(point_idx[j]);
                                if (it == landmarks_map.end()) // does not exist → add
                                {
                                    valid_tgt tgt;
                                    tgt.map_point_index = point_idx[j];
                                    tgt.norm = norm;
                                    tgt.seen = 1;
                                    tgt.negative_OA_dot_norm = negative_OA_dot_norm;
                                    tgt.line_idx.push_back(l);     // seen from line l
                                    tgt.scan_idx.push_back(p_idx); // and point p_idx from this line

                                    landmarks_map[point_idx[j]] = tgt; // Directly use the target object to insert
                                }
                                else // Already exists → increment "seen"
                                {
                                    it->second.seen++;                    // Increment the seen count
                                    it->second.line_idx.push_back(l);     // seen from line l
                                    it->second.scan_idx.push_back(p_idx); // and point p_idx from this line
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (const auto &[index, tgt] : landmarks_map)
    {
        landmarks.push_back(tgt);
        // std::cout << "Point index: " << index
        //           << ", Normal: [" << tgt.norm.x() << ", " << tgt.norm.y() << ", " << tgt.norm.z() << "]"
        //           << ", Seen: " << tgt.seen << std::endl;
    }

    std::cout << "Finish Landmark association, landmarks:" << landmarks.size() << std::endl;

    // return 0;

    std::cout << "Start BA ..." << std::endl;
    // formulate the BA here--------------------------------------------
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.5);

    for (size_t l = 0; l < lidar_lines.size(); l++)
    {
        Eigen::Quaterniond &q = q_params[l];
        Eigen::Vector3d &t = t_params[l];

        // Add parameter blocks (with quaternion parameterization)
        problem.AddParameterBlock(q.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
        problem.AddParameterBlock(t.data(), 3);
    }

    // for (const auto &land : landmarks) // for each landmark
    for (const auto &[index, land] : landmarks_map)
    {
        if (land.seen > 2) // seen more than once
        {
            for (int i = 0; i < land.seen; i++)
            {
                const auto &l = land.line_idx[i];     // point from line l
                const auto &p_idx = land.scan_idx[i]; // at index p_idx

                Eigen::Quaterniond &q = q_params[l];
                Eigen::Vector3d &t = t_params[l];

                const auto &raw_point = lidar_lines[l][p_idx]; //error here - should be transformed 

                if (p2p)
                {
                    V3D target_point(
                        reference_localMap_cloud->points[land.map_point_index].x,
                        reference_localMap_cloud->points[land.map_point_index].y,
                        reference_localMap_cloud->points[land.map_point_index].z);
                    // Add residuals to Ceres problem
                    ceres::CostFunction *cost_function = VuxP2P_local::Create(raw_point, target_point);
                    problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                }
                else
                {
                    if (local_error)
                    {
                        V3D target_point(
                            reference_localMap_cloud->points[land.map_point_index].x,
                            reference_localMap_cloud->points[land.map_point_index].y,
                            reference_localMap_cloud->points[land.map_point_index].z);

                        ceres::CostFunction *cost_function = VuxPlanes_local::Create(raw_point, target_point, land.norm);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                    }
                    else
                    {
                        ceres::CostFunction *cost_function = VuxPlanes::Create(raw_point, land.norm, land.negative_OA_dot_norm);
                        problem.AddResidualBlock(cost_function, loss_function, q.coeffs().data(), t.data());
                    }
                }

                points_used_for_registration++;
            }
        }
    }
    std::cout << "Optimization complete. Points used: " << points_used_for_registration << std::endl;

    // return 0;

    // Solve the problem
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;

    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << "Optimization complete. Points used: " << points_used_for_registration << std::endl;

    return summary.final_cost;
}




// V3D P0(reference_localMap_cloud->points[point_idx[0]].x,
//        reference_localMap_cloud->points[point_idx[0]].y,
//        reference_localMap_cloud->points[point_idx[0]].z);

// double sin_threshold = std::sin(5.0 * M_PI / 180.0); // 5 degrees threshold
// for (int j = 1; j < 5; j++)
// {
//     V3D P(reference_localMap_cloud->points[point_idx[j]].x,
//           reference_localMap_cloud->points[point_idx[j]].y,
//           reference_localMap_cloud->points[point_idx[j]].z);

//     V3D v = P - P0;                                     // Vector from P0 to P
//     double projection = norm.dot(v);                    // Perpendicular distance to the plane
//     double sin_theta = std::abs(projection) / v.norm(); // Sine of the angle

//     if (sin_theta > sin_threshold)
//     {
//         planeValid = false;
//         break;
//     }
// }

/*

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <deque>
#include <iostream>

using V3D = Eigen::Vector3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Perform joint ICP to align multiple scans to a reference map
void jointICP(
    const pcl::PointCloud<VUX_PointType>::Ptr &reference_map,
    std::deque<pcl::PointCloud<VUX_PointType>::Ptr> &scans,
    std::deque<Sophus::SE3> &initial_guesses,
    int max_iterations = 100, double threshold_nn = 1.0) {

    pcl::KdTreeFLANN<VUX_PointType> kdtree;
    kdtree.setInputCloud(reference_map);

    int num_scans = scans.size();
    for (int iter_num = 0; iter_num < max_iterations; iter_num++) {
        std::cout << "Iteration " << iter_num << std::endl;

        MatrixXd H = MatrixXd::Zero(6 * num_scans, 6 * num_scans);
        VectorXd g = VectorXd::Zero(6 * num_scans);
        int total_points_found = 0;

        // Iterate over each scan
        for (int l = 0; l < num_scans; l++) {
            Eigen::Matrix6d JTJ_private = Eigen::Matrix6d::Zero();
            Eigen::Vector6d JTr_private = Eigen::Vector6d::Zero();
            int nn_found = 0;

            for (const auto &raw_point : scans[l]->points) {
                V3D src(raw_point.x, raw_point.y, raw_point.z);
                V3D transformed_point = initial_guesses[l] * src;

                VUX_PointType search_point;
                search_point.x = transformed_point.x();
                search_point.y = transformed_point.y();
                search_point.z = transformed_point.z();

                std::vector<int> point_idx(1);
                std::vector<float> point_dist(1);
                if (kdtree.nearestKSearch(search_point, 1, point_idx, point_dist) > 0 && point_dist[0] < threshold_nn) {
                    V3D tgt(reference_map->points[point_idx[0]].x, reference_map->points[point_idx[0]].y, reference_map->points[point_idx[0]].z);
                    V3D residual = transformed_point - tgt;

                    Eigen::Matrix3_6d J_r;
                    J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                    J_r.block<3, 3>(0, 3) = -Sophus::SO3::hat(transformed_point);

                    JTJ_private.noalias() += J_r.transpose() * J_r;
                    JTr_private.noalias() += J_r.transpose() * residual;
                    nn_found++;
                }
            }

            total_points_found += nn_found;
            if (nn_found > 0) {
                H.block<6, 6>(l * 6, l * 6) = JTJ_private;
                g.segment<6>(l * 6) = JTr_private;
            }
        }

        // Solve the system H * deltaT = -g
        VectorXd deltaT = H.ldlt().solve(-g);

        // Apply the corrections to each scan's initial guess
        double max_update = 0.0;
        for (int l = 0; l < num_scans; l++) {
            Eigen::Vector6d dTi = deltaT.segment<6>(l * 6);
            const Sophus::SE3 estimation = Sophus::SE3::exp(dTi);
            initial_guesses[l] = estimation * initial_guesses[l];
            max_update = std::max(max_update, dTi.norm());
        }

        std::cout << "Total Points Found: " << total_points_found << ", Max Update Norm: " << max_update << std::endl;

        if (max_update < 0.001) {
            std::cout << "Converged in " << iter_num + 1 << " iterations." << std::endl;
            break;
        }
    }
}


*/
