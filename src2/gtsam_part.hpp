
#include "DataHandler_vux.hpp"

#pragma once

#include <pcl/kdtree/kdtree_flann.h>

#include <gtsam/slam/expressions.h>

#include <gtsam/inference/Key.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

// TargetFrame   PointType VUX_PointType
// const typename pcl::PointCloud<PointType>::Ptr& reference_map,
// const typename pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
// const typename pcl::PointCloud<VUX_PointType>::Ptr& scan,


using namespace gtsam;

// Convert Sophus::SE3 to GTSAM Pose3
gtsam::Pose3 SE3ToGtsamPose3(const Sophus::SE3 &se3)
{
    Eigen::Matrix4d T = se3.matrix();
    return gtsam::Pose3(gtsam::Rot3(T.topLeftCorner<3, 3>()), gtsam::Point3(T(0, 3), T(1, 3), T(2, 3)));
}


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
    //gtsam::Vector evaluateError(const gtsam::Pose3 &pose, boost::optional<gtsam::Matrix &> H = boost::none) const override
    {
        // Transform landmark from world frame to LiDAR frame
        gtsam::Point3 predicted = pose.transformTo(landmark_, H);

        // Compute residual: measured - predicted
        return measured_ - predicted;
    }
};

using gtsam::symbol_shorthand::L; // Landmark (fixed)
using gtsam::symbol_shorthand::X; // Pose (x,y,z,roll,pitch,yaw)

// test this
void my_function()
{
    // Noise models
    auto pose_noise = gtsam::noiseModel::Isotropic::Sigma(6, 0.01); // Pose noise
    auto meas_noise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // Measurement noise

    // Initialize graph and values
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimate;

    // ====== Step 1: Add Prior on First Pose (Anchor) ======
    gtsam::Pose3 initial_pose(Eigen::Matrix4d::Identity());
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), initial_pose, pose_noise);
    initial_estimate.insert(X(0), initial_pose);

    // ====== Step 2: Add Fixed Landmarks ======
    std::vector<gtsam::Point3> landmarks = {
        gtsam::Point3(1.0, 2.0, 3.0),
        gtsam::Point3(4.0, 5.0, 6.0),
        gtsam::Point3(7.0, 8.0, 9.0)};

    // ====== Step 3: Add Poses and Measurements ======
    // Simulated poses
    std::vector<gtsam::Pose3> lidar_poses = {
        gtsam::Pose3(gtsam::Rot3::RzRyRx(0.1, 0.2, 0.3), gtsam::Point3(1.0, 0.0, 0.0)),
        gtsam::Pose3(gtsam::Rot3::RzRyRx(-0.1, 0.1, -0.2), gtsam::Point3(0.0, 1.0, 0.0))};

    // Add initial estimates for LiDAR poses
    for (size_t i = 0; i < lidar_poses.size(); ++i)
    {
        initial_estimate.insert(X(i + 1), lidar_poses[i]);
    }

    // Simulated measurements (3D points in the LiDAR frame)
    std::vector<std::vector<gtsam::Point3>> measurements = {
        {gtsam::Point3(1.1, 2.1, 3.0), gtsam::Point3(4.0, 5.0, 6.0)}, // Pose 0
        {gtsam::Point3(7.0, 8.0, 9.1), gtsam::Point3(1.1, 2.0, 3.2)}  // Pose 1
    };

    // Add measurement factors
    // Add measurement factors to the graph
    for (size_t i = 0; i < lidar_poses.size(); ++i)
    {
        for (size_t j = 0; j < measurements[i].size(); ++j)
        {
            graph.emplace_shared<LiDARMeasurementFactor>(X(i + 1), measurements[i][j], landmarks[j], meas_noise);
        }
    }

    // ====== Step 4: Optimize ======
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_estimate);
    gtsam::Values result = optimizer.optimize();

    // ====== Step 5: Print Results ======
    result.print("Optimized Results:\n");
}

// #define compile_next






#ifdef compile_next
template <typename PointType, typename VUX_PointType>
class CustomPointToPointFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
    CustomPointToPointFactor(
        gtsam::Key scan_key,
        const typename pcl::PointCloud<PointType>::Ptr &reference_map,
        const typename pcl::KdTreeFLANN<PointType>::Ptr &kdtree,
        const typename pcl::PointCloud<VUX_PointType>::Ptr &scan,
        const gtsam::SharedNoiseModel &noise_model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, scan_key),
          reference_map_(reference_map),
          kdtree_(kdtree),
          scan_(scan) {}

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override
    {

        gtsam::Vector3 total_error = gtsam::Vector3::Zero();
        gtsam::Matrix36 total_jacobian = gtsam::Matrix36::Zero();

        // for (const auto& point : scan_->points) {
        //     // Transform the scan point using the pose
        //     gtsam::Point3 transformed_point = pose.transformFrom(gtsam::Point3(point.x, point.y, point.z));

        //     // Find the nearest neighbor in the reference map
        //     PointType search_point;
        //     search_point.x = transformed_point.x();
        //     search_point.y = transformed_point.y();
        //     search_point.z = transformed_point.z();

        //     std::vector<int> pointIdxNKNSearch(1);
        //     std::vector<float> pointNKNSquaredDistance(1);

        //     if (kdtree_->nearestKSearch(search_point, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        //         const auto& closest_point = reference_map_->points[pointIdxNKNSearch[0]];
        //         gtsam::Point3 ref_point(closest_point.x, closest_point.y, closest_point.z);

        //         // Compute the point-to-point error
        //         gtsam::Vector3 error = transformed_point - ref_point;
        //         total_error += error;

        //         // If Jacobians are requested, compute them
        //         if (H) {
        //             gtsam::Matrix36 jacobian = gtsam::Matrix36::Zero();
        //             jacobian.block<3, 3>(0, 0) = gtsam::Matrix3::Identity(); // Translation part
        //             jacobian.block<3, 3>(0, 3) = -gtsam::skewSymmetric(transformed_point.vector()); // Rotation part
        //             total_jacobian += jacobian;
        //         }
        //     } else {
        //         throw std::runtime_error("No closest point found in the reference map.");
        //     }
        // }

        // // Average the error and Jacobian over all points
        // size_t num_points = scan_->size();
        // total_error /= num_points;
        // if (H) {
        //     *H = total_jacobian / num_points;
        // }

        return total_error;
    }

private:
    const typename pcl::PointCloud<PointType>::Ptr &reference_map_;
    const typename pcl::KdTreeFLANN<PointType>::Ptr &kdtree_;
    const typename pcl::PointCloud<VUX_PointType>::Ptr &scan_;
};

// template <typename PointType>
// class StandardICPFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
// {
// public:
//     StandardICPFactor(
//         gtsam::Key scan_key,
//         const typename pcl::PointCloud<PointType>::Ptr &reference_map,
//         const typename pcl::KdTreeFLANN<PointType>::Ptr &kdtree,
//         const typename pcl::PointCloud<VUX_PointType>::Ptr &scan,
//         bool use_point_to_plane,
//         const gtsam::SharedNoiseModel &noise)
//         : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise, scan_key),
//           reference_map(reference_map),
//           kdtree(kdtree),
//           scan(scan),
//           use_point_to_plane(use_point_to_plane) {}

//     gtsam::Vector evaluateError(
//         const gtsam::Pose3 &T_world_scan,
//             boost::optional<gtsam::Matrix &> H = boost::none) const override
//     //gtsam::Vector evaluateError(const gtsam::Pose3 &T_world_scan,
//     //    boost::optional<gtsam::Matrix &> H_scan = boost::none) const override
//     {
//         Eigen::Matrix4d T = T_world_scan.matrix();
//         gtsam::Vector residuals(scan->size());

//         for (size_t i = 0; i < scan->size(); ++i)
//         {
//             Eigen::Vector4d src_pt(scan->points[i].x, scan->points[i].y, scan->points[i].z, 1.0);
//             Eigen::Vector4d transformed_pt = T * src_pt;

//             std::vector<int> nn_indices(1);
//             std::vector<float> nn_dists(1);

//             PointType query;
//             query.x = transformed_pt.x();
//             query.y = transformed_pt.y();
//             query.z = transformed_pt.z();

//             if (kdtree->nearestKSearch(query, 1, nn_indices, nn_dists) > 0)
//             {
//                 PointType nearest_pt = reference_map->points[nn_indices[0]];
//                 V3D residual = V3D(
//                     nearest_pt.x - transformed_pt.x(),
//                     nearest_pt.y - transformed_pt.y(),
//                     nearest_pt.z - transformed_pt.z());

//                 if (use_point_to_plane)
//                 {
//                     // V3D normal(nearest_pt.normal_x, nearest_pt.normal_y, nearest_pt.normal_z);
//                     // residual = normal.dot(residual) * normal;
//                 }

//                 residuals[i] = residual.squaredNorm();
//             }
//         }

//         return residuals;
//     }

// private:
//     typename pcl::PointCloud<PointType>::Ptr reference_map;
//     typename pcl::KdTreeFLANN<PointType>::Ptr kdtree;
//     typename pcl::PointCloud<VUX_PointType>::Ptr scan;
//     bool use_point_to_plane;
// };

gtsam::Pose3 sophusSE3ToPose3(const Sophus::SE3 &se3)
{
    // return gtsam::Pose3(se3.rotationMatrix(), se3.translation());
    Eigen::Matrix4d T = se3.matrix();
    return gtsam::Pose3(gtsam::Rot3(T.topLeftCorner<3, 3>()), gtsam::Point3(T(0, 3), T(1, 3), T(2, 3)));
}

/**
 * @brief ICP distance factor with GTSAM's expression capability
 *        This is experimental and may not be suitable for practical use
 */
class ICPFactorExpr : public gtsam::NoiseModelFactor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ICPFactorExpr(
        gtsam::Key target_key,
        gtsam::Key source_key,
        const pcl::PointCloud<PointType>::Ptr &target,
        const pcl::KdTreeFLANN<PointType>::Ptr &target_tree,
        const gtsam::Point3 &source,
        const gtsam::SharedNoiseModel &noise_model);

    ~ICPFactorExpr();

    // virtual gtsam::Vector unwhitenedError(const gtsam::Values &values, boost::optional<std::vector<gtsam::Matrix> &> H = boost::none) const;
    virtual gtsam::Vector unwhitenedError(const gtsam::Values &values, boost::optional<std::vector<gtsam::Matrix> *> H = boost::none) const override
    {
        return error_expr.value(values, H ? *H : nullptr);
    }

    void update_correspondence(const gtsam::Values &values) const;

    gtsam::Point3_ calc_error() const;

private:
    const pcl::PointCloud<PointType>::Ptr target;
    const pcl::KdTreeFLANN<PointType>::Ptr target_tree;

    const gtsam::Pose3_ delta;
    const gtsam::Point3 source;

    mutable int target_index;
    mutable gtsam::Point3_ error_expr;
};

ICPFactorExpr::ICPFactorExpr(
    gtsam::Key target_key,
    gtsam::Key source_key,
    const pcl::PointCloud<PointType>::Ptr &target,
    const pcl::KdTreeFLANN<PointType>::Ptr &target_tree,
    const gtsam::Point3 &source,
    const gtsam::SharedNoiseModel &noise_model)
    : gtsam::NoiseModelFactor(noise_model, gtsam::KeyVector{target_key, source_key}),
      target(target),
      target_tree(target_tree),
      delta(gtsam::between(gtsam::Pose3_(target_key), gtsam::Pose3_(source_key))),
      source(source),
      target_index(-1),
      error_expr(calc_error()) {}

ICPFactorExpr::~ICPFactorExpr() {}

// gtsam::Vector ICPFactorExpr::unwhitenedError(const gtsam::Values &values, boost::optional<std::vector<gtsam::Matrix> &> H) const
// {
//     // Update corresponding point at the first call and every linearization call
//     if (target_index < 0 || H)
//     {
//         update_correspondence(values);
//         error_expr = calc_error();
//     }

//     return error_expr.value(values, H);
// }

void ICPFactorExpr::update_correspondence(const gtsam::Values &values) const
{
    // to be implemented
    //  gtsam::Point3 transed_source = delta.value(values) * source;

    // size_t k_index = -1;
    // double k_sq_dists = -1;
    // target_tree->knn_search(transed_source.data(), 1, &k_index, &k_sq_dists);
    // target_index = k_index;
}

gtsam::Point3_ ICPFactorExpr::calc_error() const
{
    if (target_index < 0)
    {
        return gtsam::Point3_(gtsam::Point3::Zero());
    }

    // to be implemented

    // return gtsam::Point3_(target->points[target_index].head<3>()) - gtsam::transformFrom(delta, gtsam::Point3_(source));
}

//-----------------------------------------------------

// A class to hold a set of ICP factors
class IntegratedICPFactorExpr : public gtsam::NonlinearFactor
{
public:
    IntegratedICPFactorExpr(const gtsam::NonlinearFactorGraph::shared_ptr &graph)
        : gtsam::NonlinearFactor(graph->keys()), graph(graph) {}

    virtual size_t dim() const override { return 3; }

    virtual double error(const gtsam::Values &values) const override { return graph->error(values); }
    virtual std::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &values) const override
    {
        // I'm not 100% sure if it is a correct way
        gtsam::Ordering ordering;
        ordering.push_back(keys()[0]);
        ordering.push_back(keys()[1]);

        auto linearized = graph->linearize(values);
        auto hessian = linearized->hessian(ordering);
        const auto &H = hessian.first;
        const auto &b = hessian.second;

        return gtsam::GaussianFactor::shared_ptr(
            new gtsam::HessianFactor(keys()[0], keys()[1], H.block<6, 6>(0, 0), H.block<6, 6>(0, 6), b.head<6>(), H.block<6, 6>(6, 6), b.tail<6>(), 0.0));
    }

    gtsam::NonlinearFactorGraph::shared_ptr graph;
};

gtsam::NonlinearFactorGraph::shared_ptr create_icp_factors(gtsam::Key target_key, gtsam::Key source_key,
                                                           const pcl::PointCloud<PointType>::Ptr &target,
                                                           const pcl::PointCloud<VUX_PointType>::Ptr &source,
                                                           const gtsam::SharedNoiseModel &noise_model)
{
    gtsam::NonlinearFactorGraph::shared_ptr factors(new gtsam::NonlinearFactorGraph);

    pcl::KdTreeFLANN<PointType>::Ptr target_tree(new pcl::KdTreeFLANN<PointType>());
    // std::shared_ptr<KdTree> target_tree(new KdTree(target->points, target->size()));
    target_tree->setInputCloud(target); // take this from mls

    for (int i = 0; i < source->size(); i++)
    {
        // create a factor for each point
        // gtsam::NonlinearFactor::shared_ptr factor(new ICPFactorExpr(target_key, source_key, target, target_tree, source->points[i].head<3>(), noise_model));
        // factors->add(factor);
    }

    return factors;
}

gtsam::NonlinearFactor::shared_ptr create_integrated_icp_factor(gtsam::Key target_key, gtsam::Key source_key,
                                                                const pcl::PointCloud<PointType>::Ptr &target,
                                                                const pcl::PointCloud<VUX_PointType>::Ptr &source,
                                                                const gtsam::SharedNoiseModel &noise_model)
{
    auto factors = create_icp_factors(target_key, source_key, target, source, noise_model);
    return gtsam::NonlinearFactor::shared_ptr(new IntegratedICPFactorExpr(factors));
}

#endif