#pragma once

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

#include <pcl/kdtree/kdtree_flann.h>

using namespace gtsam;

using gtsam::symbol_shorthand::A; // for anchor
using gtsam::symbol_shorthand::X; // Pose symbols

class PointToDistributionFactor : public NoiseModelFactor1<Pose3>
{
    using Base = NoiseModelFactor1<Pose3>;
    Point3 point_;   // LiDAR point in local frame
    Point3 mean_;    // Mean of nearby map points in world frame
    Matrix3 invCov_; // Inverse covariance of the ndt cell

public:
    PointToDistributionFactor(Key poseKey,
                              const Point3 &point,
                              const Point3 &mean,
                              const Matrix3 &invCov)
        : Base(noiseModel::Unit::Create(3), poseKey),
          point_(point), mean_(mean), invCov_(invCov) {}

    Vector evaluateError(const Pose3 &pose, OptionalMatrixType H) const override
    {
        Point3 p_world = pose.transformFrom(point_, H);

        Vector3 diff = p_world - mean_;
        if (H)
        {
            // Whitened Jacobian: ∂e/∂pose = inv(Cov) * ∂(p_world)/∂pose
            *H = invCov_ * (*H); // Chain rule: dMahalanobis/dPose = Σ⁻¹ * dP/dPose
        }
        return invCov_ * diff; // Mahalanobis residual vector
    }

    virtual ~PointToDistributionFactor() {}
};

std::deque<Sophus::SE3> pose_buffer;
std::deque<pcl::PointCloud<VUX_PointType>::Ptr> scan_buffer;

const size_t max_buffer_size = 50;
const size_t step_size = 45;

Pose3 anchor_pose, relative_anchor;
bool has_prev_solution = false;

int optimize_iterations = 10;

double prev_error = 9999., current_error;  // std::numeric_limits<double>::max();
const double convergence_threshold = .001;

Pose3 sophusToGtsam(const Sophus::SE3 &pose)
{
    return Pose3(pose.matrix());
}

Sophus::SE3 GtsamToSophus(const Pose3 &pose)
{
    Eigen::Matrix4d T = pose.matrix(); // Get 4x4 matrix
    return Sophus::SE3(T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>());
}

void updateDataAssociation(int pose_key,
                           const pcl::PointCloud<VUX_PointType>::Ptr &scan, // scan in sensor frame
                           const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
                           const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud,
                           const Sophus::SE3 &nn_init_guess_T, NonlinearFactorGraph &this_Graph)
{

    double threshold_nn = 1.0;
    bool radius_based = true; // false;

    //todo 
    //landmarks_map is a vector of nearest neighbour cells for each point in the current scan scan
    //a list of nearest cells for scan
    
    for (const auto &[landmark_id, land] : landmarks_map)
    {
        for (int i = 0; i < land.scan_idx.size(); i++)
        {
            const auto &p_idx = land.scan_idx[i]; // at index p_idx from that scan
            const auto &raw_point = scan->points[p_idx];

            Point3 measured_point(raw_point.x, raw_point.y, raw_point.z); // measured_landmark_in_sensor_frame
            Point3 target_point(land.landmark_point.x(), land.landmark_point.y(), land.landmark_point.z());
            Matrix3 invCov = land.covariance.inverse();

            // p2distribution
            this_Graph.emplace_shared<PointToDistributionFactor>(X(pose_key), measured_point, target_point, invCov);
        }
    }
}

Sophus::SE3 updateBatch(
    const pcl::PointCloud<VUX_PointType>::Ptr &scan, // scan in sensor frame
    const Sophus::SE3 &initial_pose,                 // absolute T,
    const pcl::KdTreeFLANN<PointType>::Ptr &refference_kdtree,
    const pcl::PointCloud<PointType>::Ptr &reference_localMap_cloud)
{

    bool debug = true;

    pose_buffer.push_back(initial_pose);
    scan_buffer.push_back(scan);

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

    for (size_t i = 0; i < pose_buffer.size(); i++)
    {
        Pose3 pose = sophusToGtsam(pose_buffer[i]);
        current_values.insert(X(i), pose);
    }

    // optimization refinement
    NonlinearFactorGraph graph;
    for (int iter = 0; iter < optimize_iterations; ++iter)
    {
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
            Sophus::SE3 nn_init_guess_T = GtsamToSophus(curr_pose); // current best guess - use it for nearest neighbour search

            updateDataAssociation(i,  // add planes for graph from pose i
                                  scan_buffer[i], // scan in sensor frame
                                  refference_kdtree, reference_localMap_cloud,
                                  nn_init_guess_T, graph);
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
    }

    has_prev_solution = true;              // do not use anchor for now, untill figure how noise works which frame
    for (size_t i = 0; i < step_size; i++) // remove first step_size poses
    {
        ros::spinOnce();
        if (flag || !ros::ok())
            break;

        if (!pose_buffer.empty())
        {
            anchor_scan = scan_buffer.front();

            pose_buffer.pop_front();
            scan_buffer.pop_front();

            anchor_pose = current_values.at<Pose3>(X(i));
            relative_anchor = anchor_pose.between(current_values.at<Pose3>(X(i + 1)));
        }
    }

    return GtsamToSophus(anchor_pose);
}
