#ifndef COMMON_LIB_H1
#define COMMON_LIB_H1

#pragma once
#define PCL_NO_PRECOMPILE

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <gps_common/GPSFix.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <eigen_conversions/eigen_msg.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/crop_box.h>

#include <ctime>
#include <fstream>
#include <iostream>
#include <chrono>

//#include "sophus/se3.h"
//#include "sophus/so3.h"

#include <sophus/se3.h>
#include <sophus/so3.h>


#include <Pose6D.h>
#include <point_definitions.hpp>

constexpr int MAX_NUM_ITERATIONS_ = 500; // icp
constexpr double ESTIMATION_THRESHOLD_ = 0.001;

// 1.0 for no gravity
const double G_m_s2 = 9.81;

#define NUM_MATCH_POINTS (5)

#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

typedef s_mls::Pose6D Pose6D;

typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

extern M3D Eye3d;
extern M3F Eye3f;
extern V3D Zero3d;
extern V3F Zero3f;

typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

namespace ekf
{
    float calc_dist(PointType p1, PointType p2);

    template <typename T>
    inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
    {
        Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0f;

        // A/Dx + B/Dy + C/Dz + 1 = 0
        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

        T n = normvec.norm();
        // pca_result
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;

        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const std::vector<PointType> &point, const T &threshold)
    {
        Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0f;

        // A/Dx + B/Dy + C/Dz + 1 = 0
        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

        T n = normvec.norm();
        // pca_result
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;

        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
            {
                return false;
            }
        }
        return true;
    }


    //TODO - to be  tested in time and accuracy 

    /*
    PCA method---------------------------------------------------
    template <typename T>
inline bool esti_plane_pca(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector<T> &points)
{
    if (points.size() < 3) return false; // Need at least 3 points for plane fitting

    // Compute the centroid
    Eigen::Matrix<T, 3, 1> centroid(0, 0, 0);
    for (const auto &p : points)
    {
        centroid(0) += p.x;
        centroid(1) += p.y;
        centroid(2) += p.z;
    }
    centroid /= static_cast<T>(points.size());

    // Compute covariance matrix
    Eigen::Matrix<T, 3, 3> covariance;
    covariance.setZero();
    for (const auto &p : points)
    {
        Eigen::Matrix<T, 3, 1> diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
        covariance += diff * diff.transpose();
    }
    covariance /= static_cast<T>(points.size());

    // Compute Eigenvalues and Eigenvectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 3, 3>> solver(covariance);
    Eigen::Matrix<T, 3, 1> normal = solver.eigenvectors().col(0); // Smallest eigenvector

    // Normalize the normal
    normal.normalize();

    // Assign to result
    pca_result(0) = normal(0);
    pca_result(1) = normal(1);
    pca_result(2) = normal(2);
    pca_result(3) = -normal.dot(centroid); // Plane equation D = -N . Centroid

    return true;
}

// Function to estimate the plane from 3 points------------------------------------------------
template <typename T, typename PointType>
bool estimatePlaneFrom3Points(Eigen::Matrix<T, 4, 1>& plane_coefficients, const PointType& p1, const PointType& p2, const PointType& p3) {
    Eigen::Matrix<T, 3, 1> v1 = {p2.x - p1.x, p2.y - p1.y, p2.z - p1.z};
    Eigen::Matrix<T, 3, 1> v2 = {p3.x - p1.x, p3.y - p1.y, p3.z - p1.z};
    Eigen::Matrix<T, 3, 1> normal = v1.cross(v2);

    T norm = normal.norm();
    if (norm < std::numeric_limits<T>::epsilon()) {
        // Points are collinear
        return false;
    }

    // Normalize the normal vector
    normal /= norm;

    // Plane equation: ax + by + cz + d = 0
    plane_coefficients(0) = normal(0);
    plane_coefficients(1) = normal(1);
    plane_coefficients(2) = normal(2);
    plane_coefficients(3) = -normal.dot(Eigen::Matrix<T, 3, 1>{p1.x, p1.y, p1.z});

    return true;
}

// Function to estimate the best normal from all combinations of 3 points ---------------------------------------
template <typename T, typename PointType>
bool estimateBestNormal(Eigen::Matrix<T, 4, 1>& best_plane_coefficients, const std::vector<PointType>& neighbors, T threshold) {
    const int num_neighbors = neighbors.size();
    if (num_neighbors < 3) {
        // Not enough neighbors to form a plane
        return false;
    }

    // Variables to store the best plane
    T best_error = std::numeric_limits<T>::max();
    Eigen::Matrix<T, 4, 1> current_plane_coefficients;

    // Iterate over all combinations of 3 points
    for (int i = 0; i < num_neighbors - 2; ++i) {
        for (int j = i + 1; j < num_neighbors - 1; ++j) {
            for (int k = j + 1; k < num_neighbors; ++k) {
                // Estimate the plane from the current combination of 3 points
                if (!estimatePlaneFrom3Points(current_plane_coefficients, neighbors[i], neighbors[j], neighbors[k])) {
                    continue; // Skip collinear points
                }

                // Calculate the error (sum of squared distances) for the current plane
                T error = 0;
                for (const auto& point : neighbors) {
                    T distance = std::abs(current_plane_coefficients(0) * point.x +
                                          current_plane_coefficients(1) * point.y +
                                          current_plane_coefficients(2) * point.z +
                                          current_plane_coefficients(3));
                    error += distance * distance;
                }

                // Update the best plane if the current error is smaller
                if (error < best_error) {
                    best_error = error;
                    best_plane_coefficients = current_plane_coefficients;
                }
            }
        }
    }

    // Check if the best plane fits all points within the threshold
    for (const auto& point : neighbors) {
        T distance = std::abs(best_plane_coefficients(0) * point.x +
                      best_plane_coefficients(1) * point.y +
                      best_plane_coefficients(2) * point.z +
                      best_plane_coefficients(3));
        if (distance > threshold) {
            return false;
        }
    }

    return true;
}
    */

};

namespace gnss
{
    V3D computeWeightedAverage(const std::vector<V3D> &measurements, const std::vector<V3D> &covariances);
    double findAngle(const V3D &gps0, const V3D &gps1, const V3D &imu0, const V3D &imu1);
    double checkAlignment(const Eigen::Vector2d &vec1, const Eigen::Vector2d &vec2);
    Eigen::Vector2d rotateVector(const Eigen::Vector2d &vec, double angle_deg);
    double verifyAngle(const double &theta_degrees, const V3D &gps0, const V3D &gps1, const V3D &imu0, const V3D &imu1);

    struct LineModel
    {
        double m; // Slope
        double b; // Intercept
    };

    LineModel fitLine(const std::pair<double, double> &p1, const std::pair<double, double> &p2);
    double pointToLineDistance(double x, double y, const LineModel &model);
    LineModel ransacFitLine(const std::vector<double> &x, const std::vector<double> &y, int iterations = 100, double threshold = 5.0);
};

void computeTransformation(const std::vector<V3D> &gnss_points, const std::vector<V3D> &mls_points, M3D &R, V3D &t);

inline const bool time_sort(PointType &x, PointType &y) { return (x.time < y.time); };

struct MeasureGroup // Lidar data and imu measurements for the current process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
    std::deque<sensor_msgs::Imu::ConstPtr> imu;
#ifdef SAVE_DATA
    sensor_msgs::PointCloud2::ConstPtr lidar_msg;
#endif
};

struct Config // used for ICP
{
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 1.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;
};

// Define a fixed-size queue class
template <typename T>
class FixedSizeQueue
{
public:
    explicit FixedSizeQueue(size_t size) : maxSize(size) {}

    void push(const T &item)
    {
        if (queue.size() == maxSize)
        {
            queue.pop_front(); // Remove the oldest item if the queue is full
        }
        queue.push_back(item); // Add the new item
    }

    bool contains(const T &item) const
    {
        return std::find(queue.begin(), queue.end(), item) != queue.end();
    }

    void print() const
    {
        for (const auto &item : queue)
        {
            std::cout << "Filename: " << item << std::endl;
        }
        std::cout << std::endl;
    }

    // private:
    size_t maxSize;
    std::deque<T> queue;
};

template <typename T>
inline auto set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                       const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)
            rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return std::move(rot_kp);
}

V3D calculateStdDev(const std::vector<V3D> &measurements, const V3D &mean);

double calculateMean(const std::vector<double> &values);

double calculateMedian(std::vector<double> values);

void TransformPoints(const M3D &R, const V3D &T, PointCloudXYZI::Ptr &points);

void TransformPoints(const Sophus::SE3 &T, std::vector<V3D> &points);

void Eigen2PCL(PointCloudXYZI::Ptr &pcl_cloud, const std::vector<V3D> &eigen_cloud);

void PCL2EIGEN(const PointCloudXYZI::Ptr &pcl_cloud, std::vector<V3D> &eigen_cloud);

sensor_msgs::PointField GetTimestampField(const sensor_msgs::PointCloud2::ConstPtr &msg);

std::vector<double>  NormalizeTimestamps(const std::vector<double> &timestamps);

Sophus::SE3 registerClouds(pcl::PointCloud<PointType>::Ptr &src, pcl::PointCloud<PointType>::Ptr &tgt, pcl::PointCloud<PointType>::Ptr &cloud_aligned);


#endif