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

#include <Pose6D.h>
#include <point_definitions.hpp>

#include <sophus/se3.h>
#include <sophus/so3.h>

#include <tbb/global_control.h>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>


#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif


#define LASER_POINT_COV (0.001) 
#define INIT_TIME (0.1)

constexpr double ESTIMATION_THRESHOLD_ = 0.001;
const double G_m_s2 = 9.81; // positive z axis up

inline int NUM_THREADS = 16;

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
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

inline double tolerance() { return 1e-5; }
inline M3D hat(const V3D &v)
{
    M3D m;
    m << 0.0, -v.z(), v.y(),
        v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

// right jacobian
inline M3D J_right(const V3D &v)
{
    M3D I = M3D::Identity();
    const double squaredNorm = v.squaredNorm();
    const double norm = std::sqrt(squaredNorm);

    if (norm < tolerance())
    {
        return I;
    }

    const M3D v_hat = hat(v);

    return (I - (1.0 - std::cos(norm)) / squaredNorm * v_hat + (1.0 - std::sin(norm) / norm) / squaredNorm * (v_hat * v_hat));
}

inline M3D Jr_inv(const Eigen::Vector3d &phi)
{
    double theta = phi.norm();
    M3D I = M3D::Identity();

    if (theta < tolerance())
    {
        M3D phi_hat = hat(phi);
        return I + 0.5 * phi_hat + (1.0 / 12.0) * phi_hat * phi_hat;
    }
    else
    {
        M3D phi_hat = hat(phi);
        M3D phi_hat2 = phi_hat * phi_hat;

        double a = 1.0 / (theta * theta);
        double b = (1 + cos(theta)) / (2 * theta * sin(theta));

        return I + 0.5 * phi_hat + (a - b) * phi_hat2;
    }
}

// this returns left jacobian
inline M3D A_matrix(const V3D &v)
{
    M3D I = M3D::Identity();
    const double squaredNorm = v.squaredNorm();
    const double norm = std::sqrt(squaredNorm);

    if (norm < tolerance())
    {
        return I;
    }

    const M3D v_hat = hat(v);
    return I + (1.0 - std::cos(norm)) / squaredNorm * v_hat + (1.0 - std::sin(norm) / norm) / squaredNorm * (v_hat * v_hat);
}

inline M3D Jl_inv(const V3D &phi)
{
    const double eps = 1e-8;
    double theta = phi.norm();

    M3D I = M3D::Identity();
    M3D phi_hat = hat(phi);
    M3D phi_hat2 = phi_hat * phi_hat;

    if (theta < tolerance())
    {
        // Taylor expansion around theta = 0
        return I + 0.5 * phi_hat + (1.0 / 12.0) * phi_hat2;
    }
    else
    {
        double half_theta = 0.5 * theta;
        double cot_half_theta = 1.0 / std::tan(half_theta);

        double A = 1.0 / (theta * theta) - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));

        return I + 0.5 * phi_hat + A * phi_hat2;
    }
}

namespace ekf
{
    template <typename T>
    inline bool esti_plane_pca(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &points,
                               const double &threshold, const std::vector<double> &point_weights, double &plane_var, bool weighted_mean = false)
    {
        const size_t neighbours = points.size();
        if (neighbours < 3)
            return false; // Need at least 3 points to define a plane

        V3D centroid(0, 0, 0); // Compute the centroid
        M3D covariance;        // Compute covariance matrix
        covariance.setZero();

        // Regularize
        double lambda_reg = 1e-6;
        covariance = covariance + lambda_reg * Eye3d;

        if (weighted_mean)
        {
            double weight_sum = 0.0;

            // Compute weighted centroid
            for (int j = 0; j < neighbours; j++)
            {
                const double &w = point_weights[j];

                centroid(0) += w * points[j].x;
                centroid(1) += w * points[j].y;
                centroid(2) += w * points[j].z;

                weight_sum += w;
            }
            // std::cout<<"weight_sum:"<<weight_sum<<std::endl;
            centroid /= weight_sum;

            // Compute weighted covariance matrix
            for (int j = 0; j < neighbours; j++)
            {
                const double &w = point_weights[j];
                const auto &p = points[j];
                V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                covariance += w * diff * diff.transpose();
            }
            covariance /= weight_sum;
        }
        else
        {
            for (int j = 0; j < neighbours; j++)
            {
                centroid(0) += points[j].x;
                centroid(1) += points[j].y;
                centroid(2) += points[j].z;
            }
            centroid /= neighbours;

            for (int j = 0; j < neighbours; j++)
            {
                const auto &p = points[j];
                V3D diff(p.x - centroid(0), p.y - centroid(1), p.z - centroid(2));
                covariance += diff * diff.transpose();
            }
            covariance /= neighbours;
        }

        // Compute Eigenvalues and Eigenvectors
        Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);

        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Eigen solver failed!" << std::endl;
            throw std::runtime_error("Error: Eigen solver failed!");
            return false;
        }

        V3D norm = solver.eigenvectors().col(0); // Smallest eigenvector
        norm.normalize();

        // Compute plane offset: d = -(n * centroid)
        double d = -norm.dot(centroid); // plane offset

        const auto &eigenvalues = solver.eigenvalues();
        double lambda0 = eigenvalues(0); // smallest
        double lambda1 = eigenvalues(1);
        double lambda2 = eigenvalues(2);

        double c = lambda0 / (lambda0 + lambda1 + lambda2);

        double eps = .00001; //to avoid some degenerate planes 
        if(c > eps && c <= 0.04)
        {
            pca_result.template head<3>() = norm.template cast<T>();
            pca_result(3) = static_cast<T>(d);

            // These are independent, so variances add
            plane_var = lambda0 + LASER_POINT_COV;

            return true;
        }

        return false;
    }
};

namespace gnss
{
    V3D computeWeightedAverage(const std::vector<V3D> &measurements, const std::vector<V3D> &covariances);
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
};

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

void TransformPoints(const Sophus::SE3 &T, pcl::PointCloud<PointType>::Ptr &cloud);

std::vector<std::string> expandBagPattern(const std::string &pattern_path);

#endif