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

constexpr int MAX_NUM_ITERATIONS_ = 500; // icp
constexpr double ESTIMATION_THRESHOLD_ = 0.001;

// 1.0 for no gravity
const double G_m_s2 = 9.81; // positive as before z axis up

// the new system has the z-axis down therefore negative
//const double G_m_s2 = -9.81; //for new lieksa data - take this as param 

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

    template <typename T>
    inline bool esti_plane2(Eigen::Matrix<T, 4, 1> &pca_result, const std::vector<PointType> &points, const T &threshold)
    {
        const size_t N = points.size();
        if (N < 3)
            return false; // Not enough points to estimate a plane

        Eigen::Matrix<T, Eigen::Dynamic, 3> A(N, 3);
        Eigen::Matrix<T, Eigen::Dynamic, 1> b(N);
        b.setConstant(-1.0);

        for (size_t j = 0; j < N; ++j)
        {
            A(j, 0) = static_cast<T>(points[j].x);
            A(j, 1) = static_cast<T>(points[j].y);
            A(j, 2) = static_cast<T>(points[j].z);
        }

        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        T norm = normvec.norm();
        if (norm < T(1e-6))
            return false; // Avoid division by zero

        // Normalized plane coefficients A, B, C, D (where D = 1/norm)
        pca_result.template head<3>() = normvec / norm;
        pca_result(3) = T(1.0) / norm;

        // Validate all points lie within threshold distance from the plane
        for (size_t j = 0; j < N; ++j)
        {
            T dist = pca_result(0) * static_cast<T>(points[j].x) +
                     pca_result(1) * static_cast<T>(points[j].y) +
                     pca_result(2) * static_cast<T>(points[j].z) +
                     pca_result(3);
            if (std::abs(dist) > threshold)
            {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    inline bool esti_plane2(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &points, const T &threshold)
    {
        const size_t N = points.size();
        // std::cout<<"N:"<<N<<std::endl;
        if (N < 3)
            return false; // Need at least 3 points to define a plane

        Eigen::Matrix<T, Eigen::Dynamic, 3> A(N, 3);
        Eigen::Matrix<T, Eigen::Dynamic, 1> b(N);
        b.setConstant(T(-1.0));

        // Fill matrix A with point coordinates
        for (size_t j = 0; j < N; ++j)
        {
            A(j, 0) = static_cast<T>(points[j].x);
            A(j, 1) = static_cast<T>(points[j].y);
            A(j, 2) = static_cast<T>(points[j].z);
        }

        // Solve for normal vector components
        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        T norm = normvec.norm();
        if (norm < T(1e-6))
            return false; // Avoid division by zero or near-singular result

        // Normalize and fill output
        pca_result.template head<3>() = normvec / norm;
        pca_result(3) = T(1.0) / norm;

        // Validate all points lie close to the plane
        for (size_t j = 0; j < N; ++j)
        {
            T dist = pca_result(0) * static_cast<T>(points[j].x) +
                     pca_result(1) * static_cast<T>(points[j].y) +
                     pca_result(2) * static_cast<T>(points[j].z) +
                     pca_result(3);
            if (std::abs(dist) > threshold)
            {
                return false;
            }
        }

        return true;
    }

    template <typename T>
    inline bool esti_plane_pca(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &points, const double &threshold, const std::vector<double> &point_weights, bool weighted_mean = false)
    {
        const size_t neighbours = points.size();
        // std::cout<<"N:"<<N<<std::endl;
        if (neighbours < 3)
            return false; // Need at least 3 points to define a plane

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
            // if (weighted_mean && point_weights.size() < neighbours) {
            //     std::cerr << "Error: point_weights has fewer elements than points." << std::endl;
            //     std::cout<<"neighbours:"<<neighbours<<", point_weights.size():"<<point_weights.size()<<std::endl;
            //     throw std::runtime_error("Error: point_weights has fewer elements than points.");
            //     return false;
            // }

            // std::cout<<"\nPerformed weighted..."<<"point_weights[0]:"<<point_weights[0]<<", point_weights[last]:"<<point_weights[neighbours-1]<<std::endl;

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

        // Compute plane offset: d = - (n * centroid)
        double d = -norm.dot(centroid);

        // Compute eigenvalue ratios to assess planarity
        const auto &eigenvalues = solver.eigenvalues();
        double lambda0 = eigenvalues(0); // smallest
        double lambda1 = eigenvalues(1);
        double lambda2 = eigenvalues(2);

        double curvature = lambda0 / (lambda0 + lambda1 + lambda2);

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

        if (curvature > .0001 && curvature <= threshold)
        {
            pca_result.template head<3>() = norm.template cast<T>();
            pca_result(3) = static_cast<T>(d);

            return true;
        }

        return false;
    }

    template <typename T>
    inline bool esti_plane_pca(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &points,
                               const double &threshold, const std::vector<double> &point_weights, double &plane_var, bool weighted_mean = false)
    {
        const size_t neighbours = points.size();
        // std::cout<<"N:"<<neighbours<<std::endl;
        if (neighbours < 3)
            return false; // Need at least 3 points to define a plane

        V3D centroid(0, 0, 0); // Compute the centroid
        M3D covariance;        // Compute covariance matrix
        covariance.setZero();

        // Regularize
        //  double lambda_reg = 1e-6;
        //  covariance = covariance + lambda_reg * Eye3d;

        if (weighted_mean)
        {
            // if (weighted_mean && point_weights.size() != neighbours) {
            //     std::cerr << "Error: point_weights has fewer elements than points." << std::endl;
            //     std::cout<<"neighbours:"<<neighbours<<", point_weights.size():"<<point_weights.size()<<std::endl;
            //     throw std::runtime_error("Error: point_weights has fewer elements than points.");
            //     return false;
            // }

            // std::cout<<"\nPerformed weighted..."<<"point_weights[0]:"<<point_weights[0]<<", point_weights[last]:"<<point_weights[neighbours-1]<<std::endl;

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

        // Compute plane offset: d = - (n * centroid)
        double d = -norm.dot(centroid);

        // Compute eigenvalue ratios to assess planarity
        const auto &eigenvalues = solver.eigenvalues();
        double lambda0 = eigenvalues(0); // smallest
        double lambda1 = eigenvalues(1);
        double lambda2 = eigenvalues(2);

        double curvature = lambda0 / (lambda0 + lambda1 + lambda2);

        if (curvature > .0001 && curvature <= threshold)
        {
            pca_result.template head<3>() = norm.template cast<T>();
            pca_result(3) = static_cast<T>(d);

            plane_var = lambda0;

            return true;
        }

        return false;
    }

    inline bool esti_cov(const PointVector &points,
                         const std::vector<double> &point_weights,
                         M3D &out_cov, V3D &out_center, bool weighted_mean = false)
    {
        const size_t neighbours = points.size();
        // std::cout<<"N:"<<N<<std::endl;
        if (neighbours < 5)
            return false; // Need at least 5 points

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
            // if (weighted_mean && point_weights.size() < neighbours) {
            //     std::cerr << "Error: point_weights has fewer elements than points." << std::endl;
            //     std::cout<<"neighbours:"<<neighbours<<", point_weights.size():"<<point_weights.size()<<std::endl;
            //     throw std::runtime_error("Error: point_weights has fewer elements than points.");
            //     return false;
            // }

            // std::cout<<"\nPerformed weighted..."<<"point_weights[0]:"<<point_weights[0]<<", point_weights[last]:"<<point_weights[neighbours-1]<<std::endl;

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

        out_cov = covariance;
        out_center = centroid;

        return true;
    }
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
    int max_points_per_voxel = 25;

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

void TransformPoints(const Sophus::SE3 &T, std::vector<V3D_4> &points);

void Eigen2PCL(PointCloudXYZI::Ptr &pcl_cloud, const std::vector<V3D> &eigen_cloud);

void Eigen2PCL(PointCloudXYZI::Ptr &pcl_cloud, const std::vector<V3D_4> &eigen_cloud);

void PCL2EIGEN(const PointCloudXYZI::Ptr &pcl_cloud, std::vector<V3D> &eigen_cloud);

sensor_msgs::PointField GetTimestampField(const sensor_msgs::PointCloud2::ConstPtr &msg);

std::vector<double> NormalizeTimestamps(const std::vector<double> &timestamps);

Sophus::SE3 registerClouds(pcl::PointCloud<PointType>::Ptr &src, pcl::PointCloud<PointType>::Ptr &tgt, pcl::PointCloud<PointType>::Ptr &cloud_aligned);

void TransformPoints(const Sophus::SE3 &T, pcl::PointCloud<VUX_PointType>::Ptr &cloud);

void TransformPoints(const Sophus::SE3 &T, pcl::PointCloud<PointType>::Ptr &cloud);

std::vector<std::string> expandBagPattern(const std::string &pattern_path);

#endif