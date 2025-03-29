#ifndef USE_RIEKF_H1
#define USE_RIEKF_H1

#include <Estimator.hpp>

#if USE_STATIC_KDTREE == 0
#include <ikd-Tree/ikd_Tree.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ekf;

#ifdef ADAPTIVE_KERNEL
inline double ComputeModelError(const Sophus::SE3 &model_deviation, double max_range)
{
    const double theta = Eigen::AngleAxisd(model_deviation.rotation_matrix()).angle();
    const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
    const double delta_trans = model_deviation.translation().norm();
    return delta_trans + delta_rot;
}

inline double square(double x) { return x * x; }

struct AdaptiveThreshold
{
    explicit AdaptiveThreshold(double initial_threshold, double min_motion_th, double max_range)
        : initial_threshold_(initial_threshold),
          min_motion_th_(min_motion_th),
          max_range_(max_range)
    {
    }

    // Update the current belief of the deviation from the prediction model
    inline void UpdateModelDeviation(const Sophus::SE3 &current_deviation)
    {
        model_deviation_ = current_deviation;
    }

    inline double ComputeThreshold()
    {
        if (!has_moved)
            return initial_threshold_;

        double model_error = ComputeModelError(model_deviation_, max_range_);
        if (model_error > min_motion_th_)
        {
            model_error_sse2_ += model_error * model_error;
            num_samples_++;
        }

        if (num_samples_ <= 1)
        {
            return initial_threshold_;
        }

        return std::sqrt(model_error_sse2_ / num_samples_);
    }

    // configurable parameters
    double initial_threshold_;
    double min_motion_th_;
    double max_range_;

    bool has_moved = true;
    // Local cache for ccomputation
    double model_error_sse2_ = 0;
    int num_samples_ = 0;
    Sophus::SE3 model_deviation_ = Sophus::SE3();
};
#endif

// put these in a namespace
extern PointCloudXYZI::Ptr normvec;
extern PointCloudXYZI::Ptr laserCloudOri;
extern PointCloudXYZI::Ptr corr_normvect;
extern std::vector<bool> point_selected_surf;

struct residual_struct
{
    bool valid;                                                // Whether the number of effective feature points meets the requirements
    bool converge;                                             // When iterating, whether it has converged
    Eigen::Matrix<double, Eigen::Dynamic, 1> innovation;       // residual
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // Jacobian matrix H
};

class RIEKF : public Estimator
{
public:
    pcl::KdTreeFLANN<PointType>::Ptr localKdTree_map;
#ifdef ADAPTIVE_KERNEL
    RIEKF()
        : adaptive_threshold(1.0, 0.1, 100.0) // Initialize with constructor initializer list
    {

#ifdef _OPENMP

        int total_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        max_threads = std::max(max_threads, 4); // at least 4 threads (optional)
        max_threads = 20;                       // remove this
        omp_set_num_threads(max_threads);
#pragma omp parallel
        {
            total_threads = omp_get_num_threads();
#pragma omp master
            {
                std::cout << "Total OpenMP threads used : " << total_threads << std::endl;
            }
        }
        localKdTree_map.reset(new pcl::KdTreeFLANN<PointType>());
#endif
    }
#else
    RIEKF()
    {

#ifdef _OPENMP

        int total_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        max_threads = std::max(max_threads, 4); // at least 4 threads (optional)
        max_threads = 20;                       // remove this
        omp_set_num_threads(max_threads);
#pragma omp parallel
        {
            total_threads = omp_get_num_threads();
#pragma omp master
            {
                std::cout << "Total OpenMP threads used : " << total_threads << std::endl;
            }
        }
#endif
        localKdTree_map.reset(new pcl::KdTreeFLANN<PointType>());
    };
#endif

    ~RIEKF() {};

private:
    // these should be in the inherit class
#ifdef ADAPTIVE_KERNEL
    double sigma = 0;
    AdaptiveThreshold adaptive_threshold;
#endif

public:
#if USE_STATIC_KDTREE == 0
    void observtion_model_parallel(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                   KD_TREE<PointType> &ikdtree, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    void lidar_observation_model(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                 KD_TREE<PointType> &ikdtree, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    bool update(double R, PointCloudXYZI::Ptr &feats_down_body,
                KD_TREE<PointType> &ikdtree, std::vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est);
#else

    void lidar_observation_model(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                 PointCloudXYZI::Ptr &map, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    bool update(double R, PointCloudXYZI::Ptr &feats_down_body,
                PointCloudXYZI::Ptr &map, std::vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est);

#endif
    // gnss update
    void update(const V3D &pos, const V3D &cov_pos_, int maximum_iter, bool global_error, M3D R = Eye3d);
};

#endif