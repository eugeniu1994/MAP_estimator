#ifndef USE_RIEKF_H1
#define USE_RIEKF_H1

#include <Estimator.hpp>

#if USE_STATIC_KDTREE == 0
#include <ikd-Tree/ikd_Tree.h>
#endif

#include <chrono>

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
extern std::vector<double> normvec_var;
extern std::vector<double> corr_normvec_var;

extern std::vector<M3D> tgt_covs;
extern std::vector<M3D> corr_tgt_covs;

extern std::vector<V3D> laserCloudTgt;
extern std::vector<V3D> corr_laserCloudTgt;

extern std::vector<V3D> laserCloudSrc;
extern std::vector<M3D> src_covs;

struct residual_struct
{
    bool valid;                                                // Whether the number of effective feature points meets the requirements
    bool converge;                                             // When iterating, whether it has converged
    Eigen::Matrix<double, Eigen::Dynamic, 1> innovation;       // residual
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // Jacobian matrix H

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H_T_R_inv;
};

const int gps_dim = 3;
const int se3_dim = 6;

const int kernel_row = 1, kernel_col = 1;// 10; // pretty fast and accurate

// Eigen provides fixed-size types
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

class RIEKF : public Estimator
{
public:
    std::vector<std::pair<int, int>> density_kernel;
    int effct_feat_num;
    pcl::KdTreeFLANN<PointType>::Ptr localKdTree_map;
    // pcl::KdTreeFLANN<PointType>::Ptr localKdTree_map_als;
    pcl::KdTreeFLANN<PointType>::Ptr cloud_tree;




    Eigen::Matrix<double, gps_dim, state_size> H_gnss;
    Eigen::Matrix<double, se3_dim, state_size> H_se3;
    // Position part
#ifdef ADAPTIVE_KERNEL
    RIEKF()
        : adaptive_threshold(1.0, 0.1, 100.0) // Initialize with constructor initializer list
    {
        for (int i = -kernel_row; i <= kernel_row; i++)
        { // row
            for (int j = -kernel_col; j <= kernel_col; j++)
            {
                if (j == 0 && i == 0)
                {
                    continue;
                }
                density_kernel.push_back(std::make_pair(i, j));
            }
        }
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
        // localKdTree_map_als.reset(new pcl::KdTreeFLANN<PointType>());

        cloud_tree.reset(new pcl::KdTreeFLANN<PointType>());

        H_gnss = Eigen::Matrix<double, gps_dim, state_size>::Zero(); // 3 * n
        H_gnss.block<3, 3>(P_ID, P_ID) = Eye3d;

        H_se3 = Eigen::Matrix<double, se3_dim, state_size>::Zero();
        H_se3.block<3, 3>(P_ID, P_ID) = Eye3d;
        H_se3.block<3, 3>(R_ID, R_ID) = Eye3d;
#endif
    }
#else
    RIEKF()
    {
        for (int i = -kernel_row; i <= kernel_row; i++)
        { // row
            for (int j = -kernel_col; j <= kernel_col; j++)
            {
                // if (j == 0 && i == 0)
                // {
                //     continue;
                // }
                density_kernel.push_back(std::make_pair(i, j));
            }
        }

// #ifdef _OPENMP

//         int total_threads = omp_get_num_threads();
//         int max_threads = omp_get_max_threads();
//         max_threads = std::max(max_threads, 4); // at least 4 threads (optional)
//         max_threads = 20;                       // remove this
//         omp_set_num_threads(max_threads);
// #pragma omp parallel
//         {
//             total_threads = omp_get_num_threads();
// #pragma omp master
//             {
//                 std::cout << "Total OpenMP threads used : " << total_threads << std::endl;
//             }
//         }
// #endif
        localKdTree_map.reset(new pcl::KdTreeFLANN<PointType>());
        // localKdTree_map_als.reset(new pcl::KdTreeFLANN<PointType>());


        cloud_tree.reset(new pcl::KdTreeFLANN<PointType>());

        H_gnss = Eigen::Matrix<double, gps_dim, state_size>::Zero(); // 3 * n
        H_gnss.block<3, 3>(P_ID, P_ID) = Eye3d;

        H_se3 = Eigen::Matrix<double, se3_dim, state_size>::Zero();
        H_se3.block<3, 3>(P_ID, P_ID) = Eye3d;
        H_se3.block<3, 3>(R_ID, R_ID) = Eye3d;
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


int update_MLS(double R, PointCloudXYZI::Ptr &feats_down_body, const PointCloudXYZI::Ptr &map, int maximum_iter, bool extrinsic_est,
              const bool use_als, const PointCloudXYZI::Ptr &als_map, const pcl::KdTreeFLANN<PointType>::Ptr &als_tree,
              const bool use_se3, const Sophus::SE3 &gnss_se3, const V3D &gnss_std_pos_m, const V3D &gnss_std_rot_deg);



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

    void lidar_observation_model_tighly_fused(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                              PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als, const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    bool update_tighly_fused(double R, PointCloudXYZI::Ptr &feats_down_body,
                             PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als, const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est);

    void observation_model_test(const double R, residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body, const V3D &gps,
                                PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als, const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    bool update_tighly_fused_test(double R, PointCloudXYZI::Ptr &feats_down_body,
                                  PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                  const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                  std::vector<PointVector> &Nearest_Points,
                                  const V3D &gps, double R_gps_cov,
                                  int maximum_iter, bool extrinsic_est);

    void observation_model_test2(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                 const PointCloudXYZI::Ptr &feats_undistort,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                                 const V3D &gps,
                                 PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als, const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

    bool update_tighly_fused_test2(double R, PointCloudXYZI::Ptr &feats_down_body, PointCloudXYZI::Ptr &feats_undistort, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                                   PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                   const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                   std::vector<PointVector> &Nearest_Points,
                                   const V3D &gps, double R_gps_cov,
                                   int maximum_iter, bool extrinsic_est);

        
    //------------------------------------------------------------
    void h(residual_struct &ekfom_data, 
            double R_lidar_cov,double R_gps_cov, 
            const PointCloudXYZI::Ptr &feats_down_body, const V3D &gps_pos,
            PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als, 
            const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points, 
            bool extrinsic_est, bool use_gnss = false, bool use_als = false, bool tightly_coupled = true);

    bool update_final(
        double R_lidar_cov,double R_gps_cov, 
        PointCloudXYZI::Ptr &feats_down_body, const V3D &gps_pos,
        PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
        const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
        std::vector<PointVector> &Nearest_Points,
        int maximum_iter, bool extrinsic_est,
        bool use_gnss = false, bool use_als = false, bool tightly_coupled = true
    );

#endif
    // gnss update
    void update(const V3D &pos, const V3D &cov_pos_, int maximum_iter, bool global_error, M3D R = Eye3d);

    void update_se3(const Sophus::SE3 &measured_, int maximum_iter, const V3D &std_pos_m, const V3D &std_rot_deg);

    M3D computeCovariance(const PointCloudXYZI::Ptr &cloud,
                      const pcl::KdTreeFLANN<PointType>::Ptr &kdtree,
                      const PointType &point,
                      const PointCloudXYZI::Ptr &feats_undistort,
                      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                      V3D &out_mean,
                      int k_neighbors = 10,
                      bool use_radius = false,
                      float radius = 1.0);
};

#endif