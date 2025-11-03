#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <tuple>

namespace Eigen
{
    const int state_size = 6;
    using Matrix6d = Eigen::Matrix<double, state_size, state_size>;
    using Matrix3_6d = Eigen::Matrix<double, 3, state_size>;
    using Vector6d = Eigen::Matrix<double, state_size, 1>;
} // namespace Eigen

// #define use_motion_correction_uncertainty

namespace
{

    struct landmark
    {
        int map_point_index;   // index of the point from the reference map
        int cloud_point_index; // index pf the points from the cloud
        V3D_4 norm;            // the normal of the plane in global frame (normalized)
        double d;              // d parameter of the plane
        double var;            // plane measurement variance
        V3D_4 tgt;
    };

    static std::vector<landmark> global_landmarks(100000);
    static std::vector<bool> global_valid(100000, false);

    static std::vector<V3D_4> local_src(100000);
    static std::vector<V3D_4> local_tgt(100000);
    static std::vector<bool> local_valid(100000, false);

    double max_displacement = 0.0, inv_max_displacement = 1.0;

    inline double square(double x) { return x * x; }

    struct ResultTuple
    {
        ResultTuple()
        {
            JTJ.setZero();
            JTr.setZero();
            cost = 0.0;
        }

        ResultTuple operator+(const ResultTuple &other)
        {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            this->cost += other.cost;
            return *this;
        }

        Eigen::Matrix6d JTJ; // state_size x state_size
        Eigen::Vector6d JTr; // state_size x 1

        double cost;
    };

    double ComputePointWeight(double sensor_stddev, double displacement)
    {
        // Total uncertainty = sensor noise + motion correction uncertainty
        double motion_stddev = displacement / 3.0; // Convert max displacement to stddev

        // Combined standard deviation
        double total_stddev = std::sqrt(sensor_stddev * sensor_stddev + motion_stddev * motion_stddev);

        // Weight is inverse of variance (information)
        double variance = total_stddev * total_stddev;
        return 1.0 / variance; // w = 1/σ²
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
        const std::vector<V3D_4> &source,
        const std::vector<V3D_4> &target,
        const double kernel)

    {
        auto compute_jacobian_and_residual = [&](auto i)
        {
            const V3D residual = source[i] - target[i];
            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                              // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(source[i]); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
        };

        // const auto &[JTJ, JTr, cost]
        const ResultTuple &result = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, source.size()},
            // Identity
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
            {
                auto &[JTJ_private, JTr_private, cost_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(i);

                    double w_robust = Weight(residual.squaredNorm());

#ifdef use_motion_correction_uncertainty
                    // double w_uncertainty = ComputePointWeight(0.01, source[i].displacement);
                    //  Fast normalized uncertainty weight: 1.0 (best) to near 0 (worst)
                    // double combined_disp = 0.5 * (source[i].displacement + target[i].displacement);
                    double w_uncertainty = 1.0 - (source[i].displacement * inv_max_displacement);
                    w_uncertainty = std::max(w_uncertainty, 0.01); // Keep minimal weight
                    double total_weight = w_robust * w_uncertainty;
#else
                    double total_weight = w_robust;
#endif
                    JTJ_private.noalias() += J_r.transpose() * total_weight * J_r;
                    JTr_private.noalias() += J_r.transpose() * total_weight * residual;
                    cost_private += residual.norm();
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr, cost] = result;

        // auto normalized_cost = cost/source.size();
        // std::cout<<"normalized_cost:"<<normalized_cost<<std::endl;

        return std::make_tuple(JTJ, JTr);
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
        const std::vector<V3D_4> &source,
        const std::vector<V3D_4> &target,
        const std::vector<Eigen::Matrix<double, 4, 1>> &plane,
        const double kernel)

    {
        /*
        source is transformed to map frame
        target and plane are in the map frame
        */

        auto compute_jacobian_and_residual_points = [&](auto i)
        {
            const V3D_4 residual = source[i] - target[i];

            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                              // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(source[i]); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto compute_jacobian_and_residual_planes = [&](auto i)
        {
            V3D unit_norm = plane[i].template head<3>();
            auto d_ = plane[i](3);

            auto [H_point_wrt_pose, p2p_residual] = compute_jacobian_and_residual_points(i); // J:3x6   r:3x1

            // auto residual = unit_norm.dot(source[i]) + d_;
            auto residual = (unit_norm).dot(p2p_residual);

            Eigen::Matrix<double, 1, 6> J_r;
            J_r = (unit_norm.transpose() * H_point_wrt_pose);
            return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            // return square(kernel) / square(kernel + residual2);

            return (kernel * kernel) / (kernel * kernel + residual2);
        };

        // const auto &[JTJ, JTr, cost]
        const ResultTuple &result = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, source.size()},
            // Identity
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
            {
                auto &[JTJ_private, JTr_private, cost_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    const auto &[J_r, residual] = compute_jacobian_and_residual_planes(i);
                    double w_robust = Weight(residual * residual);

#ifdef use_motion_correction_uncertainty
                    // double w_uncertainty = ComputePointWeight(0.01, source[i].displacement);
                    //  Fast normalized uncertainty weight: 1.0 (best) to near 0 (worst)
                    // double combined_disp = 0.5 * (source[i].displacement + target[i].displacement);
                    double w_uncertainty = 1.0 - (source[i].displacement * inv_max_displacement);
                    w_uncertainty = std::max(w_uncertainty, 0.01); // Keep minimal weight
                    double total_weight = w_robust * w_uncertainty;
#else
                    double total_weight = w_robust;
#endif

                    JTJ_private.noalias() += J_r.transpose() * total_weight * J_r;
                    JTr_private.noalias() += J_r.transpose() * total_weight * residual;

                    cost_private += (residual * residual);
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr, cost] = result;

        return std::make_tuple(JTJ, JTr);
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
        const std::vector<V3D_4> &source,
        const double kernel)
    {
        auto compute_jacobian_and_residual_points = [&](auto i)
        {
            const landmark &land = global_landmarks[i];

            const auto &src = source[land.cloud_point_index];
            const V3D_4 residual = src - land.tgt;

            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                        // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(src); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto compute_jacobian_and_residual_planes = [&](auto i)
        {
            const landmark &landmark = global_landmarks[i];
            auto [H_point_wrt_pose, p2p_residual] = compute_jacobian_and_residual_points(i); // J:3x6   r:3x1

            auto residual = (p2p_residual).dot(landmark.norm);

            Eigen::Matrix<double, 1, 6> J_r;
            J_r = (landmark.norm.transpose() * H_point_wrt_pose);
            return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
            // return (kernel*kernel) / (kernel*kernel + residual2);
        };

        // const auto &[JTJ, JTr]
        const ResultTuple &result = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, source.size()},
            // Identity
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
            {
                auto &[JTJ_private, JTr_private, cost_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    if (global_valid[i])
                    {
                        const auto &[J_r, residual] = compute_jacobian_and_residual_planes(i);
                        double w_robust = Weight(residual * residual);

#ifdef use_motion_correction_uncertainty
                        // double w_uncertainty = ComputePointWeight(0.01, source[i].displacement);
                        //  Fast normalized uncertainty weight: 1.0 (best) to near 0 (worst)
                        double w_uncertainty = 1.0 - (source[i].displacement * inv_max_displacement);
                        w_uncertainty = std::max(w_uncertainty, 0.01); // Keep minimal weight
                        double total_weight = w_robust * w_uncertainty;
#else
                        double total_weight = w_robust;
#endif
                        // find this parameter based on the number of p2p and p2pln correspondences - normalize it

                        JTJ_private.noalias() += J_r.transpose() * total_weight * J_r;
                        JTr_private.noalias() += J_r.transpose() * total_weight * residual;

                        cost_private += (residual * residual);
                    }
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr, cost] = result;

        return std::make_tuple(JTJ, JTr);
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem_preallocated(
        const int &N,
        const double kernel)
    {
        auto compute_jacobian_and_residual = [&](auto i)
        {
            const V3D residual = local_src[i] - local_tgt[i];
            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                                 // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(local_src[i]); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
        };

        // const auto &[JTJ, JTr, cost]
        const ResultTuple &result = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, N},
            // Identity
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
            {
                auto &[JTJ_private, JTr_private, cost_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    if (local_valid[i])
                    {
                        const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                        double w_robust = Weight(residual.squaredNorm());

#ifdef use_motion_correction_uncertainty
                        // double w_uncertainty = ComputePointWeight(0.01, local_src[i].displacement);
                        //  Fast normalized uncertainty weight: 1.0 (best) to near 0 (worst)
                        // double combined_disp = 0.5 * (local_src[i].displacement + target[i].displacement);
                        double w_uncertainty = 1.0 - (local_src[i].displacement * inv_max_displacement);
                        w_uncertainty = std::max(w_uncertainty, 0.01); // Keep minimal weight w_uncertainty = std::clamp(w_uncertainty, 0.1, 1.0);
                        //w_uncertainty = std::max(w_uncertainty, 0.5);

                        double total_weight = w_robust * w_uncertainty;

                        // double motion_stddev = std::max(0.01, (local_src[i].displacement * inv_max_displacement) / 3.0); // Convert normalized displacement to stddev
                        // //double motion_stddev = .01; //1cm

                        // double variance = motion_stddev * motion_stddev; // Weight is inverse of variance (information)
                        // Eigen::Matrix<double, 6, 6> C_xi = variance * Eigen::Matrix<double, 6, 6>::Identity();
                        // M3D C_p = J_r * C_xi.transpose() * J_r.transpose();

                        // // Suppose you have covariance in sensor frame
                        // Eigen::Matrix3d C_local = motion_covariance_for_point(i);  // 3x3 in LiDAR frame
                        // Eigen::Matrix3d C_map = R_map_sensor * C_local * R_map_sensor.transpose();
                        // Eigen::Matrix3d W = (C_map + 1e-6 * Eigen::Matrix3d::Identity()).inverse();

                        // M3D W = (C_p + 1e-6 * M3D::Identity()).inverse(); // small 3x3 inverse is cheap

#else
                        double total_weight = w_robust;
#endif
                        JTJ_private.noalias() += J_r.transpose() * total_weight * J_r;
                        JTr_private.noalias() += J_r.transpose() * total_weight * residual;
                        cost_private += residual.norm();
                    }
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr, cost] = result;

        // auto normalized_cost = cost/source.size();
        // std::cout<<"normalized_cost:"<<normalized_cost<<std::endl;

        return std::make_tuple(JTJ, JTr);
    }
} // namespace

#ifdef _OPENMP
#include <omp.h>
#endif

#include <chrono>

// Utility timer alias
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

#define START_TIMER(name) auto start_##name = Clock::now()
#define STOP_TIMER(name, label)                                                                             \
    std::cout << label << ": "                                                                              \
              << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_##name).count() \
              << " ms" << std::endl;

namespace p2p
{
    double computeMaxDisplacement(const std::vector<V3D_4> &points)
    {
        double max_disp = 0.0;
#pragma omp parallel for reduction(max : max_disp)
        for (size_t i = 0; i < points.size(); ++i)
        {
            max_disp = std::max(max_disp, points[i].displacement);
        }
        // Avoid division by zero
        if (max_disp < 1e-6)
            max_disp = 1e-6;
        std::cout << "max_displacement: " << max_disp << std::endl;
        return max_disp;
    };

    double computeMaxDisplacement_map(const std::vector<V3D_4> &src, const std::vector<V3D_4> &tgt)
    {
        double max_disp = 0.0;
#pragma omp parallel for reduction(max : max_disp)
        for (size_t i = 0; i < src.size(); ++i)
        {
            max_disp = std::max(max_disp, std::max(src[i].displacement, tgt[i].displacement));
        }

        // Avoid division by zero
        if (max_disp < 1e-6)
            max_disp = 1e-6;
        std::cout << "max_displacement with map: " << max_disp << std::endl;
        return max_disp;
    };

    // Precomputed 3×3×3 neighborhood offsets (executed once)
    static const std::vector<Eigen::Vector3i> kVoxelOffsets = []
    {
        std::vector<Eigen::Vector3i> offsets;
        offsets.reserve(27);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz)
                    offsets.emplace_back(dx, dy, dz);
        return offsets;
    }();

    constexpr double MAX_SQ_DIST = 1.0;
    // std::tuple<std::vector<V3D_4>, std::vector<V3D_4>, std::vector<Eigen::Matrix<double, 4, 1>>>
    void establishCorrespondences(const std::vector<V3D_4> &frame,
                                  const pcl::PointCloud<PointType>::Ptr &map,
                                  const pcl::KdTreeFLANN<PointType>::Ptr &tree)
    {
        // std::vector<V3D_4> src_points;
        // std::vector<V3D_4> tgt_points;
        // std::vector<Eigen::Matrix<double, 4, 1>> plane_coeffs;

        // src_points.reserve(frame.size());
        // tgt_points.reserve(frame.size());
        // plane_coeffs.reserve(frame.size());

#ifdef MP_EN
#pragma omp parallel for
#endif
        for (int i = 0; i < static_cast<int>(frame.size()); i++)
        {
            global_valid[i] = false;

            const V3D_4 &p_src = frame[i];
            PointType point_world;
            point_world.x = p_src.x();
            point_world.y = p_src.y();
            point_world.z = p_src.z();

            std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
            std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

            if (tree->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            {
                if (pointSearchSqDis.back() <= MAX_SQ_DIST)
                {
                    const auto &p_tgt = map->points[pointSearchInd[0]];
                    V3D_4 tgt_point(p_tgt.x, p_tgt.y, p_tgt.z);

                    PointVector points_near;
                    std::vector<double> point_weights;
                    points_near.reserve(NUM_MATCH_POINTS);
                    point_weights.reserve(NUM_MATCH_POINTS);
                    for (int j = 0; j < pointSearchInd.size(); j++)
                    {
                        points_near.push_back(map->points[pointSearchInd[j]]);
                        point_weights.push_back(1.);
                    }

                    double plane_threshold = .1;

                    // plane coefficients [nx, ny, nz, d]
                    Eigen::Matrix<double, 4, 1> pabcd;
                    //                     if (ekf::esti_plane(pabcd, points_near, plane_threshold)) // good plane
                    //                     {
                    // #pragma omp critical
                    //                         {
                    //                             src_points.push_back(p_src);
                    //                             tgt_points.push_back(tgt_point);
                    //                             plane_coeffs.push_back(pabcd);
                    //                         }
                    //                     }

                    double plane_var = 0;
                    if (ekf::esti_plane(pabcd, points_near, plane_threshold))
                    {
                        landmark l;
                        l.map_point_index = pointSearchInd[0];
                        l.cloud_point_index = i;
                        l.norm = V3D_4(pabcd(0), pabcd(1), pabcd(2));
                        l.d = pabcd(3);
                        l.var = plane_var;

                        l.tgt = tgt_point;

                        global_valid[i] = true;
                        global_landmarks[i] = l;
                    }
                }
            }
        }

        // return {src_points, tgt_points, plane_coeffs};
    }

    void establishCorrespondences_hash(const std::vector<V3D_4> &frame,
                                       const VoxelHashMap &mls_map,
                                       double max_correspondance_distance)
    {
        // Lambda Function to obtain the KNN of one point, maybe refactor
        auto GetClosestNeighboor = [&](const V3D_4 &point)
        {
            const auto &voxel = PointToVoxel(point, mls_map.inv_voxel_size_);
            size_t current_index = 0;
            std::vector<Voxel> voxels(27);
            for (int i = voxel.x() - 1; i <= voxel.x() + 1; ++i)
            {
                for (int j = voxel.y() - 1; j <= voxel.y() + 1; ++j)
                {
                    for (int k = voxel.z() - 1; k <= voxel.z() + 1; ++k)
                    {
                        voxels[current_index++] = Voxel(i, j, k);
                    }
                }
            }

            V3D_4 closest_neighbor = Zero3d;
            double closest_distance2 = std::numeric_limits<double>::max(), distance;
            std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &query_voxel)
                          {
                auto search = mls_map.map_.find(query_voxel);
                if (search != mls_map.map_.end()) {
                    const auto &points = search->second.points;
                    const V3D_4 &neighbor = *std::min_element(
                        points.cbegin(), points.cend(), [&](const auto &lhs, const auto &rhs) {
                            return (lhs - point).squaredNorm() < (rhs - point).squaredNorm();
                        });
                    distance = (neighbor - point).squaredNorm();
                    if (distance < closest_distance2) {
                        closest_neighbor = neighbor;
                        closest_distance2 = distance;
                    }
                } });

            return std::make_tuple(closest_neighbor, closest_distance2);
        };

        double max_correspondance_distance_2 = max_correspondance_distance * max_correspondance_distance;

        const size_t N = frame.size();

        if (N > 100000)
        {
            std::cout << "N:" << N << std::endl;
            throw std::runtime_error("Static buffers too small, please increase the static buffers size");
        }

        tbb::parallel_for(size_t(0), N, [&](size_t i)
                          {
                              const auto &point = frame[i];

                              const auto &[closest_neighboors, closest_sq_distance] = GetClosestNeighboor(point);
                              if (closest_sq_distance < max_correspondance_distance_2)
                              {
                                  local_valid[i] = true;
                                  local_src[i] = point;
                                  local_tgt[i] = closest_neighboors;
                              }
                              else
                              {
                                   local_valid[i] = false;
                              } });
    }

    void establishCorrespondences_hash_improved(const std::vector<V3D_4> &frame,
                                                const VoxelHashMap &mls_map,
                                                double max_correspondance_distance)
    {
        // Sanity check (avoid buffer overflow)
        if (frame.size() > local_src.size())
            throw std::runtime_error("Static buffer too small for current frame size");

        const double max_correspondance_distance_2 =
            max_correspondance_distance * max_correspondance_distance;

        const size_t N = frame.size();

        tbb::parallel_for(size_t(0), N, [&](size_t i)
                          {
        const V3D_4 &point = frame[i];

        // --- Compute voxel of current point ---
        const auto voxel = PointToVoxel(point, mls_map.inv_voxel_size_);

        // --- Search 3x3x3 neighborhood ---
        V3D_4 closest_neighbor = Zero3d;
        double closest_distance2 = std::numeric_limits<double>::max();

        for (const auto &offset : kVoxelOffsets)
        {
            const Voxel query_voxel = voxel + offset; // requires operator+(Voxel, Eigen::Vector3i)
            auto search = mls_map.map_.find(query_voxel);
            if (search == mls_map.map_.end())
                continue;

            const auto &points = search->second.points;
            const V3D_4 &neighbor = *std::min_element(
                points.cbegin(), points.cend(),
                [&](const auto &lhs, const auto &rhs) {
                    return (lhs - point).squaredNorm() < (rhs - point).squaredNorm();
                });

            double distance2 = (neighbor - point).squaredNorm();
            if (distance2 < closest_distance2)
            {
                closest_distance2 = distance2;
                closest_neighbor = neighbor;
            }
        }

        // --- Store valid matches only ---
        if (closest_distance2 < max_correspondance_distance_2)
        {
            local_valid[i] = true;
            local_src[i] = point;
            local_tgt[i] = closest_neighbor;
        }
        else
        {
            local_valid[i] = false;
        } });
    }

    Sophus::SE3 RegisterPoint(const std::vector<V3D_4> &frame,
                              const VoxelHashMap &voxel_map,
                              const Sophus::SE3 &initial_guess,
                              double max_correspondence_distance,
                              double kernel)
    {
        if (voxel_map.Empty())
        {
            return initial_guess;
        }

        int N = frame.size();

        std::vector<V3D_4> source = frame;
        TransformPoints(initial_guess, source);

        // ICP-loop
        Sophus::SE3 T_icp = Sophus::SE3();

        #ifdef use_motion_correction_uncertainty
        max_displacement = computeMaxDisplacement(source);
        // const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
        // max_displacement = computeMaxDisplacement_map(src, tgt);
        inv_max_displacement = 1.0 / max_displacement;
        #endif
        /*
        if (true)
        {
            std::cout << "\n========== Approach 1: Direct GetPointCorrespondences ==========\n";
            {
                START_TIMER(a1_corresp);
                const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
                for(int i=0;i<50;i++)
                {
                    const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
                }
                STOP_TIMER(a1_corresp, "  → Correspondence search time");

                START_TIMER(a1_build);
                const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
                for(int i=0;i<50;i++)
                {
                    const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
                }
                STOP_TIMER(a1_build, "  → Linear system build time");

                std::cout << "  Total (Approach 1): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_a1_corresp).count() + std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_a1_build).count()
                          << " ms\n";
            }

            std::cout << "\n========== Approach 2: Static Preallocated Buffers ==========\n";
            {
                START_TIMER(a2_corresp);
                establishCorrespondences_hash(source, voxel_map, max_correspondence_distance);
                for(int i=0;i<50;i++)
                {
                    establishCorrespondences_hash(source, voxel_map, max_correspondence_distance);
                }
                STOP_TIMER(a2_corresp, "  → Correspondence search time");

                START_TIMER(a2_build);
                const auto &[JTJ, JTr] = BuildLinearSystem_preallocated(N, kernel);
                for(int i=0;i<50;i++)
                {
                    const auto &[JTJ, JTr] = BuildLinearSystem_preallocated(N, kernel);
                }
                STOP_TIMER(a2_build, "  → Linear system build time");

                std::cout << "  Total (Approach 2): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_a2_corresp).count() + std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_a2_build).count()
                          << " ms\n";
            }

            std::cout << "\n========== Approach 3: Improved Correspondence Search ==========\n";
            {
                START_TIMER(a3_corresp);
                establishCorrespondences_hash_improved(source, voxel_map, max_correspondence_distance);
                for(int i=0;i<50;i++)
                {
                    establishCorrespondences_hash_improved(source, voxel_map, max_correspondence_distance);
                }
                STOP_TIMER(a3_corresp, "  → Correspondence search time");

                // Uncomment when BuildLinearSystem_preallocated ready
                START_TIMER(a3_build);
                const auto &[JTJ, JTr] = BuildLinearSystem_preallocated(N, kernel);
                for(int i=0;i<50;i++)
                {
                    const auto &[JTJ, JTr] = BuildLinearSystem_preallocated(N, kernel);
                }
                STOP_TIMER(a3_build, "  → Linear system build time");

                std::cout << "  Total (Approach 3): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_a3_corresp).count()
                          << " ms\n";
            }

            std::cout << "==============================================================\n";
        }
        */

        for (int j = 0; j <= MAX_NUM_ITERATIONS_; ++j)
        {
            // const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
            // const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);

            // establishCorrespondences_hash(source, voxel_map, max_correspondence_distance);
            establishCorrespondences_hash_improved(source, voxel_map, max_correspondence_distance);
            const auto &[JTJ, JTr] = BuildLinearSystem_preallocated(N, kernel);

            const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);   // translation and rotation perturbations
            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
            T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= MAX_NUM_ITERATIONS_)
            {
                std::cout << "RegisterPoint with src:" << N << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
        }

        return T_icp * initial_guess; // put in global, using the init guess
    }

    // Numerical Jacobian for pose residual:
    // r(T) = Log( T_meas^{-1} * T )
    // compute J = dr / d(dx) with left-multiplicative update T <- exp(dx) * T
    Eigen::Matrix<double, 6, 6> numerical_pose_jacobian(const Sophus::SE3 &T,
                                                        const Sophus::SE3 &Tmeas,
                                                        double eps = 1e-6)
    {
        Eigen::Matrix<double, 6, 6> J;
        Eigen::Matrix<double, 6, 1> r0 = (Tmeas.inverse() * T).log();

        for (int k = 0; k < 6; ++k)
        {
            Eigen::Matrix<double, 6, 1> dx = Eigen::Matrix<double, 6, 1>::Zero();
            dx[k] = eps;
            Sophus::SE3 Tp = Sophus::SE3::exp(dx) * T; // left update
            Eigen::Matrix<double, 6, 1> r_plus = (Tmeas.inverse() * Tp).log();
            J.col(k) = (r_plus - r0) / eps;
        }
        return J;
    }

    Sophus::SE3 RegisterPointAndGNSS(const Sophus::SE3 &T_measured, const std::vector<V3D_4> &frame,
                                     const VoxelHashMap &voxel_map,
                                     const Sophus::SE3 &initial_guess,
                                     double max_correspondence_distance,
                                     double kernel)
    {
        if (voxel_map.Empty())
        {
            return initial_guess;
        }

        std::vector<V3D_4> source = frame;
        TransformPoints(initial_guess, source);

        // ICP-loop
        Sophus::SE3 T_icp = Sophus::SE3();

        // Noise parameters
        // double lidar_point_std_ = 0.01;    // 1cm for LiDAR points
        double se3_trans_std_ = .2;               // 1 m for SE3 translation
        double se3_rot_std_ = 5.0 * M_PI / 180.0; // 5 degree for rotation

        Eigen::Matrix<double, 6, 6> se3_info_matrix_;
        // SE3 information matrix - block diagonal for rotation and translation
        se3_info_matrix_.setZero();
        se3_info_matrix_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() / (se3_trans_std_ * se3_trans_std_);
        se3_info_matrix_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() / (se3_rot_std_ * se3_rot_std_);
        #ifdef use_motion_correction_uncertainty
        max_displacement = computeMaxDisplacement(source);
        inv_max_displacement = 1.0 / max_displacement;
        #endif
        for (int j = 0; j <= MAX_NUM_ITERATIONS_; ++j)
        {
            const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
            const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);

            Eigen::Matrix6d H; // state_size x state_size
            Eigen::Vector6d b; // state_size x 1
            H.setZero();
            b.setZero();

            H = JTJ;
            b = JTr;
            {
                // T_current, T_gnss,
                auto T_current = T_icp * initial_guess;

                Eigen::Matrix<double, 6, 1> r_gnss = (T_current.inverse() * T_measured).log();
                Eigen::Matrix<double, 6, 6> J_gnss;

                r_gnss = (T_measured.inverse() * T_current).log(); // 6x1
                J_gnss = numerical_pose_jacobian(T_current, T_measured, 1e-6);
                std::cout << "numerical_pose_jacobian:\n"
                          << J_gnss << std::endl;
                // J_gnss.setIdentity();

                Eigen::Matrix6d H_se3;
                Eigen::Vector6d g_se3;

                H_se3 = J_gnss.transpose() * se3_info_matrix_ * J_gnss;
                g_se3 = J_gnss.transpose() * se3_info_matrix_ * r_gnss;

                H = JTJ + H_se3;
                b = JTr + g_se3;
            }

            const Eigen::Vector6d dx = H.ldlt().solve(-b);       // translation and rotation perturbations
            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
            T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= MAX_NUM_ITERATIONS_)
            {
                std::cout << "RegisterPointAndGNSS with src:" << src.size() << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
        }

        return T_icp * initial_guess; // put in global, using the init guess
    }

    Sophus::SE3 RegisterPlane(const std::vector<V3D_4> &frame,
                              const PointCloudXYZI::Ptr &map,
                              const pcl::KdTreeFLANN<PointType>::Ptr &tree,
                              const Sophus::SE3 &initial_guess,
                              double max_correspondence_distance,
                              double kernel)
    {
        std::cout << "icp update ALS" << std::endl;

        std::vector<V3D_4> source = frame;
        TransformPoints(initial_guess, source); // transformed to map frame

        // ICP-loop
        Sophus::SE3 T_icp = Sophus::SE3();
        int max_iter_ = 50;
        #ifdef use_motion_correction_uncertainty
        max_displacement = computeMaxDisplacement(source);
        inv_max_displacement = 1.0 / max_displacement;
        #endif
        for (int j = 0; j <= max_iter_; ++j)
        {
            // const auto &[src, tgt, planes] = establishCorrespondences(source, map, tree);

            establishCorrespondences(source, map, tree);

            // std::cout<<"src:"<<src.size()<<", tgt:"<<tgt.size()<<", planes:"<<planes.size()<<std::endl;
            // if(src.size() > 0)
            //{

            const auto &[JTJ, JTr] = BuildLinearSystem(source, kernel);

            // const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);   // translation and rotation perturbations
            double lambda = 1e-6;
            Eigen::Matrix6d JTJ_damped = JTJ;
            JTJ_damped += lambda * Eigen::Matrix6d::Identity();
            Eigen::Vector6d dx = JTJ_damped.ldlt().solve(-JTr);

            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
            T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= max_iter_)
            {
                // std::cout<<"src:"<<src.size()<<", tgt:"<<tgt.size()<<", planes:"<<planes.size()<<std::endl;
                std::cout << "RegisterPlane with source:" << source.size() << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
            // }
            // else
            // {
            //     std::cout<<"No point correspondences with ALS ..."<<std::endl;
            //     break;
            // }
        }

        return T_icp * initial_guess; // put in global, using the init guess;
    }

    Sophus::SE3 RegisterTightly(const std::vector<V3D_4> &frame,
                                const VoxelHashMap &mls_map,
                                const PointCloudXYZI::Ptr &als_map,
                                const pcl::KdTreeFLANN<PointType>::Ptr &als_tree,
                                const Sophus::SE3 &initial_guess,
                                double max_correspondence_distance,
                                double kernel)

    {
        if (mls_map.Empty())
        {
            return initial_guess;
        }

        int N = frame.size();

        std::vector<V3D_4> source = frame;
        TransformPoints(initial_guess, source);

        double lambda = 1e-6;

        Sophus::SE3 T_icp = Sophus::SE3();
        #ifdef use_motion_correction_uncertainty
        max_displacement = computeMaxDisplacement(source);
        // const auto &[src_mls, tgt_mls] = mls_map.GetPointCorrespondences(source, max_correspondence_distance);
        // double max_displacement = computeMaxDisplacement_map(src_mls, tgt_mls);
        inv_max_displacement = 1.0 / max_displacement;
        #endif
        int max_iter_ = 50; // MAX_NUM_ITERATIONS_
        for (int j = 0; j <= max_iter_; ++j)
        {
            // MLS correspondences & contribution
            // const auto &[src_mls, tgt_mls] = mls_map.GetPointCorrespondences(source, max_correspondence_distance);
            // const auto &[JTJ_mls, JTr_mls] = BuildLinearSystem(src_mls, tgt_mls, kernel);
            establishCorrespondences_hash_improved(source, mls_map, max_correspondence_distance);
            const auto &[JTJ_mls, JTr_mls] = BuildLinearSystem_preallocated(N, kernel);

            // ALS correspondences & contribution
            establishCorrespondences(source, als_map, als_tree);
            const auto &[JTJ_als, JTr_als] = BuildLinearSystem(source, kernel);

            Eigen::Matrix6d JTJ_damped = JTJ_mls + 2. * JTJ_als;
            JTJ_damped.noalias() += lambda * Eigen::Matrix6d::Identity();
            Eigen::Vector6d JTr; // state_size x 1
            JTr.setZero();
            JTr = JTr_mls + 2. * JTr_als;

            const Eigen::Vector6d dx = JTJ_damped.ldlt().solve(-JTr); // translation and rotation perturbations
            const Sophus::SE3 estimation = Sophus::SE3::exp(dx);      // this is in local-align init guess to map
            T_icp = estimation * T_icp;                               // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= max_iter_)
            {
                std::cout << "RegisterTightly with src:" << N << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
        }

        return T_icp * initial_guess; // put in global, using the init guess
    }

} // namespace p2p