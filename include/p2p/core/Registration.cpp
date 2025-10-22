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

namespace
{

struct landmark
{
    int map_point_index;   // index of the point from the reference map
    int cloud_point_index; // index pf the points from the cloud
    V3D norm;              // the normal of the plane in global frame (normalized)
    double d;              // d parameter of the plane
    double var;            // plane measurement variance

    V3D tgt;
};


static std::vector<landmark> global_landmarks(100000);
static std::vector<bool> global_valid(100000, false);

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

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
        const std::vector<V3D> &source,
        const std::vector<V3D> &target,
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
                    double w = Weight(residual.squaredNorm());

                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;
                    cost_private += residual.norm();
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr, cost] = result;

        auto normalized_cost = cost/source.size();
        std::cout<<"normalized_cost:"<<normalized_cost<<std::endl;

        return std::make_tuple(JTJ, JTr);
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem_planes(
        const std::vector<V3D> &source,
        const std::vector<Eigen::Matrix<double, 4, 1>> &target,
        const double kernel)

    {
        auto compute_jacobian_and_residual_points = [&](auto i)
        {
            const V3D residual = source[i] - target[i].template head<3>();

            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                              // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(source[i]); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto compute_jacobian_and_residual_planes = [&](auto i)
        {
            const V3D &unit_norm = target[i].template head<3>();
            auto [H_point_wrt_pose, p2p_residual] = compute_jacobian_and_residual_points(i);  //J:3x6   r:3x1
            
            auto residual = (unit_norm).dot(p2p_residual);
            //auto residual = (p2p_residual).dot(unit_norm);

            Eigen::Matrix<double, 1, 6> J_r;
            J_r = (unit_norm.transpose() * H_point_wrt_pose);
            return std::make_tuple(J_r, residual);

            // Eigen::Vector6d J_r;               // 6x1
            // J_r.block<3, 1>(0, 0) = unit_norm; // df/dt

            // // derivative w.r.t. in global frame
            // J_r.block<3, 1>(3, 0) = source[i].cross(unit_norm); // df/dR

            // // local frame
            // // M3D point_I_crossmat;
            // // point_I_crossmat << SKEW_SYM_MATRX(source[i]);
            // // J_r.block<3, 1>(3, 0) = point_I_crossmat * (R * unit_norm); // R is transposed already
            // //           global = local
            // //(R*p) × unit_norm = p×(R^⊤ * unit_norm)

            // return std::make_tuple(J_r, target[i](3));
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
            //return (kernel * kernel) / ((kernel * kernel) + residual2);
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
                    // if (target[i](3) == -100) // point to point
                    // {
                    //     const auto &[J_r, residual] = compute_jacobian_and_residual_points(i);
                    //     double w = Weight(residual.squaredNorm());
                    //     JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    //     JTr_private.noalias() += J_r.transpose() * w * residual;

                    //     cost_private += residual.norm();
                    // }
                    // else // point to plane
                    // {
                        const auto &[J_r, residual] = compute_jacobian_and_residual_planes(i);
                        double w = Weight(residual * residual);

                        // JTJ_private.noalias() += J_r * w * J_r.transpose();
                        // JTr_private.noalias() += J_r * w * residual;

                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;

                        cost_private += (residual * residual);
                    //}
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
        const std::vector<V3D> &source,
        const std::vector<V3D> &target,
        const std::vector<Eigen::Matrix<double, 4, 1>> &plane,
        const double kernel)

    {
        /*
        source is transformed to map frame
        target and plane are in the map frame
        */

        auto compute_jacobian_and_residual_points = [&](auto i)
        {
            const V3D residual = source[i] - target[i];

            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                              // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(source[i]); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto compute_jacobian_and_residual_planes = [&](auto i)
        {
            V3D unit_norm = plane[i].template head<3>();
            auto d_ = plane[i](3);

            auto [H_point_wrt_pose, p2p_residual] = compute_jacobian_and_residual_points(i);  //J:3x6   r:3x1

            //auto residual = unit_norm.dot(source[i]) + d_;
            auto residual = (unit_norm).dot(p2p_residual);

            Eigen::Matrix<double, 1, 6> J_r;

            J_r = (unit_norm.transpose() * H_point_wrt_pose);
            return std::make_tuple(J_r, residual);

            //J_r.block<1, 3>(0, 0) = unit_norm.transpose();  //derivative w.r.t to translation 
            //J_r.block<1, 3>(0, 3) = source[i].cross(unit_norm); //derivative w.r.t. rotation 

                    
            //fails because of the frame in which the src points are
            // Use rotated point in local frame before computing rotation jacobian.
            // Eigen::Vector3d p_local = R_T * source[i];
            // J_r.block<1, 3>(0, 3) = -(p_local.cross(unit_norm)).transpose(); //derivative w.r.t. rotation 
            

            //M3D point_I_crossmat; //issue here source[i] should be in sensor frame 
            //point_I_crossmat << SKEW_SYM_MATRX(source[i]);
            //J_r.block<1, 3>(0, 3) = (point_I_crossmat * (R_T * unit_norm)).transpose(); // R is transposed already
            //           global = local
            //(R*p) × unit_norm = p×(R^⊤ * unit_norm)

            //

            //return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            //return square(kernel) / square(kernel + residual2);

            return (kernel*kernel) / (kernel*kernel + residual2);
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
                    double w = Weight(residual * residual);

                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;

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
        const std::vector<V3D> &source,
        const double kernel)

    {
        auto compute_jacobian_and_residual_points = [&](auto i)
        {
            const landmark &land = global_landmarks[i];

            const auto &src = source[land.cloud_point_index];
            const V3D residual = src - land.tgt;

            Eigen::Matrix3_6d J_r;
            J_r.block<3, 3>(0, 0) = Eye3d;                        // df/dt
            J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(src); // df/dR
            return std::make_tuple(J_r, residual);
        };

        auto compute_jacobian_and_residual_planes = [&](auto i)
        {
            const landmark &landmark = global_landmarks[i];

            //V3D unit_norm = plane[i].template head<3>();

            auto [H_point_wrt_pose, p2p_residual] = compute_jacobian_and_residual_points(i);  //J:3x6   r:3x1

            auto residual = (p2p_residual).dot(landmark.norm);

            Eigen::Matrix<double, 1, 6> J_r;
            J_r = (landmark.norm.transpose() * H_point_wrt_pose);
            return std::make_tuple(J_r, residual);
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
            //return (kernel*kernel) / (kernel*kernel + residual2);
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
                    if(global_valid[i])
                    {
                        const auto &[J_r, residual] = compute_jacobian_and_residual_planes(i);
                        double w = Weight(residual * residual);

                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;

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

} // namespace

#ifdef _OPENMP
#include <omp.h>
#endif

namespace p2p
{
    Sophus::SE3 RegisterPoint(const std::vector<V3D> &frame,
                              const VoxelHashMap &voxel_map,
                              const Sophus::SE3 &initial_guess,
                              double max_correspondence_distance,
                              double kernel)
    {
        if (voxel_map.Empty())
        {
            return initial_guess;
        }

        std::vector<V3D> source = frame;
        TransformPoints(initial_guess, source);

        // ICP-loop
        Sophus::SE3 T_icp = Sophus::SE3();

        for (int j = 0; j <= MAX_NUM_ITERATIONS_; ++j)
        {
            const auto &[src, tgt] = voxel_map.GetPointCorrespondences(source, max_correspondence_distance);
            const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
            const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);   // translation and rotation perturbations
            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
            T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= MAX_NUM_ITERATIONS_)
            {
                std::cout << "Points registered with src:" << src.size() << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
        }

        return T_icp * initial_guess; // put in global, using the init guess
    }

    Sophus::SE3 RegisterPlane(const std::vector<V3D> &frame,
                              const VoxelHashMap &voxel_map,
                              const Sophus::SE3 &initial_guess,
                              double max_correspondence_distance,
                              double kernel, bool save_nn)
    {
        if (voxel_map.Empty())
        {
            return initial_guess;
        }

        std::vector<V3D> source = frame;
        TransformPoints(initial_guess, source);

        // ICP-loop
        Sophus::SE3 T_icp = Sophus::SE3();
        int max_iter_ = 25;
        for (int j = 0; j <= max_iter_; ++j)
        {
            //auto R = (T_icp * initial_guess).so3().matrix().transpose();

            const auto &[src, tgt] = voxel_map.GetPlaneCorrespondences(source, max_correspondence_distance);
            const auto &[JTJ, JTr] = BuildLinearSystem_planes(src, tgt, kernel);
            const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);   // translation and rotation perturbations
            const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
            T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

            // Termination criteria
            if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= max_iter_)
            {
                std::cout << "Plane registered with src:" << src.size() << " in " << j << " iterations " << std::endl;
                break;
            }

            TransformPoints(estimation, source);
        }

        return T_icp * initial_guess; // put in global, using the init guess
    }

    constexpr double MAX_SQ_DIST = 1.0;
    //std::tuple<std::vector<V3D>, std::vector<V3D>, std::vector<Eigen::Matrix<double, 4, 1>>>
    void establishCorrespondences(const std::vector<V3D> &frame,
                             const pcl::PointCloud<PointType>::Ptr &map,
                             const pcl::KdTreeFLANN<PointType>::Ptr &tree)
    {
        // std::vector<V3D> src_points;
        // std::vector<V3D> tgt_points;
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

            const V3D &p_src = frame[i];
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
                    V3D tgt_point(p_tgt.x, p_tgt.y, p_tgt.z);

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
                        l.norm = V3D(pabcd(0), pabcd(1), pabcd(2));
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

    Sophus::SE3 RegisterPlane(const std::vector<V3D> &frame,
                              const PointCloudXYZI::Ptr &map,
                              const pcl::KdTreeFLANN<PointType>::Ptr &tree,
                              const Sophus::SE3 &initial_guess,
                              double max_correspondence_distance,
                              double kernel)
    {
        std::cout << "icp update ALS" << std::endl;

        std::vector<V3D> source = frame;
        TransformPoints(initial_guess, source); // transformed to map frame


        // required modification 

        // do not transform all the source points 
        //     do the transformation in the NN search - return the transformed ones 
        // in the establish correspondences - keep a predifined global landmarks and re-used them 
        // check for convergence when to update the DA correspondences 
        //     similar to iekf
        //     do register untill no more changes - then proceed with update DA 
        // everything for 5 iterations only 

        // ICP-loop 
        Sophus::SE3 T_icp = Sophus::SE3();
        int max_iter_ = 50;

        constexpr double MAX_SQ_DIST = 1.0;

        for (int j = 0; j <= max_iter_; ++j)
        {
            //auto R_T = (T_icp * initial_guess).so3().matrix().transpose();
            //const auto &[src, tgt, planes] = establishCorrespondences(source, map, tree);

            establishCorrespondences(source, map, tree);

            // std::cout<<"src:"<<src.size()<<", tgt:"<<tgt.size()<<", planes:"<<planes.size()<<std::endl;
            //if(src.size() > 0)
            //{
                //const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, planes, kernel, R_T);

                const auto &[JTJ, JTr] = BuildLinearSystem(source, kernel);

                //const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);   // translation and rotation perturbations
                double lambda = 1e-6;
                Eigen::Matrix6d JTJ_damped = JTJ;
                JTJ_damped += lambda * Eigen::Matrix6d::Identity();
                Eigen::Vector6d dx = JTJ_damped.ldlt().solve(-JTr);

                const Sophus::SE3 estimation = Sophus::SE3::exp(dx); // this is in local-align init guess to map
                T_icp = estimation * T_icp;                          // the amount of correction starting from init guess

                // Termination criteria
                if (dx.norm() < ESTIMATION_THRESHOLD_ || j >= max_iter_)
                {
                    //std::cout<<"src:"<<src.size()<<", tgt:"<<tgt.size()<<", planes:"<<planes.size()<<std::endl;
                    std::cout << "Plane registered with source:" << source.size() << " in " << j << " iterations " << std::endl;
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
} // namespace p2p