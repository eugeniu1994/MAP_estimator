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
    inline double square(double x) { return x * x; }

    struct ResultTuple
    {
        ResultTuple()
        {
            JTJ.setZero();
            JTr.setZero();
        }

        ResultTuple operator+(const ResultTuple &other)
        {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        Eigen::Matrix6d JTJ; // state_size x state_size
        Eigen::Vector6d JTr; // state_size x 1
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

        // const auto &[JTJ, JTr]
        const ResultTuple &result = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<size_t>{0, source.size()},
            // Identity
            ResultTuple(),
            // 1st Lambda: Parallel computation
            [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple
            {
                auto &[JTJ_private, JTr_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                    double w = Weight(residual.squaredNorm());

                    JTJ_private.noalias() += J_r.transpose() * w * J_r;
                    JTr_private.noalias() += J_r.transpose() * w * residual;
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr] = result;

        return std::make_tuple(JTJ, JTr);
    }

    std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
        const std::vector<V3D> &source,
        const std::vector<Eigen::Matrix<double, 4, 1>> &target,
        const double kernel, const M3D &R)

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
            Eigen::Vector6d J_r;               // 6x1
            J_r.block<3, 1>(0, 0) = unit_norm; // df/dt

            // derivative w.r.t. in global frame
            J_r.block<3, 1>(3, 0) = source[i].cross(unit_norm);       //df/dR

            // local frame
            //M3D point_I_crossmat;
            //point_I_crossmat << SKEW_SYM_MATRX(source[i]);
            //J_r.block<3, 1>(3, 0) = point_I_crossmat * (R * unit_norm); // R is transposed already
            //           global = local
            //(R*p) × unit_norm = p×(R^⊤ * unit_norm)

            return std::make_tuple(J_r, target[i](3));
        };

        auto Weight = [&](double residual2)
        {
            return square(kernel) / square(kernel + residual2);
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
                auto &[JTJ_private, JTr_private] = J;
                for (auto i = r.begin(); i < r.end(); ++i)
                {
                    if (target[i](3) == -100) // point to point
                    {
                        const auto &[J_r, residual] = compute_jacobian_and_residual_points(i);
                        double w = Weight(residual.squaredNorm());
                        JTJ_private.noalias() += J_r.transpose() * w * J_r;
                        JTr_private.noalias() += J_r.transpose() * w * residual;
                    }
                    else // point to plane
                    {
                        const auto &[J_r, residual] = compute_jacobian_and_residual_planes(i);
                        double w = Weight(residual * residual);

                        JTJ_private.noalias() += J_r * w * J_r.transpose();
                        JTr_private.noalias() += J_r * w * residual;
                    }
                }
                return J;
            },
            // 2nd Lambda: Parallel reduction of the private Jacboians
            [&](ResultTuple a, const ResultTuple &b) -> ResultTuple
            { return a + b; });

        const auto &[JTJ, JTr] = result;

        return std::make_tuple(JTJ, JTr);
    }

} // namespace

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
            const auto &[src, tgt] = voxel_map.GetPlaneCorrespondences(source, max_correspondence_distance);
            auto R = (T_icp * initial_guess).so3().matrix().transpose();
            const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel, R);
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

} // namespace p2p