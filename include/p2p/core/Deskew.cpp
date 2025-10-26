
#include "Deskew.hpp"

#include <tbb/parallel_for.h>

namespace
{
    constexpr double mid_pose_timestamp{0.5};
} // namespace

namespace kiss_icp
{
    // this will deskew the scan at the mid
    void DeSkewScan(std::vector<V3D_4> &frame,
                    const std::vector<double> &timestamps,
                    const Sophus::SE3 &start_pose,
                    const Sophus::SE3 &finish_pose)
    {
        const auto delta_pose = (start_pose.inverse() * finish_pose).log();
        tbb::parallel_for(size_t(0), frame.size(), [&](size_t i)
                          {
            const auto motion = Sophus::SE3::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);
            frame[i] = motion * frame[i]; });
    }

    void DeSkewScan_StartFrame(std::vector<V3D_4> &frame,
                               const std::vector<double> &timestamps,
                               const Sophus::SE3 &start_pose,
                               const Sophus::SE3 &finish_pose)
    {
        const auto delta_pose = (start_pose.inverse() * finish_pose).log();
        tbb::parallel_for(size_t(0), frame.size(), [&](size_t i)
                          {
                              const auto motion = Sophus::SE3::exp(timestamps[i] * delta_pose);
                              frame[i] = motion * frame[i]; });
    }

    void DeSkewScan_EndFrame(std::vector<V3D_4> &frame,
                             const std::vector<double> &timestamps,
                             const Sophus::SE3 &start_pose,
                             const Sophus::SE3 &finish_pose)
    {
        const auto delta_pose = (start_pose.inverse() * finish_pose).log();
        tbb::parallel_for(size_t(0), frame.size(), [&](size_t i)
                          {
                              const auto motion = Sophus::SE3::exp((1. - timestamps[i]) * delta_pose).inverse();
                              frame[i] = motion * frame[i]; });
    }

} // namespace kiss_icp
