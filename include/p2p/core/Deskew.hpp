
#ifndef COMMON_DESKEW_H1
#define COMMON_DESKEW_H1

#pragma once

#include "../../utils.h"

namespace p2p
{
    /// Compensate the frame by estimatng the velocity between the given poses
    void DeSkewScan(std::vector<V3D_4> &frame,
                    const std::vector<double> &timestamps,
                    const Sophus::SE3 &start_pose,
                    const Sophus::SE3 &finish_pose);

    void DeSkewScan_StartFrame(std::vector<V3D_4> &frame,
                               const std::vector<double> &timestamps,
                               const Sophus::SE3 &start_pose,
                               const Sophus::SE3 &finish_pose);

    void DeSkewScan_EndFrame(std::vector<V3D_4> &frame,
                             const std::vector<double> &timestamps,
                             const Sophus::SE3 &start_pose,
                             const Sophus::SE3 &finish_pose);

} // namespace p2p

#endif