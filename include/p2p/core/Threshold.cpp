
#include "Threshold.hpp"

namespace
{
    double ComputeModelError(const Sophus::SE3 &model_deviation, double max_range)
    {
        const double theta = Eigen::AngleAxisd(model_deviation.rotation_matrix()).angle();
        const double delta_rot = 2.0 * max_range * std::sin(theta / 2.0);
        const double delta_trans = model_deviation.translation().norm();
        return delta_trans + delta_rot;
    }
} // namespace

namespace p2p
{
    double AdaptiveThreshold::ComputeThreshold()
    {
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
} // namespace p2p
