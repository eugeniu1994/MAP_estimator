#ifndef USE_ICP_H1
#define USE_ICP_H1

#include <Estimator.hpp>
#include <tuple>

#include "p2p/core/Threshold.hpp"
#include "p2p/core/VoxelHashMap.hpp"


class ICP : public Estimator
{
public:
    using Vector3dVector = std::vector<V3D>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

    Vector3dVector ALS_tgt;
    
    explicit ICP(const Config &config)
        : config_(config), local_map_(config.voxel_size, config.max_range, config.max_points_per_voxel), adaptive_threshold_(config.initial_threshold, config.min_motion_th, config.max_range)
    {
    }
    ICP() : ICP(Config{}) {}
    ~ICP() {}

    bool update(const std::vector<V3D> &src, const p2p::VoxelHashMap &local_map_, bool p2p_ = true, bool save_nn=false);
   
    void LocalMap(PointCloudXYZI::Ptr &map_points) const;
    std::vector<Sophus::SE3> poses() const { return poses_; };
    std::vector<Sophus::SE3> poses_;

    Vector3dVectorTuple Voxelize(PointCloudXYZI::Ptr &frame, bool deskew = false) const;
    p2p::VoxelHashMap local_map_;

private:
    //ICP pipeline modules
    Config config_;
    p2p::AdaptiveThreshold adaptive_threshold_;

    double GetAdaptiveThreshold();

    Sophus::SE3 GetPredictionModel() const;

    bool HasMoved();
    bool has_moved = false;
};

#endif