
#include "Preprocessing.hpp"
#include <tsl/robin_map.h>
#include "icp_util.hpp"

namespace
{
    struct VoxelHash
    {
        size_t operator()(const Voxel &voxel) const
        {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };
} // namespace

namespace p2p
{
    std::vector<V3D> VoxelDownsample(const std::vector<V3D> &frame, double voxel_size)
    {
        tsl::robin_map<Voxel, V3D, VoxelHash> grid;
        grid.reserve(frame.size());
        double inv_voxel_size = 1.0 / voxel_size;
        std::for_each(frame.cbegin(), frame.cend(), [&](const auto &point)
                      {
                    const auto &voxel = PointToVoxel(point, inv_voxel_size);
                    if (!grid.contains(voxel)) grid.insert({voxel, point}); });

        std::vector<V3D> frame_dowsampled(grid.size());
        //std::vector<V3D> frame_dowsampled; frame_dowsampled.reserve(grid.size());
        std::transform(grid.begin(), grid.end(), frame_dowsampled.begin(),
               [](const auto& entry) { return entry.second; });

        return frame_dowsampled;
    }

    std::vector<V3D> VoxelDownsample(const PointCloudXYZI::Ptr &frame, double voxel_size)
    {
        tsl::robin_map<Voxel, V3D, VoxelHash> grid;
        grid.reserve(frame->size());
        double inv_voxel_size = 1.0 / voxel_size;
        for (const auto &pointXYZI : frame->points)
        {
            V3D point(pointXYZI.x, pointXYZI.y, pointXYZI.z);
            const auto &voxel = PointToVoxel(point, inv_voxel_size);
            if (!grid.contains(voxel)) grid.insert({voxel, point});
        }
        
        std::vector<V3D> frame_dowsampled(grid.size());
        //std::vector<V3D> frame_dowsampled; frame_dowsampled.reserve(grid.size());
        //for (const auto &[voxel, point] : grid){
        //    (void)voxel;
        //    frame_dowsampled.emplace_back(point);}

        std::transform(grid.begin(), grid.end(), frame_dowsampled.begin(),
               [](const auto& entry) { return entry.second; });

        return frame_dowsampled;
    }
} // namespace p2p