#include "VoxelHash.hpp"

Voxel PointToVoxel(const V3D &point, const double &voxel_size)
{
    return Voxel(static_cast<int>(std::floor(point.x() / voxel_size)),
                 static_cast<int>(std::floor(point.y() / voxel_size)),
                 static_cast<int>(std::floor(point.z() / voxel_size)));
}

void VoxelHashMap::Build(const pcl::PointCloud<PointType>::Ptr &points)
{
    std::cout << "Build to hash map" << std::endl;
    int level = 1;

    int max_levels = 3;
    double good_plane = .1; // 10cm for p2plane

    //good_plane = .01;   //for curvature 

    for (int i = 0; i < points->size(); i++)
    {
        V3D p(points->points[i].x, points->points[i].y, points->points[i].z);
        const auto &voxel = PointToVoxel(p, voxel_size_);

        auto search = map_.find(voxel);
        if (search != map_.end())
        {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(p);
        }
        else
        {
            VoxelBlock v;
            v.AddPoint(p);
            v.level = level;
            map_.insert({voxel, v});
        }
    }

    std::cout << "Map size: " << map_.size() << std::endl;
    std::vector<Voxel> voxels_require_split;
    for (const auto &entry : map_) // compute normal for each voxel
    {
        auto search = map_.find(entry.first);
        if (search != map_.end())
        {
            auto &voxel_block = search.value();
            if (voxel_block.has_normal) // this block has good normal already
                continue;

            if (!voxel_block.ComputePlaneNormal(good_plane)) // does not have a good normal
                voxels_require_split.push_back(entry.first); // save they keys for later split
        }
    }
    if (voxels_require_split.size() == 0) // all voxel have good normals
        return;
   
    int removed_voxels = 0;
    for (level = 2; level <= max_levels; level++)
    {
        std::cout << "\nSplit level:" << level << std::endl;
        std::vector<Voxel> new_added_voxels; // that do not meet planarity

        if (voxels_require_split.size() == 0) // stop if no longer split is required
            break;

        good_plane = good_plane / 2.0; //for smaller voxel - smaller threshold
        voxel_size_ = voxel_size_ / 2.0; // half voxel size
        for (const auto &key : voxels_require_split)
        {
            auto it = map_.find(key);
            if (it == map_.end())
                continue;

            auto &voxel_block = it.value();
            std::vector<V3D> points = voxel_block.points; // a copy of the points
            map_.erase(key);                               // Safely erase old voxel

            if (points.size() < plane_min_points) // not enough points - not even added to the map
                continue;

            for (const auto &point : points)
            {
                const auto &child_voxel = PointToVoxel(point, voxel_size_);
                auto search = map_.find(child_voxel);
                if (search != map_.end()) // Voxel exists
                {
                    auto &voxel_block = search.value();
                    voxel_block.AddPoint(point);
                }
                else // Add new voxel
                {
                    VoxelBlock v;
                    v.AddPoint(point);
                    v.level = level;
                    map_.insert({child_voxel, v});
                    new_added_voxels.push_back(child_voxel);
                }
            }
        }

        std::cout << "new_added_voxels:" << new_added_voxels.size() << std::endl;
        voxels_require_split.clear();

        for (const auto &key : new_added_voxels) // iterate the newlly added voxels
        {
            auto it = map_.find(key);
            if (it == map_.end())
                continue;

            auto &voxel_block = it.value();
            if (voxel_block.has_normal)
                continue;

            if (!voxel_block.ComputePlaneNormal(good_plane)) // does not have a good normal
            {
                if (level == max_levels) // if last level
                {
                    map_.erase(key);      // smaller voxel does not have good normal
                    removed_voxels++;
                }
                else
                {
                    voxels_require_split.push_back(key); // save they keys for later split
                }
            }
        }
    }
    std::cout << "Removed " << removed_voxels << " bad voxels\n"
              << std::endl;
}

int VoxelHashMap::Pointcloud(pcl::PointCloud<PointType>::Ptr &cloud_) const
{
    cloud_->clear();
    cloud_->points.reserve(max_points_per_voxel_ * map_.size());
    int voxel_id = 1;
    for (const auto &[voxel, voxel_block] : map_)
    {
        (void)voxel;
        for (const auto &point : voxel_block.points)
        {
            PointType p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            p.intensity = voxel_id;
            // p.time = voxel_block.level;

            p.time = voxel_block.curvature;
            p.time = voxel_block.points.size();

            cloud_->points.emplace_back(p);
        }
        voxel_id++;
    }
    cloud_->width = cloud_->points.size();
    cloud_->height = 1;
    cloud_->is_dense = false;

    return cloud_->size();
}

int VoxelHashMap::Pointcloud_and_Normals(pcl::PointCloud<PointType>::Ptr &cloud_,
                                         pcl::PointCloud<pcl::Normal>::Ptr &normals) const
{
    cloud_->clear();
    cloud_->points.reserve(map_.size());
    normals->clear();
    normals->points.reserve(map_.size());
    int voxel_id = 1;
    for (const auto &[voxel, voxel_block] : map_)
    {
        (void)voxel;
        if (voxel_block.has_normal)
        {
            const auto &normal = voxel_block.normal;

            pcl::Normal pcl_normal;
            pcl_normal.normal_x = normal.x();
            pcl_normal.normal_y = normal.y();
            pcl_normal.normal_z = normal.z();

            const auto &point = voxel_block.points[0];
            PointType p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            p.intensity = voxel_id;
            p.time = voxel_block.level;

            cloud_->points.emplace_back(p);
            normals->points.emplace_back(pcl_normal);
        }

        voxel_id++;
    }
    cloud_->width = cloud_->points.size();
    cloud_->height = 1;
    cloud_->is_dense = false;

    normals->width = cloud_->points.size();
    normals->height = 1;
    normals->is_dense = false;

    return cloud_->size();
}
