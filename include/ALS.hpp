#ifndef USE_ALS_H1
#define USE_ALS_H1

#include <boost/filesystem.hpp>
#include <iostream>

#include <utils.h>

#ifdef USE_EKF
    #if USE_STATIC_KDTREE == 0
        #include <ikd-Tree/ikd_Tree.h>
    #endif
#else
    #include "p2p/core/VoxelHashMap.hpp"
#endif

struct PlanePrimitive {
    V3D centroid;
    V3D normal;
    float curvature;
};

class ALS_Handler
{
public:
    ALS_Handler(const std::string &folder_root_, bool donwsample_, int closest_N_files_, double leaf_size_);
    ~ALS_Handler() {};

    Sophus::SE3 als_to_mls;
    M3D R_to_mls;
#ifdef USE_EKF
    PointCloudXYZI::Ptr als_cloud;
    pcl::CropBox<PointType> cropBoxFilter;
    #if USE_STATIC_KDTREE == 0
        KD_TREE<PointType> ikdtree;
        std::vector<BoxPointType> cub_needrm;
    #endif
#else
    Config config_;
    p2p::VoxelHashMap local_map_;
    //int points_per_leaf_size = 15; //make this a parameter
#endif

    bool init(const V3D &gps_origin_ENU_, const M3D &init_R_2_mls, const PointCloudXYZI::Ptr &mls_cloud_full);
    bool init(const Sophus::SE3 &known_als2mls);
    bool init(const Sophus::SE3 &known_als2mls, const PointCloudXYZI::Ptr &mls_cloud_full);
    
    void getCloud(PointCloudXYZI::Ptr &in_);
    bool Update(const Sophus::SE3 &mls_pose);

    void computePlanes(double leaf_size, double curvature_threshold, int min_points_per_voxel = 5);
    std::vector<PlanePrimitive> stable_planes;


    bool refine_als = false, initted_ = false;
    int min_points_per_patch = 0;
    

private:

    V3D gps_origin_ENU;
    int closest_N_files = 4;
    double boxSize = 50., leaf_size = 1.0;
    bool downsampled_ALS = false, found_ALS = false;
    bool shift_initted_ = false;
    // std::string folder_root = "/media/eugeniu/T7/EVO_las/";
    std::string folder_root = "/media/eugeniu/T7/NLS_las/"; // this is the ALS from NlS

    V3D prev_mls_pos, curr_mls_pos;
    pcl::KdTreeFLANN<pcl::PointXYZ> ALS_manager;
    bool als_manager_setup = false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr all_las_files;
    int key = 0;

    pcl::VoxelGrid<PointType> downSizeFilterSurf;


    void setupALS_Manager();
    void AddPoints_from_file(const std::string &filename);
    void RemovePointsFarFromLocation(const V3D &mls_position);
};

#endif