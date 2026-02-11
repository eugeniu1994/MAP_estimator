#ifndef USE_POINTDEF_H1
#define USE_POINTDEF_H1

#pragma once
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

struct EIGEN_ALIGN16 PointType // point used for registration
{
  PCL_ADD_POINT4D; // This macro adds x, y, z fields
  float intensity;
  float time;                     // time in s relative to start of the scan
  // uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // For Eigen compatibility
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointType,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)
  // (uint16_t, ring, ring)
)

typedef pcl::PointCloud<PointType> PointCloudXYZI;

namespace hesai_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    uint16_t ring;
    // float range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace hesai_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
  (float, x, x)(float, y, y)(float, z, z)
  (float, intensity, intensity)
  (double, timestamp, timestamp)
  (uint16_t, ring, ring)
  // (float, range, range)
)

#endif