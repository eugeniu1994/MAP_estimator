#ifndef USE_POINTDEF_H1
#define USE_POINTDEF_H1

#pragma once
#define PCL_NO_PRECOMPILE

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Eigen>
#include <Eigen/Core>

struct EIGEN_ALIGN16 PointType // point used for registration
{
  PCL_ADD_POINT4D; // This macro adds x, y, z fields
  float intensity;
  float time;                     // time in s relative to start of the scan
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // For Eigen compatibility
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointType,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time))

typedef pcl::PointCloud<PointType> PointCloudXYZI;

namespace hesai_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace hesai_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))

// the one provided by Petri

// namespace velodyne_ros
// {
//   struct EIGEN_ALIGN16 Point
//   {
//     PCL_ADD_POINT4D;
//     float intensity;
//     std::uint8_t ring;
//     std::uint32_t time; // ns from the beginning of the scan
//     float distance;     // distance, return type coded in sign, negative -> 'last return'; positive -> 'strongest'
//     EIGEN_MAKE_ALIGNED_OPERATOR_NEW // ensure proper alignment
//   };
// };

// POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
//   (float, x, x)
//   (float, y, y)
//   (float, z, z)
//   (float, intensity, intensity)
//   (std::uint8_t, ring, ring)
//   (std::uint32_t, time, time)
//   (float, distance, distance))

namespace velodyne_ros
{
  struct EIGEN_ALIGN16 Point
  {
    PCL_ADD_POINT4D; // Always required for PCL point types
    float intensity;
    std::uint16_t ring;
    double time; // Datatype 8 corresponds to double
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}

// Register the custom point type with PCL
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(double, time, time))

struct VUX_PointType
{
  PCL_ADD_POINT4D;
  
  float range;
  // float echo_range; //! echo range in units of meter
  double time; ////! time stamp in [s]
  // double time_sorg; // The timestamp of the start of the rangegate (internal time).
  // float amplitude;  //! relative amplitude in [dB]
  float reflectance;
  // float deviation;
  // unsigned segment; // segment number
  //  bool is_line_end;
  // int single_echo;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

} EIGEN_ALIGN16; // Align the structure to 16-byte boundary for SSE optimizations

// Register your custom point type with PCL's point cloud library
POINT_CLOUD_REGISTER_POINT_STRUCT(VUX_PointType,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (float, range, range)
                                  //(float, echo_range, echo_range)
                                  (double, time, time)
                                  //(double, time_sorg, time_sorg)
                                  //(float, amplitude, amplitude)
                                  (float, reflectance, reflectance)
                                  //(float, deviation, deviation)
                                  //(unsigned, segment, segment)
                                  //(bool, is_line_end, is_line_end)
                                  //(int, single_echo, single_echo)
                                  ) // Register custom fields

#endif