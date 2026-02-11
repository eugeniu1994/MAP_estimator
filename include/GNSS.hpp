#ifndef USE_GNSS_H1
#define USE_GNSS_H1

#include <utils.h>

#include <GeographicLib/LocalCartesian.hpp>


class GNSS
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    bool shift_measurements_to_zero_time = false;
    bool init_shift_measurements_to_zero_time = false;
    double first_time = 0;

    double tod, diff_curr_gnss2mls;
    double gps_time, global_gps_time, theta_GPS_to_IMU = 0., max_travelled_distance_for_initialization = 20.;
    std::vector<V3D> gps_measurements, gps_covariances, enu_measurements;
    V3D ref_gps_point_lla, origin_enu, curr_enu, gps_cov;

    GeographicLib::LocalCartesian geo_converter;

    Sophus::SE3 gps_pose, postprocessed_gps_pose, first_gps_pose, als2mls_T;
    M3D R_GNSS_to_MLS, R_compas; 
    V3D carthesian;
    bool GNSS_extrinsic_init = false, use_postprocessed_gnss = false, gps_init_origin, first_gpsFix_received_ = false;
    int curr_gnss = 0, total_gnss;
    M3D gt_first_rot, gt_rotation;
    V3D gt_first_translation;
    std::vector<V3D> all_gnss, all_mls;
    std::vector<double> ind, all_theta;

    GNSS()
    {
        gps_time = 0.;
        global_gps_time = 0.;
        gps_init_origin = false;
        carthesian = Zero3d;
        origin_enu = Zero3d;
        curr_enu = Zero3d;
        R_GNSS_to_MLS = Eye3d;
        R_compas = Eye3d;
        first_gps_pose = Sophus::SE3();
        gps_cov = V3D(1,1,1);
        gps_pose = Sophus::SE3();
        postprocessed_gps_pose = Sophus::SE3();
        als2mls_T = Sophus::SE3();
    };
    
    ~GNSS() {};
    
    const double gps_epoch = 315964800; // Unix timestamp for 1980-01-06 00:00:00 // GPS time started on January 6, 1980
    
    void Process(std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer, const double &lidar_end_time, const V3D &MLS_pos);
    
    void calibrateGnssExtrinsic(const V3D &MLS_pos);

    void set_param(const double &GNSS_IMU_calibration_distance);

    void updateExtrinsic(const M3D &R_);
};


#endif