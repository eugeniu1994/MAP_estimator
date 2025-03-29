#ifndef USE_GNSS_H1
#define USE_GNSS_H1

#include <utils.h>

#include <GeographicLib/LocalCartesian.hpp>

struct GNSS_IMU_Measurement
{
    double UTCTime; //(sec)
    double GPSTime; //(sec)

    double Easting;  //(m)
    double Northing; //(m)
    double H_Ell;    //(m)

    double Heading; //(deg)
    double Pitch;   //(deg)
    double Roll;    //(deg)

    /*double AccBdyX; //((m/s^2))
    double AccBdyY; //((m/s^2))
    double AccBdyZ; //((m/s^2))

    double AccBiasX; //((m/s^2))
    double AccBiasY; //((m/s^2))
    double AccBiasZ; //((m/s^2))

    double GyroDriftX; //(deg/s)
    double GyroDriftY; //(deg/s)
    double GyroDriftZ; //(deg/s)

    double Cx11; //(m^2)
    double Cx22; //(m^2)
    double Cx33; //(m^2)

    double CxVHH; //(deg^2)
    double CxVPP; //(deg^2)
    double CxVRR; //(deg^2)*/

    double original_GPSTime;
};


class GNSS
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double tod;
    double gps_time, global_gps_time, theta_GPS_to_IMU = 0., max_travelled_distance_for_initialization = 20.;
    std::vector<V3D> gps_measurements, gps_covariances;
    V3D ref_gps_point_lla, origin_enu, curr_enu, gps_cov;

    GeographicLib::LocalCartesian geo_converter;

    Sophus::SE3 gps_pose, postprocessed_gps_pose, first_gps_pose;
    M3D R_GNSS_to_MLS, R_compas; 
    V3D carthesian,GNSS_T_wrt_IMU;
    bool GNSS_extrinsic_init = false, use_postprocessed_gnss = false, gps_init_origin, first_gpsFix_received_ = false;
    int curr_gnss = 0, total_gnss;
    M3D gt_first_rot, gt_rotation;
    V3D gt_first_translation;
    std::vector<GNSS_IMU_Measurement> gnss_measurements;
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
        GNSS_T_wrt_IMU = Zero3d;
        R_GNSS_to_MLS = Eye3d;
        R_compas = Eye3d;
        first_gps_pose = Sophus::SE3();
        gps_cov = V3D(1,1,1);
        gps_pose = Sophus::SE3();
        postprocessed_gps_pose = Sophus::SE3();
    };
    
    ~GNSS() {};
    
    const double gps_epoch = 315964800; // Unix timestamp for 1980-01-06 00:00:00 // GPS time started on January 6, 1980
    
    void Process(std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer, const double &lidar_end_time, const V3D &MLS_pos);
    
    void Process(const Sophus::SE3 &gps_local, const Sophus::SE3 &gps_global, const double &lidar_end_time, const V3D &MLS_pos);

    void calibrateGnssExtrinsic(const V3D &MLS_pos);

    void set_param(const V3D &tran, const double &GNSS_IMU_calibration_distance, std::string _postprocessed_gnss_path = "");

    void updateExtrinsic(const M3D &R_);

    std::vector<GNSS_IMU_Measurement> parseGNSSFile(const std::string &filename);
};


#endif