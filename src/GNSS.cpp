#include "GNSS.hpp"
#include <GeographicLib/UTMUPS.hpp>

// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;

using namespace gnss;

void GNSS::Process(std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer,
                   const double &lidar_end_time, const V3D &MLS_pos)
{
    while ((!gps_buffer.empty()) && gps_time <= lidar_end_time)
    {
        auto msg = gps_buffer.front();
        auto msg_time = msg->header.stamp.toSec();

        if (!init_shift_measurements_to_zero_time)
        {
            init_shift_measurements_to_zero_time = true;
            first_time = msg->header.stamp.toSec();
        }
        if (shift_measurements_to_zero_time)
        {
            msg_time = msg_time - first_time;
        }

        gps_time = msg_time; // msg->header.stamp.toSec(); //THIS IS FROM ROS - align the gps and lidar based on ROS TIMES
        gps_buffer.pop_front();

        global_gps_time = msg->time + gps_epoch;

        if (gps_init_origin && !gps_buffer.empty()) // position initialised and still msgs exists
        {
            diff_curr_gnss2mls = fabs(gps_time - lidar_end_time);
            auto next_msg_time = gps_buffer.front()->header.stamp.toSec();
            if (shift_measurements_to_zero_time)
            {
                next_msg_time = next_msg_time - first_time;
            }
            double diff_next = fabs(next_msg_time - lidar_end_time);
            if (diff_curr_gnss2mls > diff_next)
            {
                continue; // continue to go to the next message
            }

            diff_curr_gnss2mls = gps_time - lidar_end_time;
        }

        if (true) // compute time of the day
        {
            auto gpsTime = msg->time;   // this is the actual GPS time  1980-01-06
            const int leapSeconds = 18; // Number of leap seconds as of October 2023
            double utcTime = gpsTime - leapSeconds;
            const int secondsInADay = 24 * 60 * 60;

            // Compute the time of the day in seconds (including fractional part)
            tod = fmod(utcTime, secondsInADay);
            // std::cout << "GNSS tod:" << tod << ", diff_curr_gnss2mls:" << diff_curr_gnss2mls << std::endl;
            tod = tod - diff_curr_gnss2mls;
            // std::cout << "Corrected tod of the mls:" << tod << std::endl;
        }

        Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->altitude);
        double easting, northing, height = msg->altitude;
        int zone;
        bool northp;

        double compas = msg->track;
        compas = compas * M_PI / 180.;
        R_compas << cos(compas), -sin(compas), 0,
            sin(compas), cos(compas), 0,
            0, 0, 1; // Yaw rotation around the z-axis

        float noise_x = msg->position_covariance[0];
        float noise_y = msg->position_covariance[4];
        float noise_z = msg->position_covariance[8];
        gps_cov = V3D(noise_x, noise_y, noise_z);

        // GET THE WEIGHTED AVERAGE lla of the current position
        if (!gps_init_origin)
        {
            if (gps_measurements.size() < 10) //
            {
                gps_measurements.push_back(lla);
                gps_covariances.push_back(gps_cov);

                GeographicLib::UTMUPS::Forward(lla[0], lla[1], zone, northp, easting, northing);
                V3D origin_enu_i = V3D(easting, northing, height);
                enu_measurements.push_back(origin_enu_i);

                return;
            }
            else
            {
                // current averaged lla
                ref_gps_point_lla = computeWeightedAverage(gps_measurements, gps_covariances);
                std::cout << "ref_gps_point_lla:" << ref_gps_point_lla.transpose() << std::endl;
                gps_init_origin = true;
                gps_measurements.clear();
                gps_covariances.clear();
                enu_measurements.clear();

                GeographicLib::UTMUPS::Forward(ref_gps_point_lla[0], ref_gps_point_lla[1], zone, northp, easting, northing);
                origin_enu = V3D(easting, northing, height);

                // origin_enu = computeWeightedAverage(enu_measurements, gps_covariances);

                first_gps_pose = Sophus::SE3(Eye3d, origin_enu).inverse();
                std::cout << "first_gps_pose:\n"
                          << first_gps_pose.matrix() << std::endl;

                geo_converter.Reset(ref_gps_point_lla[0], ref_gps_point_lla[1], ref_gps_point_lla[2]); // set origin the current point
                return;
            }
        }

        // Convert Latitude and Longitude to UTM
        GeographicLib::UTMUPS::Forward(lla[0], lla[1], zone, northp, easting, northing);
        curr_enu = V3D(easting, northing, height);

        double x, y, z; // LLA->carthesian
        geo_converter.Forward(lla[0], lla[1], lla[2], x, y, z);
        carthesian = V3D(x, y, z); // this will be relative w.r.t. zero origin
        if (abs(carthesian.x()) > 10000 || abs(carthesian.x()) > 10000 || abs(carthesian.x()) > 10000)
        {
            ROS_INFO("Error origin : %f, %f, %f", carthesian(0), carthesian(1), carthesian(2));
            return;
        }

        calibrateGnssExtrinsic(MLS_pos);

        if (GNSS_extrinsic_init)
        {
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, (R_GNSS_to_MLS * curr_enu));
        }
        else
        {
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, curr_enu);
        }
        gps_pose.so3() = Sophus::SO3(R_compas);
    }
}

void GNSS::calibrateGnssExtrinsic(const V3D &MLS_pos)
{
    if (!GNSS_extrinsic_init)
    {
        if (MLS_pos.norm() > .5) // start moving
        {
            all_gnss.push_back(carthesian);
            all_mls.push_back(MLS_pos);

            // std::cout << "GPS dist:" << carthesian.norm() << ", MLS dist:" << MLS_pos.norm() << std::endl;
            //double theta = gnss::findAngle(Zero3d, carthesian, Zero3d, MLS_pos); // in degrees
            // ind.push_back(all_theta.size());
            // all_theta.push_back(theta);

            double xy_range = std::sqrt(MLS_pos.x() * MLS_pos.x() + MLS_pos.y() * MLS_pos.y());

            if (xy_range > max_travelled_distance_for_initialization)
            {
                M3D R_svd;
                V3D t_svd;

                computeTransformation(all_gnss, all_mls, R_svd, t_svd);
                std::cout << "Translation vector t:\n"
                          << t_svd.transpose() << std::endl;
                double yaw_svd = std::atan2(R_svd(1, 0), R_svd(0, 0)) * (180.0 / M_PI);
                std::cout << "yaw_svd angle (degrees): " << yaw_svd << std::endl;

                theta_GPS_to_IMU = yaw_svd * M_PI / 180.0;

                R_GNSS_to_MLS << cos(theta_GPS_to_IMU), -sin(theta_GPS_to_IMU), 0,
                    sin(theta_GPS_to_IMU), cos(theta_GPS_to_IMU), 0,
                    0, 0, 1; // Yaw rotation around the z-axis

                // an approximation where is the GNSS antena
                first_gps_pose = Sophus::SE3(Eye3d, (R_GNSS_to_MLS * origin_enu)).inverse();

                GNSS_extrinsic_init = true;
            }
        }
    }
}

void GNSS::updateExtrinsic(const M3D &R_)
{
    R_GNSS_to_MLS = R_;
    first_gps_pose = Sophus::SE3(Eye3d, (R_GNSS_to_MLS * origin_enu)).inverse();
}

void GNSS::set_param(const double &GNSS_IMU_calibration_distance)
{
    max_travelled_distance_for_initialization = GNSS_IMU_calibration_distance;
}
