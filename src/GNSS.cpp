#include "GNSS.hpp"
#include <GeographicLib/UTMUPS.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace gnss;

void GNSS::Process(const Sophus::SE3 &gps_local, const Sophus::SE3 &gps_global, const double &lidar_end_time, const V3D &MLS_pos)
{
    {
        double easting, northing, height = gps_global.translation()[2];
        int zone;
        bool northp;

        float noise_x = 1;
        float noise_y = 1;
        float noise_z = 1;
        gps_cov = V3D(noise_x, noise_y, noise_z);

        // GET THE WEIGHTED AVERAGE lla of the current position
        if (!gps_init_origin)
        {
            origin_enu = gps_global.translation();
            first_gps_pose = Sophus::SE3(Eye3d, origin_enu).inverse();
            //first_gps_pose = gps_global.inverse();
            gps_init_origin = true;
        }
        
        curr_enu = gps_global.translation();        
        carthesian = gps_local.translation(); // this will be relative w.r.t. zero origin
        
        gps_pose = first_gps_pose * Sophus::SE3(Eye3d, curr_enu);
        carthesian = gps_pose.translation();

        
        calibrateGnssExtrinsic(MLS_pos);

        if (GNSS_extrinsic_init)
        {
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, (R_GNSS_to_MLS * curr_enu) + GNSS_T_wrt_IMU);
        }
        else
        {
            // we cannot use the translation offset yet, sice the rotation is unknown
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, curr_enu);
        }
        
        //gps_pose = gps_local;
    }
}

void GNSS::Process(std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer,
                   const double &lidar_end_time, const V3D &MLS_pos)
{

    while ((!gps_buffer.empty()) && gps_time <= lidar_end_time)
    {
        auto msg = gps_buffer.front();
        gps_time = msg->header.stamp.toSec();
        gps_buffer.pop_front();

        global_gps_time = msg->time + gps_epoch;

        

        if (gps_init_origin && !gps_buffer.empty()) // position initialised and still msgs exists
        {
            double diff_curr = fabs(gps_time - lidar_end_time);
            double diff_next = fabs(gps_buffer.front()->header.stamp.toSec() - lidar_end_time);
            // std::cout<<"diff_curr:"<<diff_curr<<", diff_next:"<<diff_next<<std::endl;
            if (diff_curr > diff_next)
            {
                continue; // continue to go to the next message
            }
        }

        Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->altitude);
        double easting, northing, height = msg->altitude;
        int zone;
        bool northp;

        double compas = msg->track;
        // compas += 90.; //shift to get the degrees from EAST/WEST
        // std::cout << "compas:" << compas << " degrees, rad:" << compas * M_PI / 180. << std::endl;
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
            if (gps_measurements.size() < 10)
            {
                gps_measurements.push_back(lla);
                gps_covariances.push_back(gps_cov);
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

                GeographicLib::UTMUPS::Forward(ref_gps_point_lla[0], ref_gps_point_lla[1], zone, northp, easting, northing);
                origin_enu = V3D(easting, northing, height);
                first_gps_pose = Sophus::SE3(Eye3d, origin_enu).inverse();

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
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, (R_GNSS_to_MLS * curr_enu) + GNSS_T_wrt_IMU);
        }
        else
        {
            // we cannot use the translation offset yet, sice the rotation is unknown
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, curr_enu);
        }
        gps_pose.so3() = Sophus::SO3(R_compas);

        std::cout<<"HESAI GNSS TIME:"<<msg->time<<std::endl;

        auto gpsTime = msg->time;
        
        const int leapSeconds = 18; // Number of leap seconds as of October 2023
        double utcTime = gpsTime - leapSeconds;
        long totalSeconds = static_cast<long>(utcTime);
        const int secondsInADay = 24 * 60 * 60;
        int timeOfDaySeconds = totalSeconds % secondsInADay;

        int hours = timeOfDaySeconds / 3600;
        int remainingSeconds = timeOfDaySeconds % 3600;
        int minutes = remainingSeconds / 60;
        int seconds = remainingSeconds % 60;

        // Format the time as HH:MM:SS
        char buffer[9];
        snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, minutes, seconds);

        std::string timeOfDay = std::string(buffer);
        std::cout << "GNSS Time of the day: " << timeOfDay << std::endl;

        // Compute the time of the day in seconds (including fractional part)
        tod = fmod(utcTime, secondsInADay);
        std::cout<<"GNSS tod:"<<tod<<std::endl;
    }

    if (use_postprocessed_gnss)
    {
        while (curr_gnss < total_gnss && gnss_measurements[curr_gnss].GPSTime <= global_gps_time)
        {
            const auto &pose = gnss_measurements[curr_gnss];
            curr_gnss++;

            if (fabs(gnss_measurements[curr_gnss].GPSTime - global_gps_time) > fabs(gnss_measurements[curr_gnss + 1].GPSTime - global_gps_time))
                continue;

            // curr_gnss++;
            double Heading = -pose.Heading * M_PI / 180.;
            double Pitch = pose.Pitch * M_PI / 180.;
            double Roll = pose.Roll * M_PI / 180.;

            if (!first_gpsFix_received_)
            {
                gt_first_rot = Eigen::AngleAxisd(Heading, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(Pitch, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(Roll, Eigen::Vector3d::UnitX());
                gt_first_rot = gt_first_rot.transpose();
                gt_first_translation = V3D(pose.Easting, pose.Northing, pose.H_Ell);
                first_gpsFix_received_ = true;
            }

            gt_rotation = Eigen::AngleAxisd(Heading, Eigen::Vector3d::UnitZ()) *
                          Eigen::AngleAxisd(Pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(Roll, Eigen::Vector3d::UnitX());
            V3D gt_translation(pose.Easting, pose.Northing, pose.H_Ell);

            postprocessed_gps_pose = Sophus::SE3(gt_first_rot * gt_rotation, R_GNSS_to_MLS * (gt_translation - gt_first_translation));
        }
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

            std::cout << "GPS dist:" << carthesian.norm() << ", MLS dist:" << MLS_pos.norm() << std::endl;
            double theta = gnss::findAngle(Zero3d, carthesian, Zero3d, MLS_pos); // in degrees

            ind.push_back(all_theta.size());
            all_theta.push_back(theta);

            if (MLS_pos.norm() > max_travelled_distance_for_initialization)
            {
                std::cout<<"all_theta:"<<all_theta.size()<<std::endl;
                gnss::LineModel ransac = gnss::ransacFitLine(ind, all_theta, 200., .5);
                std::cout << "ransac m:" << ransac.m << ", b:" << ransac.b << std::endl;
                theta_GPS_to_IMU = ransac.b;

                // verify the sign
                theta_GPS_to_IMU = gnss::verifyAngle(theta_GPS_to_IMU, Zero3d, carthesian, Zero3d, MLS_pos); // in degrees
                std::cout << "theta_GPS_to_IMU:" << theta_GPS_to_IMU << std::endl;

                std::cout << "all_theta:" << all_theta.size() << ", ind:" << ind.size() << std::endl;

                plt::scatter(ind, all_theta, 5, {{"label", "Angles"}, {"color", "blue"}});
                std::vector<double> line_x = {ind.front(), ind.back()};
                std::vector<double> line_y = {
                    (ransac.m * M_PI / 180.) * line_x.front() + ransac.b,
                    (ransac.m * M_PI / 180.) * line_x.back() + ransac.b};
                plt::plot(line_x, line_y, {{"label", "RANSAC Fit Line"}, {"color", "red"}});

                plt::xlabel("Measurements");
                plt::ylabel("Angles (deg)");
                plt::title("GPS_IMU Initialization");
                plt::grid(true);
                plt::legend();

                // test SVD
                M3D R_svd;
                V3D t_svd;

                computeTransformation(all_gnss, all_mls, R_svd, t_svd);
                std::cout << "Translation vector t:\n"
                          << t_svd.transpose() << std::endl;
                double yaw_svd = std::atan2(R_svd(1, 0), R_svd(0, 0)) * (180.0 / M_PI);
                std::cout << "yaw_svd angle (degrees): " << yaw_svd << std::endl;
                theta_GPS_to_IMU = yaw_svd; //

                theta_GPS_to_IMU = theta_GPS_to_IMU * M_PI / 180.0;
                R_GNSS_to_MLS << cos(theta_GPS_to_IMU), -sin(theta_GPS_to_IMU), 0,
                    sin(theta_GPS_to_IMU), cos(theta_GPS_to_IMU), 0,
                    0, 0, 1; // Yaw rotation around the z-axis

                // this will use the one from SVD
                // R_GNSS_to_MLS = R_svd;
                first_gps_pose = Sophus::SE3(Eye3d, (R_GNSS_to_MLS * origin_enu) + GNSS_T_wrt_IMU).inverse();

                plt::show();
                GNSS_extrinsic_init = true;
            }
        }
    }
}

void GNSS::updateExtrinsic(const M3D &R_)
{
    R_GNSS_to_MLS = R_;
    first_gps_pose = Sophus::SE3(Eye3d, (R_GNSS_to_MLS * origin_enu) + GNSS_T_wrt_IMU).inverse();
    //GNSS_extrinsic_init = true;
}

void GNSS::set_param(const V3D &tran, const double &GNSS_IMU_calibration_distance, std::string _postprocessed_gnss_path)
{
    GNSS_T_wrt_IMU = tran; // offset translation to go from GNSS antena frame to IMU frame
    max_travelled_distance_for_initialization = GNSS_IMU_calibration_distance;
    if (!_postprocessed_gnss_path.empty())
    {
        if (boost::filesystem::exists(_postprocessed_gnss_path))
        {
            std::cout << "Use postprocessed GNSS solution" << std::endl;
            use_postprocessed_gnss = true;
            gnss_measurements = parseGNSSFile(_postprocessed_gnss_path);
        }
    }
}

std::vector<GNSS_IMU_Measurement> GNSS::parseGNSSFile(const std::string &filename)
{
    std::vector<GNSS_IMU_Measurement> measurements;
    std::ifstream file(filename);
    std::string line;
    bool dataSection = false;

    const double GPS_TO_UTC_OFFSET = 18.0; // Difference between GPS time and UTC time in seconds
    if (file.is_open())
    {
        double dummy; // To skip unwanted fields
        while (std::getline(file, line))
        {
            // Skip lines until we reach the header of the data section
            if (!dataSection)
            {
                if (line.find("UTCTime") != std::string::npos)
                {
                    dataSection = true;
                    std::getline(file, line); // Skip the units line
                }
                continue;
            }

            std::istringstream iss(line);
            GNSS_IMU_Measurement measurement;

            iss >> measurement.UTCTime >> measurement.GPSTime >> measurement.Easting >> measurement.Northing >> measurement.H_Ell >> measurement.Heading >> measurement.Pitch >> measurement.Roll; // >> measurement.AccBdyX >> measurement.AccBdyY >> measurement.AccBdyZ >> measurement.AccBiasX >> measurement.AccBiasY >>
                                                                                                                                                                                                   // measurement.AccBiasZ >> measurement.GyroDriftX >> measurement.GyroDriftY >> measurement.GyroDriftZ >> measurement.Cx11 >> measurement.Cx22 >> measurement.Cx33
                                                                                                                                                                                                   //>> measurement.CxVHH >> measurement.CxVPP >> measurement.CxVRR;
            measurement.original_GPSTime = measurement.GPSTime;
            measurement.GPSTime = measurement.UTCTime + GPS_TO_UTC_OFFSET;

            measurements.push_back(measurement);
        }
        file.close();
        total_gnss = measurements.size() - 1;
        std::cout << "parseGNSSFile measurements:" << total_gnss << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file " << filename << std::endl;
        use_postprocessed_gnss = false;
    }

    return measurements;
}