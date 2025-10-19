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
            // first_gps_pose = gps_global.inverse();
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

        // gps_pose = gps_local;
    }
}

void GNSS::Process(std::deque<gps_common::GPSFix::ConstPtr> &gps_buffer,
                   const double &lidar_end_time, const V3D &MLS_pos)
{
    while ((!gps_buffer.empty()) && gps_time <= lidar_end_time)
    {
        auto msg = gps_buffer.front();
        auto msg_time = msg->header.stamp.toSec();

        if(!init_shift_measurements_to_zero_time)
        {
            init_shift_measurements_to_zero_time = true;
            first_time = msg->header.stamp.toSec();
        }
        if(shift_measurements_to_zero_time)
        {
            msg_time = msg_time - first_time;
        }

        gps_time = msg_time;//msg->header.stamp.toSec(); //THIS IS FROM ROS - align the gps and lidar based on ROS TIMES
        gps_buffer.pop_front();

        global_gps_time = msg->time + gps_epoch;

        if (gps_init_origin && !gps_buffer.empty()) // position initialised and still msgs exists
        {
            diff_curr_gnss2mls = fabs(gps_time - lidar_end_time);
            auto next_msg_time = gps_buffer.front()->header.stamp.toSec();
            if(shift_measurements_to_zero_time)
            {
                next_msg_time = next_msg_time - first_time;
            }
            double diff_next = fabs(next_msg_time - lidar_end_time);
            std::cout<<"gps_time:"<<gps_time<<", lidar_end_time:"<<lidar_end_time<<std::endl;
            std::cout<<"diff_curr_gnss2mls:"<<diff_curr_gnss2mls<<", diff_next:"<<diff_next<<std::endl;
            if (diff_curr_gnss2mls > diff_next)
            {
                continue; // continue to go to the next message
            }

            diff_curr_gnss2mls = gps_time - lidar_end_time; // gps_time = diff_curr_gnss2mls + lidar_end_time
        }

        if(true)
        {
            auto gpsTime = msg->time; //this is the actual GPS time  1980-01-06
            std::cout<<"GPS time from 1980-01-06: "<<gpsTime<<std::endl;
            const int leapSeconds = 18; // Number of leap seconds as of October 2023
            double utcTime = gpsTime - leapSeconds;
            //long totalSeconds = static_cast<long>(utcTime);
            const int secondsInADay = 24 * 60 * 60;
            //int timeOfDaySeconds = totalSeconds % secondsInADay;

            //int hours = timeOfDaySeconds / 3600;
            //int remainingSeconds = timeOfDaySeconds % 3600;
            //int minutes = remainingSeconds / 60;
            //int seconds = remainingSeconds % 60;

            // Format the time as HH:MM:SS
            //char buffer[9];
            //snprintf(buffer, sizeof(buffer), "%02d:%02d:%02d", hours, minutes, seconds);
            //std::string timeOfDay = std::string(buffer);
            // std::cout << "GNSS Time of the day: " << timeOfDay << std::endl;

            // Compute the time of the day in seconds (including fractional part)
            tod = fmod(utcTime, secondsInADay);
            std::cout << "GNSS tod:" << tod << ", diff_curr_gnss2mls:" << diff_curr_gnss2mls << std::endl;
            // add diff_curr_gnss2mls  correction
            tod = tod - diff_curr_gnss2mls;
            std::cout << "Corrected tod of the mls:" << tod << std::endl;
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

                // modification here, enu is the weighted average of all prev enu's
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
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, (R_GNSS_to_MLS * curr_enu) + GNSS_T_wrt_IMU);
        }
        else
        {
            // we cannot use the translation offset yet, sice the rotation is unknown
            gps_pose = first_gps_pose * Sophus::SE3(Eye3d, curr_enu);
        }
        gps_pose.so3() = Sophus::SO3(R_compas);

        // as for ppk --------------------------------------------
        //  auto t_z = als2mls_T.translation();// - V3D(0,0,first_gps_pose.translation()[2]);

        // auto origin_enu_in_mls = (als2mls_T * origin_enu);
        // //t_z[0] += origin_enu_in_mls[0];
        // //t_z[1] += origin_enu_in_mls[1];
        // t_z[2] -= origin_enu_in_mls[2];

        // gps_pose = Sophus::SE3(als2mls_T.so3(), t_z) * Sophus::SE3(Eye3d, curr_enu); //transform enu to local mls
        // gps_pose = gps_pose * Sophus::SE3(Eye3d, GNSS_T_wrt_IMU); //put in MLS frame
        // std::cout<<"HESAI GNSS TIME:"<<msg->time<<std::endl;

        if(false){
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
            // std::cout << "GNSS Time of the day: " << timeOfDay << std::endl;

            // Compute the time of the day in seconds (including fractional part)
            tod = fmod(utcTime, secondsInADay);
            std::cout << "GNSS tod:" << tod << ", diff_curr_gnss2mls:" << diff_curr_gnss2mls << std::endl;
            // add diff_curr_gnss2mls  correction
            tod = tod - diff_curr_gnss2mls;
            std::cout << "Corrected tod of the mls:" << tod << std::endl;
        }
    }

    if (use_postprocessed_gnss)
    {
        // std::cout<<"gnss_measurements[curr_gnss].GPSTime:"<<gnss_measurements[curr_gnss].GPSTime<<std::endl;
        // std::cout<<"gnss_measurements[curr_gnss].original_GPSTime:"<<gnss_measurements[curr_gnss].original_GPSTime<<std::endl;
        while (curr_gnss < total_gnss && gnss_measurements[curr_gnss].GPSTime <= global_gps_time)
        // while (curr_gnss < total_gnss && gnss_measurements[curr_gnss].original_GPSTime <= tod)
        {
            const auto &pose = gnss_measurements[curr_gnss];
            curr_gnss++;

            if (fabs(gnss_measurements[curr_gnss].GPSTime - global_gps_time) > fabs(gnss_measurements[curr_gnss + 1].GPSTime - global_gps_time))
                continue;
            // if (fabs(gnss_measurements[curr_gnss].original_GPSTime - tod) > fabs(gnss_measurements[curr_gnss + 1].original_GPSTime - tod))
            //     continue;

            // curr_gnss++;
            double Heading = pose.Heading * M_PI / 180.;
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

            //-------------------------------------------
            // do the same for raw data
            postprocessed_gps_pose = als2mls_T * Sophus::SE3(gt_rotation, gt_translation);

            postprocessed_gps_pose = postprocessed_gps_pose * Sophus::SE3(Eye3d, GNSS_T_wrt_IMU);
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

            double xy_range = std::sqrt(MLS_pos.x() * MLS_pos.x() + MLS_pos.y() * MLS_pos.y());

            if (xy_range > max_travelled_distance_for_initialization)
            {
                std::cout << "all_theta:" << all_theta.size() << std::endl;
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

                if (!use_ransac_alignment)
                {
                    std::cout << "Use the SVD transformation..." << std::endl;
                    theta_GPS_to_IMU = yaw_svd; //
                    theta_GPS_to_IMU = theta_GPS_to_IMU * M_PI / 180.0;
                }
                else
                {
                    std::cout << "Use the RANSAC transformation..." << std::endl;
                }

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

void GNSS::calibrateGnssExtrinsic(const std::vector<Sophus::SE3> &MLS_pos, const std::vector<Sophus::SE3> &GNSS_pos)
{
    ind.clear();
    all_theta.clear();
    all_gnss.clear();
    all_mls.clear();
    int n = MLS_pos.size()-1;
    for (int i = 0; i < n; i++)
    {
        double theta = gnss::findAngle(Zero3d, GNSS_pos[i].translation(), Zero3d, MLS_pos[i].translation()); // in degrees

        ind.push_back(all_theta.size());
        all_theta.push_back(theta);

        all_gnss.push_back(GNSS_pos[i].translation());
        all_mls.push_back(MLS_pos[i].translation());
    }

    gnss::LineModel ransac = gnss::ransacFitLine(ind, all_theta, 200., .5);
    std::cout << "ransac m:" << ransac.m << ", b:" << ransac.b << std::endl;
    theta_GPS_to_IMU = ransac.b;

    // verify the sign
    theta_GPS_to_IMU = gnss::verifyAngle(theta_GPS_to_IMU, Zero3d, GNSS_pos[n].translation(), Zero3d, MLS_pos[n].translation()); // in degrees
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

    if (!use_ransac_alignment)
    {
        std::cout << "Use the SVD transformation..." << std::endl;
        theta_GPS_to_IMU = yaw_svd; //
        theta_GPS_to_IMU = theta_GPS_to_IMU * M_PI / 180.0;
    }
    else
    {
        std::cout << "Use the RANSAC transformation..." << std::endl;
    }

    R_GNSS_to_MLS << cos(theta_GPS_to_IMU), -sin(theta_GPS_to_IMU), 0,
        sin(theta_GPS_to_IMU), cos(theta_GPS_to_IMU), 0,
        0, 0, 1; // Yaw rotation around the z-axis

    // this will use the one from SVD
    // R_GNSS_to_MLS = R_svd;

    
    plt::show();
    GNSS_extrinsic_init = true;
}

void GNSS::updateExtrinsic(const M3D &R_)
{
    R_GNSS_to_MLS = R_;
    first_gps_pose = Sophus::SE3(Eye3d, (R_GNSS_to_MLS * origin_enu) + GNSS_T_wrt_IMU).inverse();
    // GNSS_extrinsic_init = true;
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

            iss >> measurement.UTCTime >> measurement.GPSTime >> measurement.Easting >> measurement.Northing >>
                measurement.H_Ell >> measurement.Heading >> measurement.Pitch >> measurement.Roll; // >> measurement.AccBdyX >> measurement.AccBdyY >> measurement.AccBdyZ >> measurement.AccBiasX >> measurement.AccBiasY >>
                                                                                                   // measurement.AccBiasZ >> measurement.GyroDriftX >> measurement.GyroDriftY >> measurement.GyroDriftZ >> measurement.Cx11 >> measurement.Cx22 >> measurement.Cx33
                                                                                                   //>> measurement.CxVHH >> measurement.CxVPP >> measurement.CxVRR;
            measurement.original_GPSTime = measurement.GPSTime;                                    // gps time of the week
            measurement.GPSTime = measurement.UTCTime + GPS_TO_UTC_OFFSET;                         // gps global time

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