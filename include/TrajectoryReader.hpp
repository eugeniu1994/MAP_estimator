#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include <time.h>
#include <boost/filesystem.hpp>
#include <ros/ros.h>
#include <ctime>

#include "csv_parser.hpp"
#include "utils.h"


// GPS epoch (1980-01-06T00:00:00Z) UTC in UNIX seconds
constexpr int64_t GPS_UNIX_OFFSET = 315964800; // seconds since 1970-01-01
// GPS-UNIX leap seconds (update this if needed)
constexpr int LEAP_SECONDS = 18;



struct Measurement
{
    double GPSTime = 0.0;  //time of the week in seconds
    double UTCTime = 0.0;
    double tod = 0.0; // time of the day

    double utc_usec = 0.0;
    double utc_usec2 = 0.0;


    // position (m) 
    double Easting = 0.0;
    double Northing = 0.0;
    double H_Ell = 0.0;

    //(deg)
    double Phi = 0.0;
    double Omega = 0.0;
    double Kappa = 0.0;

    // gyro angular velocity (deg/s)
    double AngRateX = 0.0;
    double AngRateY = 0.0;
    double AngRateZ = 0.0;

    // acc acceleration  (m/s^2)
    double AccBdyX = 0.0;
    double AccBdyY = 0.0;
    double AccBdyZ = 0.0;

    //(m/s^2)
    double AccBiasX = 0.0;
    double AccBiasY = 0.0;
    double AccBiasZ = 0.0;

    //(m/s)
    double VelBdyX = 0.0;
    double VelBdyY = 0.0;
    double VelBdyZ = 0.0;

    // std position (m)
    double SDEast = 0.0;
    double SDNorth = 0.0;
    double SDHeight = 0.0;

    double E_Sep = 0.0;
    double N_Sep = 0.0;
    double H_Sep = 0.0;

    
};

struct MeasurementQueryResult {
    const Measurement* closest = nullptr;
    const Measurement* prev = nullptr;
    const Measurement* next = nullptr;
};

double gpsTowToRosTime(const int gps_week, const double tow_sec)
{
    // Convert GPS week + TOW -> UNIX time
    int64_t unix_time = GPS_UNIX_OFFSET + gps_week * 7 * 24 * 3600 + static_cast<int64_t>(tow_sec);

    // Adjust for leap seconds
    unix_time -= LEAP_SECONDS;
    //return unix_time;

    // Fractional seconds
    double frac = tow_sec - static_cast<int64_t>(tow_sec);

    return ros::Time(unix_time, static_cast<uint32_t>(frac * 1e9)).toSec();


    // // Full UNIX seconds (with fraction)
    // double unix_time = GPS_UNIX_OFFSET + gps_week * 604800.0 + tow_sec;
    // // Adjust for leap seconds
    // unix_time -= LEAP_SECONDS;
    // // Split into sec + nsec
    // int64_t sec  = static_cast<int64_t>(unix_time);
    // uint32_t nsec = static_cast<uint32_t>((unix_time - sec) * 1e9);

    // return ros::Time(sec, nsec).toSec();
}





inline std::vector<std::string> splitStr(const std::string& str, const std::string& delims = " ") {
  std::vector<std::string> strings;
  boost::split(strings, str, boost::is_any_of(delims));

  return strings;
}

// Convert calendar date to UNIX time (seconds since 1970-01-01)
inline std::time_t dateTime2UnixTime(const int &year  = 1970,
                                const int &month =    1,
                                const int &day   =    1,
                                const int &hour  =    0,
                                const int &min   =    0,
                                const int &sec   =    0) {

  struct tm timeinfo;
  timeinfo.tm_year = year - 1900;
  timeinfo.tm_mon  = month - 1;
  timeinfo.tm_mday = day;
  timeinfo.tm_hour = hour;
  timeinfo.tm_min  = min;
  timeinfo.tm_sec  = sec;

  return timegm(&timeinfo);
}

// Compute GPS week and seconds-of-week
void unixToGps(std::time_t unix_time, int &gps_week, double &tow_sec) {
    std::time_t gps_seconds = unix_time - GPS_UNIX_OFFSET;
    gps_week = gps_seconds / 604800;
    tow_sec  = gps_seconds % 604800;
}

inline int getDayOfWeekIndex(const time_t &unixTime) {
  struct tm *timeinfo;
  timeinfo = gmtime(&unixTime);

  return timeinfo->tm_wday;
}


class TrajectoryReader
{
public:
    TrajectoryReader() = default;

    void read(const std::string &filepath)
    {   
        std::cout<<"TrajectoryReader: read:"<<filepath<<std::endl;
        double angle = -M_PI / 2.0; // -90 degrees in radians
        Rz << cos(angle), -sin(angle), 0,
                sin(angle), cos(angle), 0,
                0, 0, 1;

        std::ifstream infile(filepath);
        if (!infile.is_open())
            throw std::runtime_error("TrajectoryReader: Could not open file: " + filepath);

        aria::csv::CsvParser parser(infile);

        bool inMeasurements = false;
        std::unordered_map<std::string, int> paramMap;


        for (auto &row : parser)
        {
            if (row.size() == 0)
                continue;
            
            // first  fullWeekSecs: 1721865600
            // second fullWeekSecs: 1721520000
            // GPS Week: 2324 TOW at midnight: 345600.000000000000 sec
                  
            if (row[0] == "Project:") { // e.g., row = Project:     Hanhivaara_20250520_2
                try{
                    std::cout<<"Date - row[1]:"<<row[1]<<std::endl;
                    std::vector<std::string> splitProjectName = splitStr(row[1], "_");
                    for (int i=0;i<splitProjectName.size();i++)
                    {
                        std::cout<<" "<<splitProjectName[i]<<std::endl;
                    }
                    std::vector<unsigned int> yearMonthDay = {std::stod (splitProjectName[1].substr (0,4)),
                                                                std::stod (splitProjectName[1].substr (4,2)),
                                                                std::stod (splitProjectName[1].substr (6,2))};
                    
                    // Convert YYYY-MM-DD midnight -> UNIX time (seconds since 1970)
                    fullWeekSecs = dateTime2UnixTime(yearMonthDay[0], yearMonthDay[1], yearMonthDay[2]);
                    std::cout<<"first fullWeekSecs:"<<fullWeekSecs<<std::endl;
                    
                    // Move back to Sunday 00:00 of that GPS week
                    fullWeekSecs -= 86400. * getDayOfWeekIndex(fullWeekSecs);   //86,400 seconds in a day (24 hours * 60 minutes * 60 seconds).
                    std::cout<<"second fullWeekSecs:"<<fullWeekSecs<<std::endl;


                    std::time_t unix_time = dateTime2UnixTime(yearMonthDay[0], yearMonthDay[1], yearMonthDay[2]);
                    unixToGps(unix_time, gps_week, tow_sec);
                    std::cout << "GPS Week: " << gps_week << " TOW at midnight: " << tow_sec << " sec\n";
                }
                catch (const std::exception &e)
                {
                    std::cout<<"Got error in the time parsing"<<std::endl;
                    std::cerr << "Error at: " << e.what() << std::endl;
                    return;
                }
            }

            if (row[0] == "IMU") //&& row[1] == "to" && row[2] == "GNSS"
            {
                std::string line;
                // Reconstruct the full row into a single string
                for (const auto &s : row)
                    line += s + " ";

                // Remove trailing "(...)" part
                auto paren_pos = line.find('(');
                if (paren_pos != std::string::npos)
                    line = line.substr(0, paren_pos);

                double x = 0, y = 0, z = 0;
                if (sscanf(line.c_str(), "IMU to GNSS Antenna Lever Arms: x=%lf, y=%lf, z=%lf", &x, &y, &z) == 3)
                {
                    leverArms_ = Eigen::Vector3d(x, y, z);
                    std::cout << "Parsed lever arms (IMU -> GNSS): " << leverArms_.transpose() << std::endl;
                }
                else
                {
                    std::cerr << "Failed to parse lever arms (IMU -> GNSS): " << line << std::endl;
                }

                continue;
            }
            if (row[0] == "Body") // Body to IMU Rotations
            {
                std::string line;
                for (const auto &s : row)
                    line += s + " ";

                // Remove trailing "(...)" part
                auto paren_pos = line.find('(');
                if (paren_pos != std::string::npos)
                    line = line.substr(0, paren_pos);

                double xRot = 0, yRot = 0, zRot = 0;
                if (sscanf(line.c_str(), "Body to IMU Rotations: xRot=%lf, yRot=%lf, zRot=%lf", &xRot, &yRot, &zRot) == 3)
                {
                    bodyToIMU_ = Eigen::Vector3d(xRot, yRot, zRot);
                    std::cout << "Parsed Body-to-IMU rotations: " << bodyToIMU_.transpose() << std::endl;

                    // Use ZYX Euler convention (NovAtel uses x-y-z rotations)
                    R_body = (Eigen::AngleAxisd(bodyToIMU_[2]* M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(bodyToIMU_[1]* M_PI / 180.0, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(bodyToIMU_[0]* M_PI / 180.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
                }
                else
                {
                    std::cerr << "Failed to parse Body-to-IMU rotations: " << line << std::endl;
                }

                continue;
            }
            if (row[0] == "Axes:")
            {
                parseAxes(row);
                continue;
            }

            /// --- Initialize measurements ---
            if (row[0] == "GPSTime" || row[0] == "UTCTime")
            {
                for (int i = 0; i < row.size(); i++)
                {
                    paramMap[row[i]] = i;
                }
                continue;
            }

            if (row[0] == "(sec)")
            {
                inMeasurements = true;
                continue;
            }

            /// --- Parse measurement rows ---
            if (inMeasurements)
            {
                Measurement m;
                try
                {
                    auto get = [&](const std::string &key) -> double
                    {
                        if (paramMap.find(key) == paramMap.end())
                            return 0.0;
                        return std::stod(row[paramMap[key]]);
                    };

                    m.GPSTime = get("GPSTime"); // time of the week  (tow)

                    m.UTCTime = m.GPSTime - 18.;           // convert to UTC
                    m.tod = std::fmod(m.UTCTime, 86400.0); // Get the time of the day from time of the week;

                    m.UTCTime = get("UTCTime"); //warning - this might be 0 if UTCTime field does not exist 

                    
                    // double weekTimeSec = std::stod (row [paramMap ["UTCTime"]]);
                    double weekTimeSec = m.GPSTime - 18.; 
                    m.utc_usec = static_cast<std::uint64_t> (fullWeekSecs * 1e6 + weekTimeSec * 1e6);
                    // m.utc_usec - scan->header.stamp

                    m.utc_usec2 = gpsTowToRosTime(gps_week, m.GPSTime);


                    m.Easting = get("Easting");
                    m.Northing = get("Northing");
                    m.H_Ell = get("H-Ell");

                    m.Phi = get("Phi");
                    m.Omega = get("Omega");
                    m.Kappa = get("Kappa");

                    m.AccBdyX = get("AccBdyX");
                    m.AccBdyY = get("AccBdyY");
                    m.AccBdyZ = get("AccBdyZ");

                    m.AngRateX = get("AngRateX");
                    m.AngRateY = get("AngRateY");
                    m.AngRateZ = get("AngRateZ");

                    m.AccBiasX = get("AccBiasX");
                    m.AccBiasY = get("AccBiasY");
                    m.AccBiasZ = get("AccBiasZ");

                    m.VelBdyX = get("VelBdyX");
                    m.VelBdyY = get("VelBdyY");
                    m.VelBdyZ = get("VelBdyZ");

                    m.SDEast = get("SDEast");
                    m.SDNorth = get("SDNorth");
                    m.SDHeight = get("SDHeight");

                    m.E_Sep = get("E-Sep");
                    m.N_Sep = get("N-Sep");
                    m.H_Sep = get("H-Sep");

                    measurements_.push_back(m);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Skipping invalid row: " << e.what() << std::endl;
                }
            }

            
        }
        
        std::cout<<"Finished data reading with "<<measurements_.size()<<" measurements"<<std::endl;
    }

    bool init(const double tod)
    {
        if (measurements_.empty())
        {
            curr_index = 0;
            return false;
        }

        if ((tod < measurements_[0].tod) || (tod > measurements_[measurements_.size() - 1].tod))
        {
            std::cout << "Given tod("<< tod << ") is in out of the GNSS time range" << std::endl;
            std::cout << "Start:" << measurements_[0].tod << "(s), end:" << measurements_[measurements_.size() - 1].tod << "(s)" << std::endl;
            return false;
        }

        // simple linear scan to find initial closest index
        size_t idx = 0;
        double minDiff = std::abs(measurements_[0].tod - tod);
        for (size_t i = 1; i < measurements_.size(); ++i)
        {
            double diff = std::abs(measurements_[i].tod - tod);
            // std::cout << "dt:" << diff << std::endl;
            if (diff < minDiff)
            {
                minDiff = diff;
                idx = i;
            }
            else
            {
                initted = true;
                std::cout << "Sync at GNSS time:" << measurements_[i].tod << ", and given tod:" << tod << std::endl;

                // measurements are sorted by TOD, diff increasing -> break
                break;
            }
        }
        curr_index = idx;

        return initted;
    }

    bool init_unix(const double rost_time)
    {
        std::cout<<"init_unix..."<<std::endl;
        if (measurements_.empty())
        {
            curr_index = 0;
            return false;
        }

        if ((rost_time < measurements_[0].utc_usec2) || (rost_time > measurements_[measurements_.size() - 1].utc_usec2))
        {
            std::cout << "Given rost_time("<< rost_time << ") is in out of the UNIX GNSS time range" << std::endl;
            std::cout << "Start:" << measurements_[0].utc_usec2 << "(s), end:" << measurements_[measurements_.size() - 1].utc_usec2 << "(s)" << std::endl;
            return false;
        }

        // simple linear scan to find initial closest index
        size_t idx = 0;
        double minDiff = std::abs(measurements_[0].utc_usec2 - rost_time);
        std::cout<<"init minDiff:"<<minDiff<<std::endl;
        // for (size_t i = 1; i < 15; ++i)
        // {
        //     std::cout<<"\n utc_usec:"<<measurements_[i].utc_usec<<std::endl;
        //     std::cout<<"utc_usec2:"<<measurements_[i].utc_usec2<<std::endl;
        // }
        for (size_t i = 1; i < measurements_.size(); ++i)
        {
            double diff = std::abs(measurements_[i].utc_usec2 - rost_time);
            //std::cout << "dt:" << diff << std::endl;
            if (diff < minDiff)
            {
                minDiff = diff;
                idx = i;
            }
            else
            {
                if(minDiff < 0.5)
                    initted = true;
                else
                    std::cout<<"to big time difference, minDiff:"<<minDiff<<std::endl;

                std::cout << "Sync at UNIX GNSS time:" << measurements_[i].utc_usec2 << ", and given ros_time:" << rost_time << std::endl;
                std::cout << "dt:" << diff<<", minDiff:"<<minDiff << std::endl;
                // measurements are sorted by TOD, diff increasing -> break
                break;
            }
        }
        curr_index = idx;

        return initted;
    }

    M3D RotationMatrix(const Measurement &m) const
    {
        // Convert degrees to radians
        double omega = m.Omega * M_PI / 180.0; // or m.Omega if that’s your naming
        double phi = m.Phi * M_PI / 180.0;     // m.Phi
        double kappa = m.Kappa * M_PI / 180.0; // m.Kappa

        M3D R_;
        R_ << cos(phi) * cos(kappa), -cos(phi) * sin(kappa), sin(phi),
            cos(omega) * sin(kappa) + cos(kappa) * sin(omega) * sin(phi), cos(omega) * cos(kappa) - sin(omega) * sin(phi) * sin(kappa), -cos(phi) * sin(omega),
            sin(omega) * sin(kappa) - cos(omega) * cos(kappa) * sin(phi), cos(kappa) * sin(omega) + cos(omega) * sin(phi) * sin(kappa), cos(omega) * cos(phi);

        return R_;
    }

    void addEarthGravity(const Measurement &m, V3D &raw_gyro, V3D &raw_acc, const double &g)
    {
        auto R = RotationMatrix(m);
        V3D acceleration_in_body_no_gravity(m.AccBdyX, m.AccBdyY, m.AccBdyZ);
        V3D accel_body_with_g = R.inverse() * ((R * acceleration_in_body_no_gravity) + V3D(0, 0, g));
        raw_acc = accel_body_with_g;

        // Convert angular rates from degrees/s to radians/s
        double AngRateX_rad = m.AngRateX * (M_PI / 180.0);
        double AngRateY_rad = m.AngRateY * (M_PI / 180.0);
        double AngRateZ_rad = m.AngRateZ * (M_PI / 180.0);
        raw_gyro = V3D(AngRateX_rad, AngRateY_rad, AngRateZ_rad);
    }

    void addGravity(const Measurement &m, const Sophus::SE3 &curr_pose, V3D &raw_gyro, V3D &raw_acc, const double &g)
    {
        //auto R = RotationMatrix(m);
        const auto &R = curr_pose.so3().matrix(); 

        V3D acceleration_in_body_no_gravity(m.AccBdyX, m.AccBdyY, m.AccBdyZ);
        V3D accel_body_with_g = R.inverse() * ((R * acceleration_in_body_no_gravity) + V3D(0, 0, g));
        raw_acc = accel_body_with_g;

        // Convert angular rates from degrees/s to radians/s
        double AngRateX_rad = m.AngRateX * (M_PI / 180.0);
        double AngRateY_rad = m.AngRateY * (M_PI / 180.0);
        double AngRateZ_rad = m.AngRateZ * (M_PI / 180.0);
        raw_gyro = V3D(AngRateX_rad, AngRateY_rad, AngRateZ_rad);
    }

    Sophus::SE3 interpolateSE3(const Sophus::SE3 &pose1, const double time1,
                               const Sophus::SE3 &pose2, const double time2,
                               const double t)
    {
        Sophus::SE3 delta = pose1.inverse() * pose2;

        // Compute interpolation weight
        double alpha = (t - time1) / (time2 - time1); // when t will be bigger thatn time 2 it will extrapolate

        // exponential map to interpolate
        Sophus::SE3 interpolated = pose1 * Sophus::SE3::exp(alpha * delta.log());

        return interpolated;
    }

    void toSE3(const Measurement &m, Sophus::SE3 &out)
    {
        auto R = RotationMatrix(m);
        V3D t(m.Easting, m.Northing, m.H_Ell);

        auto pose = Sophus::SE3(R, t); //in GNSS

        // axes rotation: NovAtel -> IMU        
        // body to IMU rotation (from Inertial Explorer)

        //rotate -90 z axis to match the imu orientation x- right, y - forward
        //IMU to GNSS Antenna Lever Arms: x=0.000, y=0.142, z=-0.400 m (x-right, y-fwd, z-up)
        
        //out = pose * Sophus::SE3(Rz, leverArms_);
        //do not use the leverArms_, the trajectory is in IMU already, just rotate to match the IMU installation
        
        out = pose * Sophus::SE3(Rz, V3D(0,0,0));
    }

    MeasurementQueryResult queryMeasurement(const double tod) {
        MeasurementQueryResult result;
        if (measurements_.empty())
            return result;

        size_t idx = curr_index;

        // walk forward
        while (idx + 1 < measurements_.size() && measurements_[idx + 1].tod < tod) {
            ++idx;
        }

        // walk backward if necessary
        while (idx > 0 && measurements_[idx].tod > tod) {
            --idx;
        }

        curr_index = idx;

        // determine closest
        const Measurement* prevM = (idx > 0) ? &measurements_[idx - 1] : nullptr;
        const Measurement* nextM = (idx + 1 < measurements_.size()) ? &measurements_[idx + 1] : nullptr;
        const Measurement* currentM = &measurements_[idx];

        if (!prevM) {
            result.closest = currentM;
            result.prev = nullptr;
            result.next = nextM;
        } else if (!nextM) {
            result.closest = currentM;
            result.prev = prevM;
            result.next = nullptr;
        } else {
            // closest between current and next
            double diffCurr = std::abs(currentM->tod - tod);
            double diffNext = std::abs(nextM->tod - tod);
            if (diffCurr <= diffNext) {
                result.closest = currentM;
            } else {
                result.closest = nextM;
                idx = idx + 1;
                curr_index = idx;
            }
            result.prev = prevM;
            result.next = nextM;
        }

        return result;
    }

    MeasurementQueryResult queryMeasurementUnix(const double ros_time) {
        MeasurementQueryResult result;
        if (measurements_.empty())
            return result;

        size_t idx = curr_index;

        // walk forward
        while (idx + 1 < measurements_.size() && measurements_[idx + 1].utc_usec2 < ros_time) {
            ++idx;
        }

        // walk backward if necessary
        while (idx > 0 && measurements_[idx].utc_usec2 > ros_time) {
            --idx;
        }

        curr_index = idx;

        // determine closest
        const Measurement* prevM = (idx > 0) ? &measurements_[idx - 1] : nullptr;
        const Measurement* nextM = (idx + 1 < measurements_.size()) ? &measurements_[idx + 1] : nullptr;
        const Measurement* currentM = &measurements_[idx];

        if (!prevM) {
            result.closest = currentM;
            result.prev = nullptr;
            result.next = nextM;
        } else if (!nextM) {
            result.closest = currentM;
            result.prev = prevM;
            result.next = nullptr;
        } else {
            // closest between current and next
            double diffCurr = std::abs(currentM->utc_usec2 - ros_time);
            double diffNext = std::abs(nextM->utc_usec2 - ros_time);
            if (diffCurr <= diffNext) {
                result.closest = currentM;
            } else {
                result.closest = nextM;
                idx = idx + 1;
                curr_index = idx;
            }
            result.prev = prevM;
            result.next = nextM;
        }

        return result;
    }

    Sophus::SE3 closestPose(const double &tod)
    {
        if ((tod < measurements_[0].tod) || (tod > measurements_[measurements_.size() - 1].tod))
        {
            std::cout << "Given tod is in out of the GNSS time range" << std::endl;
            std::cout << "Start:" << measurements_[0].tod << "(s), end:" << measurements_[measurements_.size() - 1].tod << "(s)" << std::endl;
            return Sophus::SE3();
        }

        MeasurementQueryResult result = queryMeasurement(tod);
        Sophus::SE3 p1,p2;
        toSE3(*result.prev, p1);
        toSE3(*result.next, p2);
        return interpolateSE3(p1,result.prev->tod, p2,result.next->tod,tod);
    }

    Sophus::SE3 closestPoseUnix(const double &ros_time)
    {
        if ((ros_time < measurements_[0].utc_usec2) || (ros_time > measurements_[measurements_.size() - 1].utc_usec2))
        {
            std::cout << "Given tod is in out of the GNSS time range" << std::endl;
            std::cout << "Start:" << measurements_[0].utc_usec2 << "(s), end:" << measurements_[measurements_.size() - 1].utc_usec2 << "(s)" << std::endl;
            return Sophus::SE3();
        }

        MeasurementQueryResult result = queryMeasurementUnix(ros_time);
        Sophus::SE3 p1,p2;
        toSE3(*result.prev, p1);
        toSE3(*result.next, p2);
        return interpolateSE3(p1,result.prev->utc_usec2, p2,result.next->utc_usec2,ros_time);
    }

    const std::vector<Measurement> &measurements() const { return measurements_; }
    const Eigen::Vector3d &leverArms() const { return leverArms_; }
    const Eigen::Vector3d &bodyToIMU() const { return bodyToIMU_; }
    bool initted = false;
    int curr_index = 0;
    M3D Rz;

    std::uint64_t fullWeekSecs = 0;
    int gps_week;
    double tow_sec;

private:    
    std::vector<Measurement> measurements_;

    Eigen::Vector3d leverArms_{0, 0, 0}; // (x, y, z) meters
    Eigen::Vector3d bodyToIMU_{0, 0, 0}; // (xRot, yRot, zRot) degrees
    Sophus::SE3 T_axes = Sophus::SE3();
    M3D R_axes = M3D::Identity();
    // Use ZYX Euler convention (NovAtel uses x-y-z rotations)
    Eigen::Matrix3d R_body;
    
    void parseAxes(const std::vector<std::string> &row)
    {
        if (row.size() < 4)
        {
            ROS_ERROR_STREAM("parseAxes: row too short.");
            return;
        }

        // Extract direction strings (remove possible punctuation or spaces)
        auto cleanDir = [](const std::string &s)
        {
            std::string res = s;
            res.erase(0, 2);                           // remove "x-", "y-", "z-" prefix
            res.erase(res.find_last_not_of(" ,") + 1); // trim trailing spaces/commas
            return res;
        };

        std::string xDir = cleanDir(row[1]);
        std::string yDir = cleanDir(row[2]);
        std::string zDir = cleanDir(row[3]);

        std::cout << "x_direction: " << xDir << ", y_direction: " << yDir << ", z_direction: " << zDir << std::endl;

        // Helper lambda to convert string to vector component
        auto dirToVec = [](const std::string &dir) -> Eigen::Vector3d
        {
            if (dir == "left")
                return Eigen::Vector3d(-1, 0, 0);
            if (dir == "right")
                return Eigen::Vector3d(1, 0, 0);
            if (dir == "forward")
                return Eigen::Vector3d(0, 1, 0);
            if (dir == "backward")
                return Eigen::Vector3d(0, -1, 0);
            if (dir == "up")
                return Eigen::Vector3d(0, 0, 1);
            if (dir == "down")
                return Eigen::Vector3d(0, 0, -1);
            ROS_ERROR_STREAM("parseAxes: direction '" << dir << "' not recognized.");
            return Eigen::Vector3d::Zero();
        };

        Eigen::Vector3d xVec = dirToVec(xDir);
        Eigen::Vector3d yVec = dirToVec(yDir);
        Eigen::Vector3d zVec = dirToVec(zDir);

        // Construct rotation matrix
        Eigen::Matrix3d R;
        R.col(0) = xVec;
        R.col(1) = yVec;
        R.col(2) = zVec;
        
        T_axes = Sophus::SE3(R, V3D(0,0,0));
        std::cout << "Parsed axes rotation matrix:\n"
                  << R << std::endl;

        R_axes << 0, -1, 0,   // x (forward) -> -y
                  1,  0, 0,   // y (left)    -> x
                  0,  0, 1;   // z (up)      -> z
// | Trajectory axis | IMU axis |
// | --------------- | -------- |
// | x-forward       | y        |
// | y-left          | -x       |
// | z-up            | z        |

                
    }
};
