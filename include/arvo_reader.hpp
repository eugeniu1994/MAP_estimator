#ifndef POINTCLOUD_COMBINER_SRC_MOTIONTRANSFORMER_H_
#define POINTCLOUD_COMBINER_SRC_MOTIONTRANSFORMER_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/copy_point.h>
#include <tf/transform_datatypes.h>
#include <string>
#include <ros/time.h>
#include <tgmath.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <vector>
#include "csv_parser.hpp"
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH
#include <glob.h> // glob(), globfree()
#include "boost/filesystem.hpp"
#include <Eigen/Dense>
#include <time.h>
//#include <boost/date_time.hpp>
#include <boost/algorithm/string.hpp>
#include <numeric>      // std::iota
#include <algorithm>



class measurement
{
public:
    measurement();
    virtual ~measurement();

    void initTimeFromMicroSec(const std::uint64_t &timeMicroSec);
    bool updateFromTopOfHourTime(double timeTopOfHour);

    inline bool operator<(const measurement &m) const;
    inline bool operator>(const measurement &m) const;
    inline bool operator<=(const measurement &m) const;
    inline bool operator>=(const measurement &m) const;
    inline bool operator==(const measurement &m) const;
    inline double operator-(const measurement &m) const;

    std::uint64_t timeHourSec_; // full hours in seconds
    double timeTopOfHour_;      // top of hour seconds
    tf::Transform tf_;

    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;

    Eigen::Vector3d raw_acc;
    Eigen::Vector3d raw_gyro;
    bool has_raw = false;

    double weekTimeSec_;
};

class trajectoryReader
{
public:
    trajectoryReader(const std::string &trajectory_file);
    virtual ~trajectoryReader();

    // void initialize(const unsigned int &unixTimeSecs);
    bool getTransform(measurement &tf);
    bool init(measurement &tf);
    bool getAllTransform(measurement &first_point_measurement, measurement &last_point_measurement,
                         std::vector<Eigen::Vector3d> &_acc, std::vector<Eigen::Vector3d> &_gyro,
                         std::vector<double> &_imu_times, double &firstWeekTimeSec, std::vector<tf::Transform> &all_gt_T, tf::Transform &tf_first_, tf::Transform &T_lidar4reference_);
    
    bool getAllTransform(measurement &first_point_measurement, measurement &last_point_measurement,
                         std::vector<measurement> &_all_m, double &firstWeekTimeSec, tf::Transform &tf_first_, tf::Transform &T_lidar4reference_);
    
    bool getLastTransform(measurement &last_point_measurement);
    
    double getTimeHours();
    void printLowerMeasBound();
    void printUpperMeasBound();

private:
    void defineAxesOffset(std::vector<std::string> row);

    tf::Transform T_axesOffset_;
    std::vector<measurement> measurements_;
    std::vector<measurement>::iterator measA_it_; // Lower measurement bound
    std::vector<measurement>::iterator measB_it_; // Upper measurement bound

    std::vector<tf::Transform> trajectory_;
};

namespace Utils {

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
    template<typename T>
    std::vector <size_t> sort_indexes(const std::vector <T> &v) {

        // initialize original index locations
        std::vector <size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                         [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

        return idx;
    }


    inline void savePosesAsCSV(const std::vector <Eigen::Matrix4f> &trajectory, const std::string &filename) {
        std::ofstream file;
        int precision = 10;
        int width = precision + 6;

        file << std::fixed << std::setprecision(precision);
        file.open(filename);

        Eigen::Vector3f prevEulerAngles;
        bool initialized = false;
        for (auto &pose: trajectory) {
            tf::Matrix3x3 R(pose(0, 0), pose(0, 1), pose(0, 2),
                            pose(1, 0), pose(1, 1), pose(1, 2),
                            pose(2, 0), pose(2, 1), pose(2, 2));

            double roll, pitch, yaw;
            R.getEulerYPR(yaw, pitch, roll);

            while (roll < 0) roll += 2 * M_PI;
            while (pitch < 0) pitch += 2 * M_PI;
            while (yaw < 0) yaw += 2 * M_PI;

            file << std::setw(width) << pose(0, 3) << " "
                 << std::setw(width) << pose(1, 3) << " "
                 << std::setw(width) << pose(2, 3) << " "
                 << std::setw(width) << roll << " "
                 << std::setw(width) << pitch << " "
                 << std::setw(width) << yaw << std::endl;
        }

        file.close();
    }


    inline void printProgress(float progress) {
        int barWidth = 70;

        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }


    inline void glob(const std::string &pattern, std::vector <std::string> &filenames) {
        using namespace std;

        // glob struct resides on the stack
        glob_t glob_result;
        memset(&glob_result, 0, sizeof(glob_result));

        // do the glob operation
        int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
        if (return_value != 0) {
            globfree(&glob_result);
            stringstream ss;
            ss << "glob() failed with return_value " << return_value << endl;
            throw std::runtime_error(ss.str());
        }

        // collect all the filenames into a std::vector<std::string>
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            filenames.push_back(string(glob_result.gl_pathv[i]));
        }

        std::sort(filenames.begin(), filenames.end());

        // cleanup
        globfree(&glob_result);
    }


// Return true if directory was created and false if failed or directory exists already
    inline bool mkdir(std::string path) {
        if (!boost::filesystem::exists(path.c_str())) {
            if (!boost::filesystem::create_directories(path.c_str())) {
                ROS_ERROR_STREAM("Failed to create directory " << path);
                return false;
            }
            return true;
        } else
            return false;
    }


    inline std::vector <std::string> splitStr(const std::string &str, const std::string &delims = " ") {
        std::vector <std::string> strings;
        boost::split(strings, str, boost::is_any_of(delims));

        return strings;
    }


    inline int getDayOfWeekIndex(const time_t &unixTime) {
        struct tm *timeinfo;
        timeinfo = gmtime(&unixTime);

        return timeinfo->tm_wday;
    }

/*inline int getDayOfWeekIndex2(const int &year = 1970, const int &month = 1, const int &day = 1) {
  boost::gregorian::date date(year, month, day);
  return date.day_of_week();
}*/


    inline time_t dateTime2UnixTime(const int &year = 1970,
                                    const int &month = 1,
                                    const int &day = 1,
                                    const int &hour = 0,
                                    const int &min = 0,
                                    const int &sec = 0) {

        struct tm timeinfo;
        timeinfo.tm_year = year - 1900;
        timeinfo.tm_mon = month - 1;
        timeinfo.tm_mday = day;
        timeinfo.tm_hour = hour;
        timeinfo.tm_min = min;
        timeinfo.tm_sec = sec;

        return timegm(&timeinfo);
    }


    inline ros::Time dateTime2rosTime(const int &year = 1970,
                                      const int &month = 1,
                                      const int &day = 1,
                                      const int &hour = 0,
                                      const int &min = 0,
                                      const int &sec = 0) {

        return ros::Time(dateTime2UnixTime(year, month, day, hour, min, sec));
    }

    inline double deg2rad(const double &rad) {
        return M_PI * (rad / 180.);
    }

    inline double rad2deg(const double &deg) {
        return 180. * (deg / M_PI);
    }

    inline void printTransformationMatrix(const tf::Transform &tf) {
        int precision = 6;
        std::string printTitle = "Transformation matrix: ";

        std::cout << std::fixed;
        std::cout << std::setprecision(precision);
        std::cout << printTitle;

        tf::Vector3 translation = tf.getOrigin();
        tf::Matrix3x3 rotation = tf.getBasis();

        double X = translation.getX();
        double Y = translation.getY();
        double Z = translation.getZ();

        int x_len = X < 0 ? std::to_string(int(ceil(X))).length() : std::to_string(int(floor(X))).length();
        int y_len = Y < 0 ? std::to_string(int(ceil(Y))).length() : std::to_string(int(floor(Y))).length();
        int z_len = Z < 0 ? std::to_string(int(ceil(Z))).length() : std::to_string(int(floor(Z))).length();

        int width = x_len > y_len ? x_len : y_len;
        width = z_len > width ? z_len : width;

        for (int i = 0; i < 3; i++) {
            tf::Vector3 row = rotation.getRow(i);
            double x = row.getX();
            double y = row.getY();
            double z = row.getZ();

            if (i > 0) std::cout << std::string(printTitle.length(), ' ');
            if (x >= 0) std::cout << " ";
            std::cout << x << "  ";

            if (y >= 0) std::cout << " ";
            std::cout << y << "  ";

            if (z >= 0) std::cout << " ";
            std::cout << z << "  ";

            // Print translation part of matrix (last element of each row)
            if (i == 0) {
                if (X >= 0) std::cout << " ";
                std::cout << std::string(width - x_len, ' ') << X << std::endl;
            }
            if (i == 1) {
                if (Y >= 0) std::cout << " ";
                std::cout << std::string(width - y_len, ' ') << Y << std::endl;
            }
            if (i == 2) {
                if (Z >= 0) std::cout << " ";
                std::cout << std::string(width - z_len, ' ') << Z << std::endl;
            }
        }

        // Print the padding to create proper 4 x 4 transformation matrix
        std::cout << std::string(printTitle.length() + 1, ' ');
        std::cout << "0." << std::string(precision + 3, ' ');
        std::cout << "0." << std::string(precision + 3, ' ');
        std::cout << "0." << std::string(precision + 3, ' ');
        std::cout << std::string(width - 1, ' ') << "1." << std::endl;;
    }

    inline void printTransformationXYZRPY(const tf::Transform &tf) {
        double yaw, pitch, roll;
        tf.getBasis().getEulerYPR(yaw, pitch, roll);
        tf::Vector3 origin = tf.getOrigin();
        std::cout << origin.getX() << " "
                  << origin.getY() << " "
                  << origin.getZ() << " "
                  << rad2deg(roll) << " "
                  << rad2deg(pitch) << " "
                  << rad2deg(yaw) << std::endl;
    }

    inline void printTransformationXYZRPY(const Eigen::Matrix4f &tf) {
        Eigen::Vector3f RPY = tf.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
        std::cout << tf.block<3, 1>(0, 3).transpose() << " " << RPY.transpose() << std::endl;
    }

    inline Eigen::Matrix4f getEigenMatrix4f(tf::Transform &tf) {
        tf::Vector3 t = tf.getOrigin();
        tf::Matrix3x3 R = tf.getBasis();
        tf::Vector3 row0(R.getRow(0)), row1(R.getRow(1)), row2(R.getRow(2));

        Eigen::Matrix4f tf_mat = Eigen::Matrix4f::Identity();
        tf_mat.block<3, 3>(0, 0) << row0.getX(), row0.getY(), row0.getZ(),
                row1.getX(), row1.getY(), row1.getZ(),
                row2.getX(), row2.getY(), row2.getZ();
        tf_mat.block<3, 1>(0, 3) << t.getX(), t.getY(), t.getZ();

        /*std::cout << "getEigenMatrix4f" << std::endl;
        printTransformationXYZRPY(tf);
        printTransformationXYZRPY(tf_mat);
        std::cout << tf_mat << std::endl;*/

        return tf_mat;
    }


    inline tf::Transform xyzypr2tf(const double &x,
                                   const double &y,
                                   const double &z,
                                   const double &yaw,
                                   const double &pitch,
                                   const double &roll) {

        tf::Vector3 translation(x, y, z);
        tf::Matrix3x3 rotation;
        rotation.setEulerYPR(yaw, pitch, roll);
        //rotation.setRPY(roll, pitch, yaw);
        //tf::Quaternion quaternion( yaw, pitch, roll );
        tf::Transform tf(rotation, translation);

        /*if (quaternion.getW() < 0) {
          //std::cout << "changing sign: " << quaternion.getW();

          quaternion = -quaternion;
          tf.setRotation (quaternion);
          //std::cout << " " << quaternion.getW() << " " << tf.getRotation().getW() << std::endl;
        }*/

        return tf;//tf::Transform(quaternion, translation);
    }


    inline Eigen::Quaterniond
    novatelEulerAngles2Quaternion(const double &roll, const double &pitch, const double &yaw) {
        Eigen::AngleAxisd Rx(deg2rad(roll), Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd Ry(deg2rad(pitch), Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd Rz(deg2rad(yaw), Eigen::Vector3d::UnitZ());

        return Rz * Rx * Ry;
    }


    inline tf::Transform novatelMeasurement2tf(const std::vector <std::string> &row) {
        double x = std::stod(row[3]);
        double y = std::stod(row[4]);
        double z = std::stod(row[5]);
        double omega = deg2rad(std::stod(row[6]));
        double phi = deg2rad(std::stod(row[7]));
        double kappa = deg2rad(std::stod(row[8]));

        return xyzypr2tf(x, y, z, kappa, phi, omega);
    }


} // Utils namespace


measurement::measurement()
    : timeHourSec_(0),
      timeTopOfHour_(-1.0f)
{
    tf_.setIdentity();
}

measurement::~measurement()
{
    // TODO Auto-generated destructor stub
}

void measurement::initTimeFromMicroSec(const std::uint64_t &timeMicroSec)
{
    timeHourSec_ = timeMicroSec / std::uint64_t(3.6e9) * 3600; // Define full hours
    timeTopOfHour_ =
        double(timeMicroSec % std::uint64_t(3.6e9)) / 1.0e6f; // Mod for top of hour usecs + divide to scale secs
    ROS_DEBUG_STREAM("Time initialized from microsecs " << timeMicroSec << " to hour secs "
                                                        << timeHourSec_ << " and top of hour time " << timeTopOfHour_);
}

bool measurement::updateFromTopOfHourTime(double timeTopOfHour)
{
    // In some occasions point time can actually be > 3600, this is caused by the Velodyne driver,
    // therefore have to take modulus here
    timeTopOfHour = fmod(timeTopOfHour, 3600);
    // If hour has not changed, keep the hours and just update the TOH value
    if (fabs(timeTopOfHour - timeTopOfHour_) < 10.0f)
    {
        timeTopOfHour_ = timeTopOfHour;
        return true;
    }
    else if (timeTopOfHour - timeTopOfHour_ < 0)
    {
        timeHourSec_ += 3.6e3;
        ROS_DEBUG_STREAM("Hour changed, old time " << timeTopOfHour_ << ", new time " << timeTopOfHour);
        timeTopOfHour_ = timeTopOfHour;
        return true;
    }
    else
    {
        timeHourSec_ -= 3.6e3;
        ROS_DEBUG_STREAM("Hour changed, old time " << timeTopOfHour_ << ", new time " << timeTopOfHour);
        timeTopOfHour_ = timeTopOfHour;
        return true;
    }

    ROS_ERROR_STREAM("Exception, time was not updated, oldtime " << timeTopOfHour_ << ", new time " << timeTopOfHour);
    return false;
}

inline bool measurement::operator<(const measurement &m) const
{
    return ((timeHourSec_ < m.timeHourSec_) ||
            ((timeHourSec_ == m.timeHourSec_) && (timeTopOfHour_ < m.timeTopOfHour_)));
}

inline bool measurement::operator>(const measurement &m) const
{
    return ((timeHourSec_ > m.timeHourSec_) ||
            ((timeHourSec_ == m.timeHourSec_) && (timeTopOfHour_ > m.timeTopOfHour_)));
}

inline bool measurement::operator<=(const measurement &m) const
{
    return ((*this < m) || (*this == m));
}

inline bool measurement::operator>=(const measurement &m) const
{
    return ((*this > m) || (*this == m));
}

inline bool measurement::operator==(const measurement &m) const
{
    return ((timeHourSec_ == m.timeHourSec_) && (timeTopOfHour_ == m.timeTopOfHour_));
}

inline double measurement::operator-(const measurement &m) const
{
    return (double(timeHourSec_ - m.timeHourSec_) + (timeTopOfHour_ - m.timeTopOfHour_));
}

inline tf::Transform xyzypr2tf_(const double &x, const double &y, const double &z,
                                const double &yaw, const double &pitch, const double &roll)
{
    tf::Vector3 translation(x, y, z);
    tf::Matrix3x3 rotation;
    rotation.setEulerYPR(yaw, pitch, roll);
    // rotation.setRPY(roll, pitch, yaw);
    // tf::Quaternion quaternion( yaw, pitch, roll );
    tf::Transform tf(rotation, translation);

    /*if (quaternion.getW() < 0) {
      //std::cout << "changing sign: " << quaternion.getW();

      quaternion = -quaternion;
      tf.setRotation (quaternion);
      //std::cout << " " << quaternion.getW() << " " << tf.getRotation().getW() << std::endl;
    }*/

    return tf; // tf::Transform(quaternion, translation);
}

inline time_t dateTime2UnixTime_(const int &year = 1970,
                                 const int &month = 1,
                                 const int &day = 1,
                                 const int &hour = 0,
                                 const int &min = 0,
                                 const int &sec = 0)
{

    struct tm timeinfo;
    timeinfo.tm_year = year - 1900;
    timeinfo.tm_mon = month - 1;
    timeinfo.tm_mday = day;
    timeinfo.tm_hour = hour;
    timeinfo.tm_min = min;
    timeinfo.tm_sec = sec;

    return timegm(&timeinfo);
}

inline int getDayOfWeekIndex_(const time_t &unixTime)
{
    struct tm *timeinfo;
    timeinfo = gmtime(&unixTime);

    return timeinfo->tm_wday;
}


trajectoryReader::trajectoryReader(const std::string &trajectory_file)
{
    T_axesOffset_.setIdentity();

    std::ifstream infile(trajectory_file);
    aria::csv::CsvParser parser(infile);

    // Initialize coordinate system and search for first measurement
    bool axesInitialized = false;
    bool measurementsInitialized = false;
    std::uint64_t fullWeekSecs = 0;
    std::map<std::string, int> paramMap;

    for (auto &row : parser)
    {
        // Initialize full weeks in UNIX time
        if (row[0] == "Project:")
        {
            std::vector<std::string> splitProjectName = Utils::splitStr(row[1], "_");
            std::vector<unsigned int> yearMonthDay = {std::stod(splitProjectName[0].substr(0, 4)),
                                                      std::stod(splitProjectName[0].substr(4, 2)),
                                                      std::stod(splitProjectName[0].substr(6, 2))};
            fullWeekSecs = dateTime2UnixTime_(yearMonthDay[0], yearMonthDay[1], yearMonthDay[2]);
            fullWeekSecs -= 86400. * getDayOfWeekIndex_(fullWeekSecs);
        }

        // Initialize axes according to how they are defined in the postprocessed trajectory
        else if (row[0] == "Axes:")
        {
            defineAxesOffset(row);
            axesInitialized = true;
        }

        // Initialize parameters
        else if (row[0] == "UTCTime")
        {
            for (int i = 0; i < row.size(); i++)
                paramMap[row[i]] = i;
        }

        // Search for first measurement
        else if (row[0] == "(sec)")
        {
            measurementsInitialized = true;
            break;
        }
    }

    if (!fullWeekSecs)
        ROS_ERROR_STREAM(
            "trajectoryReader: Could not find a row starting with 'Project:', Data collection date can not be defined.");

    if (!axesInitialized)
        ROS_ERROR_STREAM(
            "trajectoryReader: Could not find a row starting with 'Axes:', Offset of axes is not defined! Errors might occur.");

    if (!measurementsInitialized)
        ROS_ERROR_STREAM(
            "trajectoryReader: Could not find a row starting with '(sec)', start of measurements could not be defined.");

    // Now search through the whole trajectory
    // std::string date = "";
    double prevWeekTimeSec = -1;

    for (auto &row : parser)
    {
        // Extract parameters
        double weekTimeSec = std::stod(row[paramMap["UTCTime"]]);
        double x = std::stod(row[paramMap["Easting"]]);
        double y = std::stod(row[paramMap["Northing"]]);
        double z = std::stod(row[paramMap["H-Ell"]]);
        double omega = std::stod(row[paramMap["Omega"]]);
        double phi = std::stod(row[paramMap["Phi"]]);
        double kappa = std::stod(row[paramMap["Kappa"]]);

        if (weekTimeSec >= 604800)
            ROS_ERROR_STREAM(
                "trajectoryReader: Undefined values in the trajectory file, UTC time exceeds 604800 secs in a week, "
                << weekTimeSec << "secs.");
        else if (weekTimeSec - prevWeekTimeSec < 0 && prevWeekTimeSec >= 0)
            ROS_ERROR_STREAM("trajectoryReader: Undefined behavior, UTC week time secs flipped from " << prevWeekTimeSec
                                                                                                      << " to "
                                                                                                      << weekTimeSec
                                                                                                      << ".");
        prevWeekTimeSec = weekTimeSec;

        double AngRateX = std::stod(row[paramMap["AngRateX"]]);
        double AngRateY = std::stod(row[paramMap["AngRateY"]]);
        double AngRateZ = std::stod(row[paramMap["AngRateZ"]]);

        double AccBdyX = std::stod(row[paramMap["AccBdyX"]]);
        double AccBdyY = std::stod(row[paramMap["AccBdyY"]]);
        double AccBdyZ = std::stod(row[paramMap["AccBdyZ"]]);

        measurement m;
        m.timeTopOfHour_ = std::fmod(weekTimeSec, 3600.);
        m.timeHourSec_ = fullWeekSecs + (unsigned int)(weekTimeSec - m.timeTopOfHour_);
        m.tf_ = xyzypr2tf_(x, y, z, kappa, phi, omega) * T_axesOffset_;
        m.weekTimeSec_ = weekTimeSec;

        m.acc = Eigen::Vector3d(AccBdyX, AccBdyY, AccBdyZ);
        m.gyro = Eigen::Vector3d(AngRateX, AngRateY, AngRateZ);

        measurements_.push_back(m);
    }
    std::cout << "loaded " << measurements_.size() << " GT measurements" << std::endl;
    // Initialize iterators to go through the measurements
    measA_it_ = measurements_.begin();
    measB_it_ = measA_it_ + 1;
}

trajectoryReader::~trajectoryReader()
{
    // TODO Auto-generated destructor stub
}

void trajectoryReader::defineAxesOffset(std::vector<std::string> row)
{
    // Separate direction from the string
    std::string x_direction = row[1].substr(2, row[1].length() - 3);
    std::string y_direction = row[2].substr(2, row[2].length() - 3);
    std::string z_direction = row[3].substr(2);

    // Assumption right hand coordinate system: x-right, y-forward, z-up
    double xx = 0, xy = 0, xz = 0,
           yx = 0, yy = 0, yz = 0,
           zx = 0, zy = 0, zz = 0;

    if (x_direction == "left")
        xx = -1;
    else if (x_direction == "right")
        xx = 1;
    else if (x_direction == "backward")
        xy = -1;
    else if (x_direction == "forward")
        xy = 1;
    else if (x_direction == "down")
        xz = -1;
    else if (x_direction == "up")
        xz = 1;
    else
        ROS_ERROR_STREAM("trajectoryReader: X-direction '" << x_direction << "' not recognized.");

    if (y_direction == "left")
        yx = -1;
    else if (y_direction == "right")
        yx = 1;
    else if (y_direction == "backward")
        yy = -1;
    else if (y_direction == "forward")
        yy = 1;
    else if (y_direction == "down")
        yz = -1;
    else if (y_direction == "up")
        yz = 1;
    else
        ROS_ERROR_STREAM("trajectoryReader: Y-direction '" << y_direction << "' not recognized.");

    if (z_direction == "left")
        zx = -1;
    else if (z_direction == "right")
        zx = 1;
    else if (z_direction == "backward")
        zy = -1;
    else if (z_direction == "forward")
        zy = 1;
    else if (z_direction == "down")
        zz = -1;
    else if (z_direction == "up")
        zz = 1;
    else
        ROS_ERROR_STREAM("trajectoryReader: Z-direction '" << z_direction << "' not recognized.");

    tf::Matrix3x3 R(xx, xy, xz, yx, yy, yz, zx, zy, zz);
    T_axesOffset_.setBasis(R);
}


bool trajectoryReader::getAllTransform(measurement &first_point_measurement, measurement &last_point_measurement,
                                       std::vector<Eigen::Vector3d> &_acc, std::vector<Eigen::Vector3d> &_gyro,
                                       std::vector<double> &_imu_times, double &firstWeekTimeSec, 
                                       std::vector<tf::Transform> &all_gt_T, 
                                       tf::Transform &tf_first_, tf::Transform &T_lidar4reference_)
{

    for (auto it = measA_it_; it != measurements_.end(); it++)
    {
        if (*it >= last_point_measurement)
        { // reached the end
            std::cout<<"reached the end"<<std::endl;
            break;
        }
        _acc.push_back(it->acc);
        _gyro.push_back(it->gyro);
        _imu_times.push_back(it->weekTimeSec_ - firstWeekTimeSec);
        all_gt_T.push_back(tf_first_ * (it->tf_ * T_lidar4reference_));
    }
    return _acc.size() > 0 ? true : false;
}

bool trajectoryReader::getAllTransform(measurement &first_point_measurement, measurement &last_point_measurement,
                                       std::vector<measurement> &_all_m, double &firstWeekTimeSec,
                                         tf::Transform &tf_first_, tf::Transform &T_lidar4reference_)
{

    for (auto it = measA_it_; it != measurements_.end(); it++)
    {
        if (*it >= last_point_measurement)
        { // reached the end
            break;
        }
        measurement _m = *it;
        //_m.weekTimeSec_ -= firstWeekTimeSec;
        _m.tf_ = tf_first_ * (it->tf_ * T_lidar4reference_);
        _all_m.push_back(_m);
    }
    return _all_m.size() > 0 ? true : false;
}


bool trajectoryReader::getTransform(measurement &m)
{
    // Need to search forward for time bounds
    if (*measB_it_ < m)
    {
        for (auto it = measB_it_; it != measurements_.end(); it++)
        {
            if (m <= *it)
            {
                measB_it_ = it;
                measA_it_ = measB_it_ - 1;
                break;
            }
        }
    }
    // Need to search backwards for time bounds
    else if (m < *measA_it_)
    {
        for (auto it = measA_it_; it != measurements_.begin(); it--)
        {
            if (*it <= m)
            {
                measA_it_ = it;
                measB_it_ = measA_it_ + 1;
                break;
            }
        }
    }

    if (m < *measA_it_ || *measB_it_ < m)
    {
        int indexLower = measA_it_ - measurements_.begin();
        int indexUpper = measB_it_ - measurements_.begin();
        ROS_ERROR_STREAM("trajectoryReader can not find a match for top of hour measurement, "
                         "\nthere might be an error in the synchronization."
                         "\nLower bound hours in secs "
                         << measA_it_->timeHourSec_ << ", TOH secs "
                         << measA_it_->timeTopOfHour_ << " with index " << indexLower << "\nUpper bound hours in secs " << measB_it_->timeHourSec_
                         << ", TOH secs " << measB_it_->timeTopOfHour_ << " with index "
                         << indexUpper << "\nMeasurement time should be between bounds but hours in sec is "
                         << m.timeHourSec_ << ", and TOH secs " << m.timeTopOfHour_);
        throw std::runtime_error("stop execution here!");
        return false;
    }
    // If the time is same as the GNSS INS lower bound time, use that transformation
    else if (m == *measA_it_)
    {
        m.tf_ = measA_it_->tf_;
        m.acc = measA_it_->acc;
        m.gyro = measA_it_->gyro;
        m.weekTimeSec_ = measA_it_->weekTimeSec_;
        return true;
    }
    // If the time is same as the GNSS INS upper bound time, use that transformation
    else if (m == *measB_it_)
    {
        m.tf_ = measB_it_->tf_;
        m.acc = measB_it_->acc;
        m.gyro = measB_it_->gyro;
        m.weekTimeSec_ = measB_it_->weekTimeSec_;
        return true;
    }
    else
    { // Interpolate between lower bound and upper bound measurement
        double proportion = (m - *measA_it_) / (*measB_it_ - *measA_it_);
        tf::Transform tf_A = tf::Transform::getIdentity();
        tf::Transform tf_B = measA_it_->tf_.inverse() * measB_it_->tf_;
        tf::Vector3 translation = tf_A.getOrigin().lerp(tf_B.getOrigin(), proportion);
        tf::Matrix3x3 rotation(tf_A.getRotation().slerp(tf_B.getRotation(), proportion));

        m.tf_ = measA_it_->tf_ * tf::Transform(rotation, translation);
        m.acc = measA_it_->acc;
        m.gyro = measA_it_->gyro;
        m.weekTimeSec_ = measA_it_->weekTimeSec_;
        return true;
    }
}

bool trajectoryReader::init (measurement &tf){
    getTransform(tf);

    // Find the index of measA_it_ relative to the beginning of measurements_
    size_t currentIndex = std::distance(measurements_.begin(), measA_it_);
    size_t distanceToEnd = measurements_.size() - currentIndex - 1;
    
    currentIndex -= 5;

    std::cout << "Current index of measA_it_: " << currentIndex << std::endl;
    std::cout << "Distance to the end: " << distanceToEnd << std::endl;

    if(currentIndex > 0){
        std::cout << "Init measurements_ : " << measurements_.size() << std::endl;
        measurements_.erase(measurements_.begin(), measurements_.begin() + currentIndex + 1);

        std::cout << "Filtered measurements_ : " << measurements_.size() << std::endl;

        //measA_it_ = measurements_.begin();
        //measB_it_ = measA_it_ + 1;

    }
    return true;
}


double trajectoryReader::getTimeHours()
{
    return measA_it_->timeHourSec_;
}

void trajectoryReader::printLowerMeasBound()
{
    std::cout << measA_it_->timeHourSec_ << " " << measA_it_->timeTopOfHour_ << std::endl;
    // Utils::printTransformationMatrix(measA_it_->tf_);
}

void trajectoryReader::printUpperMeasBound()
{
    std::cout << measB_it_->timeHourSec_ << " " << measB_it_->timeTopOfHour_ << std::endl;
    // Utils::printTransformationMatrix(measB_it_->tf_);
}

//---------------------------------------------------------



#endif