#include "Vux_publisher.hpp"
#include <boost/filesystem.hpp>

ros::Publisher point_cloud_pub;

void publishPointCloud()
{
    if (cloud->empty())
    {
        std::cerr << "VUX Point cloud is empty. Skipping publish.\n";
        return;
    }

    for (auto &point : cloud->points)
    {
        // point.time = lidarToUtcTime(global_gps_week, point.time);
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);

    // Get the first point's GPS time and convert to ROS time
    ros::Time first_point_time_ros(cloud->points[0].time);

    cloud_msg.header.stamp = first_point_time_ros; // ros::Time::now();
    cloud_msg.header.frame_id = "VUX";

    point_cloud_pub.publish(cloud_msg);

    std::cout << "\nPublished " << cloud->size() << " points" << ", Header time: " << first_point_time_ros << std::endl;

    cloud->clear();
}

void processFile(const std::string &file_path, ros::Publisher &point_cloud_pub)
{
    try
    {
        std::shared_ptr<basic_rconnection> rc = basic_rconnection::create("file:" + file_path);
        rc->open();

        decoder_rxpmarker dec(rc);
        importer imp(std::cout);
        buffer buf;

        ros::Rate loop_rate(250); // inferred from the data

        unsigned long line = imp.line;
        for (dec.get(buf); !dec.eoi(); dec.get(buf))
        {
            std::cout<<"dispatch imp.line:"<<imp.line<<std::endl;
            imp.dispatch(buf.begin(), buf.end());
            if (imp.line != line)
            {
                line = imp.line;
                ros::spinOnce();
                loop_rate.sleep();
            }
            if (!ros::ok())
                break;
        }

        rc->close();
    }
    catch (std::exception &e)
    {
        std::cerr << "Error processing file " << file_path << ": " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception processing file " << file_path << std::endl;
    }
}

void calculateGlobalTimes()
{
    // Input date and time
    int year = 2024, month = 7, day = 25;
    int timezoneOffset = 2 * 3600; // UTC+2 in seconds
    int gpsUtcOffset = 18;         // GPS-UTC offset in seconds (as of 2024)

    // Define GPS epoch (6 January 1980, 00:00:00 UTC)
    std::tm gpsEpoch = {};
    gpsEpoch.tm_year = 1980 - 1900;
    gpsEpoch.tm_mon = 0; // January
    gpsEpoch.tm_mday = 6;
    gpsEpoch.tm_hour = 0;
    gpsEpoch.tm_min = 0;
    gpsEpoch.tm_sec = 0;
    std::time_t gpsEpochTime = std::mktime(&gpsEpoch);

    // Define UTC epoch (1 January 1970, 00:00:00 UTC)
    std::tm utcEpoch = {};
    utcEpoch.tm_year = 1970 - 1900;
    utcEpoch.tm_mon = 0; // January
    utcEpoch.tm_mday = 1;
    utcEpoch.tm_hour = 0;
    utcEpoch.tm_min = 0;
    utcEpoch.tm_sec = 0;
    std::time_t utcEpochTime = std::mktime(&utcEpoch);

    // Input date in Finland timezone
    std::tm inputDate = {};
    inputDate.tm_year = year - 1900;
    inputDate.tm_mon = month - 1; // Months are 0-based
    inputDate.tm_mday = day;
    inputDate.tm_hour = 0; // TOW = 0 means midnight
    inputDate.tm_min = 0;
    inputDate.tm_sec = 0;

    // Convert to UTC (subtract timezone offset)
    std::time_t utcTime = std::mktime(&inputDate) - timezoneOffset;

    // Calculate GPS time by subtracting the GPS epoch and adjusting for GPS-UTC offset
    std::time_t gpsTime = utcTime - gpsEpochTime + gpsUtcOffset;

    // Print results
    std::cout << "UTC Time (seconds since UTC epoch): " << utcTime - utcEpochTime << " seconds" << std::endl;
    std::cout << "GPS Time (seconds since GPS epoch): " << gpsTime << " seconds" << std::endl;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "mls");
    ros::NodeHandle nh;
    std::cout << std::fixed << std::setprecision(12);

    std::cout << "========================Start VUX publisher==========================" << std::endl;
    point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 10000);

    // get this as param
    std::string folder_path = "/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-B/";
    // Example: Date and GPS seconds of the week

    {

        // Given date and time
        int year = 2024;
        int month = 7; // July
        int day = 25;
        int hour = 0;
        int minute = 0;
        int second = 0;

        // Time zone offset for Finland in summer (UTC+3)
        int timeZoneOffset = 3 * 3600; // 3 hours in seconds

        // Leap seconds difference between GPS and UTC (as of 2023, it's 18 seconds)
        int leapSeconds = 18;

        // Create a tm structure for the local time
        struct tm localTimeStruct = {0};
        localTimeStruct.tm_sec = second;
        localTimeStruct.tm_min = minute;
        localTimeStruct.tm_hour = hour;
        localTimeStruct.tm_mday = day;
        localTimeStruct.tm_mon = month - 1;    // tm_mon is 0-based (0 = January)
        localTimeStruct.tm_year = year - 1900; // tm_year is years since 1900

        // Convert local time to UTC time in seconds
        time_t localTime = mktime(&localTimeStruct); // mktime assumes local time
        time_t utcTime = localTime - timeZoneOffset;

        // Convert UTC time to GPS time in seconds
        time_t gpsTime = utcTime + leapSeconds;

        // Output the results
        std::cout << "Local Time (Finland) in seconds since epoch: " << localTime << std::endl;
        std::cout << "UTC Time in seconds since epoch: " << utcTime << std::endl;
        std::cout << "GPS Time in seconds since epoch: " << gpsTime << std::endl;
    }

    {
        std::cout << "\n\ngpt\n\n"
                  << std::endl;
        calculateGlobalTimes();
    }

    //ros::shutdown();
    //return 0;

    try
    {
        boost::filesystem::path dir(folder_path);

        if (!boost::filesystem::exists(dir) || !boost::filesystem::is_directory(dir))
        {
            std::cerr << "Invalid folder path: " << folder_path << std::endl;
            return 1;
        }

        std::vector<std::string> rxp_files;
        for (const auto &entry : boost::filesystem::directory_iterator(dir))
        {
            if (boost::filesystem::is_regular_file(entry) &&
                entry.path().extension() == ".rxp")
            {
                rxp_files.push_back(entry.path().string());
            }
        }

        if (rxp_files.empty())
        {
            std::cerr << "No .rxp files found in the folder: " << folder_path << std::endl;
            return 1;
        }

        // Sort the files alphabetically
        std::sort(rxp_files.begin(), rxp_files.end());

        for (const auto &file : rxp_files)
        {
            std::cout << "\nProcessing file: " << file << std::endl;
            processFile(file, point_cloud_pub);

            if (!ros::ok())
                break;
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred." << std::endl;
        return 1;
    }

    std::cout << "========================End VUX publisher==========================" << std::endl;
    ros::shutdown();

    return 0;
}