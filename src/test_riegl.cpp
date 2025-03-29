
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <riegl/scanlib.hpp>

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "utils.h"

using namespace scanlib;
using namespace std;

// GPS to UTC offset (19 seconds as of January 2025)
constexpr double GPS_UTC_OFFSET = 18.0;

struct VUX_PointType
{
  PCL_ADD_POINT4D;
  float echo_range; //! echo range in units of meter
  double time;      ////! time stamp in [s]
  double time_sorg; // The timestamp of the start of the rangegate (internal time).
  float amplitude;  //! relative amplitude in [dB]
  float reflectance;
  float deviation;
  unsigned segment; // segment number
  // bool is_line_end;
  int single_echo;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

} EIGEN_ALIGN16; // Align the structure to 16-byte boundary for SSE optimizations

// Register your custom point type with PCL's point cloud library
POINT_CLOUD_REGISTER_POINT_STRUCT(VUX_PointType,
                                  (float, x, x)(float, y, y)(float, z, z)(float, echo_range, echo_range)(double, time, time)(double, time_sorg, time_sorg)(float, amplitude, amplitude)(float, reflectance, reflectance)(float, deviation, deviation)(unsigned, segment, segment)
                                  //(bool, is_line_end, is_line_end)
                                  (int, single_echo, single_echo)) // Register custom fields

pcl::PointCloud<VUX_PointType>::Ptr cloud(new pcl::PointCloud<VUX_PointType>);

ros::Publisher point_cloud_pub;

void publishPointCloud()
{
  if (cloud->empty())
  {
    std::cerr << "Point cloud is empty. Skipping publish.\n";
    return;
  }

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*cloud, cloud_msg);

  // Get the first point's GPS time and convert to ROS time
  double first_point_time_gps = cloud->points[0].time; 
  ros::Time first_point_time_ros(first_point_time_gps - GPS_UTC_OFFSET);

  cloud_msg.header.stamp = first_point_time_ros; // ros::Time::now();
  cloud_msg.header.frame_id = "world";

  point_cloud_pub.publish(cloud_msg);
  
  std::cout << "First point time (GPS): " << first_point_time_gps << " seconds\n";
  std::cout << "Header time (UTC): " << first_point_time_ros << "\n";
  std::cout << "Published " << cloud->size() << " points\n"<< std::endl;

  cloud->clear();
}

// The import class is derived from pointcloud class, which assembles the
// scanner data into distinct targets and computes x,y,z pointcloud data.
// The pointcloud class and its base class have a huge number of overridables
// that give access to all of the scanners data, e.g. "gps" or "housekeeping"
// data. The pointcloud class also has the necessary logic to align the
// data to gps information (if embedded in the rxp stream) and will return
// timestamps in the domain of the gps.
class importer : public pointcloud
{
  ostream &o;
  unsigned long line;
  unsigned long frame;
  string type_id;
  string serial;

  int index = 0;

public:
  importer(ostream &o_)
      : pointcloud(true) // set this to true if you need gps aligned timing
        ,
        o(o_), line(0), frame(0)
  {
    o.precision(10);
  }

protected:
  // Overridden from pointcloud class.
  // Gets called for every decoded echo. point
  void on_echo_transformed(echo_type echo)
  {
    // here we select which target types we are interested in
    // if (pointcloud::first == echo || pointcloud::single == echo)
    if (pointcloud::none != echo)
    {
      // std::cout << "on_echo_transformed is being called--------------------------------------" << std::endl;
      //  targets is a member std::vector that contains all
      //  echoes seen so far, i.e. the current echo is always
      //  indexed by target_count-1.
      target &t(targets[target_count - 1]);

      // transform to polar coordinates
      double range = std::sqrt(t.vertex[0] * t.vertex[0] + t.vertex[1] * t.vertex[1] + t.vertex[2] * t.vertex[2]);
      // if (range > numeric_limits<double>::epsilon())
      if (range > 1 && range < 50)
      {
        //   double phi = atan2(t.vertex[1], t.vertex[0]);
        //   phi = ((phi < 0.0) ? (phi + 2.0 * pi) : phi);
        //   double theta = std::acos(t.vertex[2] / range);
        //   t.vertex[0] = static_cast<float>(range);
        //   t.vertex[1] = static_cast<float>((360.0 / (2.0 * pi)) * theta);
        //   t.vertex[2] = static_cast<float>((360.0 / (2.0 * pi)) * phi);

        VUX_PointType point;
        point.x = t.vertex[0];
        point.y = t.vertex[1];
        point.z = t.vertex[2];

        point.echo_range = t.echo_range;
        point.time = t.time;
        point.amplitude = t.amplitude;
        point.reflectance = t.reflectance;
        point.deviation = t.deviation;
        point.segment = t.segment;
        point.time_sorg = t.time_sorg;
        // point.is_line_end = t.is_line_end;
        if (pointcloud::first == echo)
        {
          point.single_echo = 0;
        }
        else if (pointcloud::single == echo)
        {
          point.single_echo = 1;
        }
        else if (pointcloud::last == echo)
        {
          point.single_echo = 2;
        }

        cloud->points.push_back(point);
      }
      // else
      //   t.vertex[0] = t.vertex[1] = t.vertex[2] = 0.0;

      // // print out the result
      // o << t.vertex[0] << ", " << t.vertex[1] << ", " << t.vertex[2] << ", " << t.time << endl;
    }
  }

  // This function gets called when a new scan line starts in up direction.
  // For a VQ-880-G, VQ-840-G, VUX-120 please see on_line_start_segment_1 and on_line_start_segment_2.
  void on_line_start_up(const line_start_up<iterator_type> &arg)
  {
    line++;
    // pointcloud::on_line_start_up(arg);
    // o << "line start up: " << line << endl;
  }

  // The following function gets called when a scan line has been finished.
  void on_line_stop(const line_stop<iterator_type> &arg)
  {
    pointcloud::on_line_stop(arg);
    o << "line stop: " << line << ",  frame:" << frame << ", points:" << cloud->size() << endl;

    // index++;
    // if(index % 100==0)
    publishPointCloud();

    cloud->clear();
  }

  // This function gets called when new units are available.
  void on_units_4(const units_4<iterator_type> &arg)
  {
    pointcloud::on_units_4(arg);

    // Depending on the used time synchronization method,
    // the external time may contain the epoch and is added to this packet.
    // E.g. using the GPGGA string
    //  epoch_exttime = "UNKNOWN"
    //  clock_source  = "UNKNOWN"
    // E.g. using the GPZDA string
    //  epoch_exttime = e.g. "20230817T00:00:00"
    //  clock_source  = "UTC"
    o << "epoch_exttime: " << arg.epoch_exttime << endl; // epoch of exttime in YYYYMMDDTHH:MM:SS or UNKNOWN
    o << "clock_source: " << arg.clock_source << endl;   // UNKNOWN, RTC, GPS or UTC
  }

  // This function gets called when a new frame starts in up direction.
  // Only 3D scanner (VZ-xxx) can be configured to perform a 3D scan (frame scan).
  void on_frame_start_up(const frame_start_up<iterator_type> &arg)
  {
    pointcloud::on_frame_start_up(arg);
    o << "frame start up: " << ++frame << endl;
  }

  // This function gets called when a new frame starts in down direction.
  // Only 3D scanner (VZ-xxx) can be configured to perform a 3D scan (frame scan).
  void on_frame_start_dn(const frame_start_dn<iterator_type> &arg)
  {
    pointcloud::on_frame_start_dn(arg);
    o << "frame start dn: " << ++frame << endl;
  }

  // This function gets called when a frame has been finished.
  // Only 3D scanner (VZ-xxx) can be configured to perform a 3D scan (frame scan).
  void on_frame_stop(const frame_stop<iterator_type> &arg)
  {
    pointcloud::on_frame_stop(arg);
    o << "frame stop: " << endl;
  }

  // This function gets called when a the scanner emits a notification
  // about an exceptional state.
  void on_unsolicited_message(const unsolicited_message<iterator_type> &arg)
  {
    pointcloud::on_unsolicited_message(arg);

    std::string type;
    if (arg.type == 1)
      type = "INFO: ";
    else if (arg.type == 2)
      type = "WARNING: ";
    else if (arg.type == 3)
      type = "ERROR: ";
    else if (arg.type == 4)
      type = "FATAL: ";

    o << type << arg.message << endl;
  }

  // Decodes the GPS synchronization status and the PPS counter.
  // This function is only called on scanners WITHOUT higher resolution (hr) system time.
  // For scanners WITH higher resolution system time the function on_hk_gps_hr is called.
  void on_hk_gps_ts_status_dop_ucs(const hk_gps_ts_status_dop_ucs<iterator_type> &arg)
  {
    pointcloud::on_hk_gps_ts_status_dop_ucs(arg);

    std::string sync_status = "not synchronized";
    if (arg.SYNC_STATUS == 1)
      sync_status = "lost synchronization";
    else if (arg.SYNC_STATUS == 3)
      sync_status = "correctly synchronized";

    o << "hk gps: " << "sync_status=" << sync_status << "\tPPS_count=" << arg.PPS_CNT << endl;
  }

  // Decodes the GPS synchronization status and the PPS counter.
  // This function is only called on scanners WITH higher resolution (hr) system time.
  // For scanners WITHOUT higher resolution (hr) system time the function on_hk_gps_ts_status_dop_ucs is called.
  void on_hk_gps_hr(const hk_gps_hr<iterator_type> &arg)
  {
    pointcloud::on_hk_gps_hr(arg);

    std::string sync_status = "not synchronized";
    if (arg.SYNC_STATUS == 1)
      sync_status = "lost synchronization";
    else if (arg.SYNC_STATUS == 3)
      sync_status = "correctly synchronized";

    o << "hk gps hr: " << "sync_status=" << sync_status << "\tPPS_count=" << arg.PPS_CNT << endl;
  }

  // This function gets called when a new header is available.
  void on_header(const header<iterator_type> &arg)
  {
    pointcloud::on_header(arg);
    this->type_id = arg.type_id;
    this->serial = arg.serial;
    o << "type id: " << this->type_id << endl;
    o << "serial: " << this->serial << endl;
  }
};

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "mls");
  ros::NodeHandle nh;

  std::cout << "Start main " << std::endl;

  point_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/vux_data", 100);

  try
  {
    std::cout << "argc:" << argc << std::endl;

    // The basic_rconnection class contains the communication protocol between
    // the scanner or file and the program.
    std::shared_ptr<basic_rconnection> rc;
    // rc = basic_rconnection::create(argv[1]);
    std::string path = "file:/media/eugeniu/T7/roamer/03_RIEGL_RAW/02_RXP/VUX-1HA-22-2022-B/240725_092351.rxp"; // 240725_092351.rxp";
    rc = basic_rconnection::create(path);

    rc->open();

    // The decoder class scans off distinct packets from the
    // continuous data stream i.e. the rxp format and manages
    // the packets in a buffer.
    decoder_rxpmarker dec(rc);

    // The importer ( based on pointcloud and basic_packets class)
    // recognizes the packet types and calls into a distinct
    // function for each type. The functions are overidable
    // virtual functions, so a derived class can get a callback
    // per packet type.
    importer imp(cout);

    // The buffer, despite its name is a structure that holds
    // pointers into the decoder buffer thereby avoiding
    // unnecessary copies of the data.
    buffer buf;

    // This is the main loop, alternately fetching data from
    // the buffer and handing it over to the packets recognizer.
    // Please note, that there is no copy overhead for packets
    // which you do not need, since they will never be touched
    // if you do not access them.

    std::cout << "targets:" << imp.targets.size() << std::endl;
    ros::Rate loop_rate(450); // 10Hz once per second

    int index_ = 0;
    for (dec.get(buf); !dec.eoi(); dec.get(buf))
    {
      // std::cout << "\nindex_:" << index_ << std::endl;
      // index_++;
      imp.dispatch(buf.begin(), buf.end());
      // std::cout << "points:" << points.size() << std::endl;
      if (cloud->size() > 0)
      {
        // publishPointCloud();

        ros::spinOnce();
        loop_rate.sleep();
      }
      // if(index_ > 400)
      //   break;

      if (!ros::ok())
        break;
    }

    rc->close();
  }
  catch (std::exception &e)
  {
    std::cerr << e.what() << std::endl;
  }
  catch (...)
  {
    cerr << "unknown exception" << endl;
    return 1;
  }

  std::cout << "It works" << std::endl;
  ros::shutdown();
  return 0;
}
