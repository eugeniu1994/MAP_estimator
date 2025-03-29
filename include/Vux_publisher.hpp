
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <ctime>

#include "utils.h"

#include <riegl/scanlib.hpp>

using namespace scanlib;
using namespace std;

bool verbose = true;

pcl::PointCloud<VUX_PointType>::Ptr cloud(new pcl::PointCloud<VUX_PointType>);

void publishPointCloud();

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
    unsigned long frame;
    string type_id;
    string serial;

public:
    unsigned long line;

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
            if (range > 1 && range < 100)
            {
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
                else
                {
                    point.single_echo = 100;
                }

                cloud->points.push_back(point);
            }
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
        //o << "line stop: " << line << ",  frame:" << frame << ", points:" << cloud->size() << endl;
        publishPointCloud();
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

        o << "\non_hk_gps_ts_status_dop_ucs gps: " << "sync_status=" << sync_status << "\tPPS_count=" << arg.PPS_CNT << endl;
        o << "TOWms (ms): " << arg.TOWms << ", PPS_TIME:" << arg.PPS_TIME << ", systime:" << arg.systime << endl;

        // uint32_t TOWms; // GPS data, GPS Time of Week [ 1 ms].
        // uint32_t PPS_CNT; // GPS data, counter of PPS, increments with each detected PPS trigger - edge.
        // uint32_t PPS_TIME; // GPS data, internal time of PPS trigger edge in units of -units.time_unit
        // uint32_t systime; // internal time in units of units.time_unit

        /*
        // In header: <riegl/ridataspec.hpp>
template<typename it> -
struct hk_gps_ts_status_dop_ucs {

enum @37 { id_main = = 10005, id_sub = = 4 };
// public data members
uint32_t TOWms; // GPS data, GPS Time of Week [ 1 ms].
int32_t ECEFX; // GPS data, ECEF X coordinate [ 1 cm].
int32_t ECEFY; // GPS data, ECEF Y coordinate [ 1 cm].
int32_t ECEFZ; // GPS data, ECEF Z coordinate [ 1 cm].
uint32_t POS_ACC; // GPS data, position accuracy estimate [ 1 cm].
int32_t LONG; // GPS data, longitude [1e-7deg].
int32_t LAT; // GPS data, latitude [1e-7deg].
int32_t HGT_ELL; // GPS data, height above ellipsoid [ 1 mm].
int32_t HGT_SEA; // GPS data, height above mean sea level [ 1 mm].
uint32_t HOR_ACC; // GPS data, horizontal accuracy estimate [ 1 mm].
uint32_t VERT_ACC; // GPS data, vertical accuracy estimate [ 1 mm].
uint8_t STATUS1; // GPS data, 0=no fix; 1=dead reckoning; 2=2D-fix; 3=3D-fix; 4=GPS+dead -
reckoning;5=Time only fix;255=undefined fix.
uint8_t STATUS2; // GPS data, 1 -.. gpsfixOK; 2 -.. diffSoln; 4 -.. wknSet; 8 -.. towSet.
uint8_t STATUS3; // GPS data, 0= no DGPS; 1=PR+PRR Correction; 2=PR+PRR+CP Correction; -
3=high accuracy PR+PRR+CP Correction.
int8_t LEAP_SEC; // GPS data, leap seconds (GPS-UTC) [ 1 s -].
uint32_t systime; // internal time in units of units.time_unit
uint8_t SYNC_STATUS; // PPS pulse and GPS data receive indicator bit 0: at least one -
valid gps information seen bit 1: valid gps information seen during last second bit 7: gps -
detected, gps data received within last second.
uint8_t SAT_NUM; // GPS data, number of satelites used.
uint16_t GEOM_DILUT; // GPS data, geometric dilution of position [0.01].
uint16_t TIME_DILUT; // GPS data, time dilution [0.01].
uint16_t VERT_DILUT; // GPS data, vertical dilution (1D) [0.01].
uint16_t HOR_DILUT; // GPS data, horizontal dilution (2D) [0.01].
uint16_t POS_DILUT; // GPS data, position dilution (3D) [0.01].
int64_t NORTHING; // GPS data, northing [mm].
int64_t EASTING; // GPS data, easting [mm].
int64_t HEIGHT; // GPS data, height [mm].
uint32_t FRAME_ANGLE; // GPS data, frame angle, unit see units.frame_circle_count.
uint32_t PPS_TIME; // GPS data, internal time of PPS trigger edge in units of -
units.time_unit.
uint32_t PPS_CNT; // GPS data, counter of PPS, increments with each detected PPS trigger -
edge.
uint32_t RCV_TIME; // GPS data, internal time of receiving the time information in units -
of units.time_unit.
uint32_t VALID_MASK; // GPS data, valid mask.

        */
    }

    // Decodes the GPS synchronization status and the PPS counter.
    // This function is only called on scanners WITH higher resolution (hr) system time.
    // For scanners WITHOUT higher resolution (hr) system time the function on_hk_gps_ts_status_dop_ucs is called.
    void on_hk_gps_hr(const hk_gps_hr<iterator_type> &arg)
    {
        pointcloud::on_hk_gps_hr(arg);

        if (verbose)
        {
            std::string sync_status = "not synchronized";
            if (arg.SYNC_STATUS == 1)
                sync_status = "lost synchronization";
            else if (arg.SYNC_STATUS == 3)
                sync_status = "correctly synchronized";

            uint32_t TOWms = (uint32_t)arg.TOWms; // GPS data, GPS Time of Week [ 1 ms].
            int64_t LONG = (int64_t)arg.LONG; // GPS data, longitude [1e-9deg].
            int64_t LAT = (int64_t)arg.LAT; // GPS data, latitude [1e-9deg].
            int32_t HGT_ELL = (int32_t)arg.HGT_ELL; // GPS data, height above ellipsoid [ 1 mm].
            int32_t HGT_SEA = (int32_t)arg.HGT_SEA; // GPS data, height above mean sea level [ 1 mm].

            uint8_t STATUS1 = (uint8_t)arg.STATUS1; // GPS data, 0=no fix; 1=dead reckoning; 2=2D-fix; 3=3D-fix; 4=GPS+dead-reckoning;5=Time only fix;255=undefined fix.

            int8_t LEAP_SEC = arg.LEAP_SEC; // GPS data, leap seconds (GPS-UTC) [ 1 s -].
            uint32_t systime = (uint32_t)arg.systime; // internal time in units of units.time_unit
            //uint8_t SYNC_STATUS; // PPS pulse and GPS data receive indicator bit 0: at least one -
            //valid gps information seen bit 1: valid gps information seen during last second bit 7: gps -
            //detected, gps data received within last second.

            int64_t NORTHING = arg.NORTHING; // GPS data, northing [ 1 mm].
            int64_t EASTING = arg.EASTING; // GPS data, easting [ 1 mm].
            int64_t HEIGHT = arg.HEIGHT; // GPS data, height [ 1 mm].

            uint32_t PPS_TIME = arg.PPS_TIME; // GPS data, internal time of PPS trigger edge in units of -units.time_unit.
            uint32_t PPS_CNT = arg.PPS_CNT; // GPS data, counter of PPS, increments with each detected PPS trigger -edge.

            double TOWseconds = TOWms / 1000.0;   // Convert milliseconds to seconds

            o << "\non_hk_gps_hr gps hr: " << "sync_status=" << sync_status << "\tPPS_count=" << PPS_CNT << endl;
            o << "PPS_TIME:"<<PPS_TIME<<", systime:"<<systime<<",  LEAP_SEC:"<<LEAP_SEC<<", TOWms:"<<TOWms<<", STATUS1:"<<STATUS1<<std::endl;
            o << "NORTHING:"<<NORTHING<<", EASTING:"<<EASTING<<", HEIGHT:"<<HEIGHT<<std::endl;
            o << "LONG:"<<LONG<<", LAT:"<<LAT<<", HGT_ELL:"<<HGT_ELL<<", HGT_SEA:"<<HGT_SEA<<std::endl;
            o << "TOWseconds:"<<TOWseconds<<std::endl;

        }

        /*
        // In header: <riegl/ridataspec.hpp>
template<typename it> -
struct hk_gps_hr {
enum @33 { id_main = = 10020, id_sub = = 0 };
// public data members
uint32_t TOWms; // GPS data, GPS Time of Week [ 1 ms].
int64_t ECEFX; // GPS data, ECEF X coordinate [ 1 mm].
int64_t ECEFY; // GPS data, ECEF Y coordinate [ 1 mm].
int64_t ECEFZ; // GPS data, ECEF Z coordinate [ 1 mm].
uint32_t POS_ACC; // GPS data, position accuracy estimate [ 1 mm].
int64_t LONG; // GPS data, longitude [1e-9deg].
int64_t LAT; // GPS data, latitude [1e-9deg]

int32_t HGT_ELL; // GPS data, height above ellipsoid [ 1 mm].
int32_t HGT_SEA; // GPS data, height above mean sea level [ 1 mm].
uint32_t HOR_ACC; // GPS data, horizontal accuracy estimate [ 1 mm].
uint32_t VERT_ACC; // GPS data, vertical accuracy estimate [ 1 mm].
uint8_t STATUS1; // GPS data, 0=no fix; 1=dead reckoning; 2=2D-fix; 3=3D-fix; 4=GPS+dead -
reckoning;5=Time only fix;255=undefined fix.
uint8_t STATUS2; // GPS data, 1 -.. gpsfixOK; 2 -.. diffSoln; 4 -.. wknSet; 8 -.. towSet.
uint8_t STATUS3; // GPS data, 0= no DGPS; 1=PR+PRR Correction; 2=PR+PRR+CP Correction; -
3=high accuracy PR+PRR+CP Correction.
int8_t LEAP_SEC; // GPS data, leap seconds (GPS-UTC) [ 1 s -].
uint32_t systime; // internal time in units of units.time_unit
uint8_t SYNC_STATUS; // PPS pulse and GPS data receive indicator bit 0: at least one -
valid gps information seen bit 1: valid gps information seen during last second bit 7: gps -
detected, gps data received within last second.
uint8_t SAT_NUM; // GPS data, number of satelites used.
uint16_t GEOM_DILUT; // GPS data, geometric dilution of position [0.01].
uint16_t TIME_DILUT; // GPS data, time dilution [0.01].
uint16_t VERT_DILUT; // GPS data, vertical dilution (1D) [0.01].
uint16_t HOR_DILUT; // GPS data, horizontal dilution (2D) [0.01].
uint16_t POS_DILUT; // GPS data, position dilution (3D) [0.01].
int64_t NORTHING; // GPS data, northing [ 1 mm].
int64_t EASTING; // GPS data, easting [ 1 mm].
int64_t HEIGHT; // GPS data, height [ 1 mm].
uint32_t FRAME_ANGLE; // GPS data, frame angle, unit see units.frame_circle_count.
uint32_t PPS_TIME; // GPS data, internal time of PPS trigger edge in units of -
units.time_unit.
uint32_t PPS_CNT; // GPS data, counter of PPS, increments with each detected PPS trigger -
edge.
uint32_t RCV_TIME; // GPS data, internal time of receiving the time information in units -
of units.time_unit.
uint32_t VALID_MASK; // GPS data, valid mask.
        */
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
