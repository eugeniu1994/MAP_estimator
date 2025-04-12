#ifndef USE_VUX_READER_H1
#define USE_VUX_READER_H1

#define PCL_NO_PRECOMPILE

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <ctime>

#include "utils.h"

#include <riegl/scanlib.hpp>

namespace vux
{

    using namespace scanlib;
    using namespace std;

    class VuxAdaptor : public pointcloud
    {
        double first_measurement_time_tow = 378408;
        ostream &o;
        string type_id;
        string serial;

        bool got_first_gnss = false;
        uint32_t TOWseconds = 0;
        bool line_read, gnss_read;
        pcl::PointCloud<VUX_PointType>::Ptr cloud_line;

        bool decoder_init = false;
        std::deque<std::string> rxp_files;
        std::deque<V3D> gnss_measurements;
        std::deque<uint32_t> gnss_corrected_tow;

        float time_unit = 0, time_unit_hi_prec = 0;

    public:
        unsigned long line;
        double max_range;
        std::shared_ptr<basic_rconnection> rc;
        std::shared_ptr<decoder_rxpmarker> dec;

    public:
        VuxAdaptor(ostream &o_, double max_range_ = 25) : pointcloud(true) // set this to true if you need gps aligned timing
                                  ,
                                  o(o_), line(0)
        {
            o.precision(10);
            max_range = max_range_;
            line_read = false;
            gnss_read = false;
            cloud_line = pcl::PointCloud<VUX_PointType>::Ptr(new pcl::PointCloud<VUX_PointType>());
        }

    protected:
        void on_echo_transformed(echo_type echo)
        {
            // std::cout<<"on_echo_transformed"<<std::endl;
            //  TODO filter points by echo
            if ((pointcloud::first == echo || pointcloud::single == echo) && got_first_gnss)
            //if (pointcloud::none != echo && got_first_gnss)
            {
                // std::cout << "on_echo_transformed is being called--------------------------------------" << std::endl;
                //  targets is a member std::vector that contains all
                //  echoes seen so far, i.e. the current echo is always
                //  indexed by target_count-1.
                target &t(targets[target_count - 1]);

                // transform to polar coordinates
                double range = std::sqrt(t.vertex[0] * t.vertex[0] + t.vertex[1] * t.vertex[1] + t.vertex[2] * t.vertex[2]);
                // if (range > numeric_limits<double>::epsilon())
                
                //std::cout<<"t.echo_range:"<<t.echo_range<<", range:"<<range<<std::endl;
                if (range > 1 && range < max_range)
                {
                    VUX_PointType point;
                    point.x = t.vertex[0];
                    point.y = t.vertex[1];
                    point.z = t.vertex[2];

                    point.range = range;
                    //point.echo_range = t.echo_range;
                    point.time = t.time;
                    // point.amplitude = t.amplitude;
                    point.reflectance = t.reflectance;
                    // point.deviation = t.deviation;
                    // point.segment = t.segment;
                    // point.time_sorg = t.time_sorg;
                    //  point.is_line_end = t.is_line_end;
                    //  if (pointcloud::first == echo)
                    //  {
                    //      point.single_echo = 0;
                    //  }
                    //  else if (pointcloud::single == echo)
                    //  {
                    //      point.single_echo = 1;
                    //  }
                    //  else
                    //  {
                    //      point.single_echo = 100;
                    //  }

                    cloud_line->points.push_back(point);
                }
            }
        }

        void on_line_start_up(const line_start_up<iterator_type> &arg)
        {
            line++;
            // o << "line start up: " << line << endl;
            pointcloud::on_line_start_up(arg);
            cloud_line = pcl::PointCloud<VUX_PointType>::Ptr(new pcl::PointCloud<VUX_PointType>());
        }

        void on_line_stop(const line_stop<iterator_type> &arg)
        {
            // o << "line stop: " << line << ", cloud_line points:" << cloud_line->size() << endl;
            pointcloud::on_line_stop(arg);
            line_read = true;
        }

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

        double tow_to_time_of_day(double tow)
        {
            int SECONDS_PER_DAY = 86400;       // Number of seconds in a day
            return fmod(tow, SECONDS_PER_DAY); // Use modulo to get time of day
        }

        // Function to convert Time of Day back to Time of Week (TOW)
        double time_of_day_to_tow(double time_of_day, int day_of_week)
        {
            int SECONDS_PER_DAY = 86400;                          // Number of seconds in a day
            return time_of_day + (day_of_week * SECONDS_PER_DAY); // Add seconds for the current day
        }


        // void on_pps_sync_hr(const pps_sync_hr<iterator_type> &arg)
        // {
        //     pointcloud::on_pps_sync_hr(arg);
        //     //std::cout << " \non_pps_sync_hr " << std::endl;

        //     //uint64_t systime = arg.systime; // internal time of external PPS pulse in units of -units.time_unit_hi_prec
        //     uint64_t pps = arg.pps;         // absolute time of pps pulse since begin of week or start of day in usec

        //     uint64_t pps_seconds = pps / 1000000;
        //     // std::cout << " systime:" << systime << ", pps:" << pps << std::endl;
        //     std::cout << " pps_seconds:" << pps_seconds << " s since begin of week or start of day" << std::endl;
        // }

        void on_hk_gps_hr(const hk_gps_hr<iterator_type> &arg)
        {
            pointcloud::on_hk_gps_hr(arg);

            std::string sync_status = "not synchronized";
            if (arg.SYNC_STATUS == 1)
                sync_status = "lost synchronization";
            else if (arg.SYNC_STATUS == 3)
                sync_status = "correctly synchronized";

            uint32_t PPS_CNT = arg.PPS_CNT; // GPS data, counter of PPS, increments with each detected PPS trigger -edge.

            //o << "\non_hk_gps_hr gps hr: " << "sync_status=" << sync_status << "\tPPS_count=" << PPS_CNT << endl;

            int64_t LONG = (int64_t)arg.LONG;       // GPS data, longitude [1e-9deg].
            int64_t LAT = (int64_t)arg.LAT;         // GPS data, latitude [1e-9deg].
            int32_t HGT_ELL = (int32_t)arg.HGT_ELL; // GPS data, height above ellipsoid [ 1 mm].
            int32_t HGT_SEA = (int32_t)arg.HGT_SEA; // GPS data, height above mean sea level [ 1 mm].
            //std::cout<<"LONG:"<<LONG<<", LAT:"<<LAT<<", HGT_ELL:"<<HGT_ELL<<", HGT_SEA:"<<HGT_SEA<<std::endl;

            // Convert nanodegrees to degrees
            double lat_deg = LAT * 1e-9;
            double lon_deg = LONG * 1e-9;
            double height_m = HGT_ELL/1000;

            auto gnss_measurement = V3D(lat_deg, lon_deg, height_m);
            //std::cout<<"gnss_measurement:"<<gnss_measurement.transpose()<<std::endl;

            uint32_t TOWms = (uint32_t)arg.TOWms; // GPS data, GPS Time of Week [ 1 ms].
            TOWseconds = TOWms / 1000; // Convert milliseconds to seconds

            if (!got_first_gnss)
            {
                got_first_gnss = true;
                uint8_t STATUS1 = (uint8_t)arg.STATUS1; // GPS data, 0=no fix; 1=dead reckoning; 2=2D-fix; 3=3D-fix; 4=GPS+dead-reckoning;5=Time only fix;255=undefined fix.

                int8_t LEAP_SEC = arg.LEAP_SEC;           // GPS data, leap seconds (GPS-UTC) [ 1 s -].
                uint32_t systime = (uint32_t)arg.systime; // internal time in units of units.time_unit

                uint32_t PPS_TIME = arg.PPS_TIME; // GPS data, internal time of PPS trigger edge in units of -units.time_unit.
                auto PPS_TIME_time_unit = PPS_TIME * time_unit;
                

                // o << "PPS_TIME:" << PPS_TIME << ", systime:" << systime << ", TOWms:" << TOWms  << std::endl;
                // o << "LONG:" << LONG << ", LAT:" << LAT << ", HGT_ELL:" << HGT_ELL << ", HGT_SEA:" << HGT_SEA << std::endl;
                o << "TOWseconds:" << TOWseconds<<" PPS_TIME_time_unit:"<<PPS_TIME_time_unit << std::endl;

                //o << "first_measurement_time_tow:" << first_measurement_time_tow << std::endl;
                // std::cout << "time_unit:" << time_unit << std::endl;

                // Convert TOW to Time of Day
                // double time_of_day = tow_to_time_of_day(first_measurement_time_tow);
                // std::cout << "\n\nTime of Day (seconds since midnight): " << time_of_day << std::endl;

                // // Convert Time of Day back to TOW
                // int day_of_week = 4; // Thursday
                // double converted_tow = time_of_day_to_tow(time_of_day, day_of_week);
                // std::cout << "Converted Time of Week (TOW): " << converted_tow <<"\n\n"<< std::endl;
            }

            gnss_measurements.push_back(gnss_measurement);
            gnss_corrected_tow.push_back(TOWseconds);
            gnss_read = true;
        }

        // This function gets called when a new header is available.
        void on_header(const header<iterator_type> &arg)
        {
            pointcloud::on_header(arg);
            // this->type_id = arg.type_id;
            // this->serial = arg.serial;
            // o << "type id: " << this->type_id << endl;
            // o << "serial: " << this->serial << endl;
        }

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
            
            //o << "epoch_exttime: " << arg.epoch_exttime << endl; // epoch of exttime in YYYYMMDDTHH:MM:SS or UNKNOWN
            //o << "clock_source: " << arg.clock_source << endl;   // UNKNOWN, RTC, GPS or UTC

            time_unit = (float)arg.time_unit;                           // duration of 1 LSB of time in [sec]
            //o << " time_unit = " << time_unit << " [sec]" << std::endl; // duration of 1 LSB of time in [sec]
        }

        //----------------------------------------------------------

    public:
        bool setUpReader(const std::string &folder_path);
        bool next(pcl::PointCloud<VUX_PointType>::Ptr &next_line);
        bool nextGNSS(V3D &out, uint32_t &correected_gnss_tow);
        bool timeAlign(const double &mls);
    };

};
#endif