#include <iostream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "riegl/scanlib.hpp"
#include "mexWrapper.hpp"
#include "rxp_counter.hpp"

const std::vector<std::string> fields = {"time", "echo_type", "xyz", "echo_range", "amplitude", "reflectance",
                                         "deviation", "background_radiation", "zone_index", "facet", "segment", "return_num", "is_high_power",
                                         "is_pps_locked", "is_rising_edge", "is_sw", "gps_time", "gps_systime", "pps_count", "pps_systime", "time_unit"};

// Not used currenlty
int find_index(const std::string &str)
{
  auto it = std::find(fields.cbegin(), fields.cend(), str);
  return (it != fields.cend() ? (it - fields.cbegin()) : -1);
}

/*
 * scanlib::pointcloud derived class. Reads point data and writes it to matlab data arrays.
 * \param &os used for debugging
 * \param &arrays vector of matlab data array pointers that the data is copied into
 * \param is_pps_synced setting this to true will exclude data that isn't pps synced
 */
class Rxp2Mat : public scanlib::pointcloud
{
public:
  Rxp2Mat(std::ostringstream &os, std::vector<matlab::data::Array *> &arrays, unsigned start, unsigned end, bool is_pps_synced = false) : scanlib::pointcloud(is_pps_synced), os_(os), arrays_(arrays), start_(start), end_(end) {}

  std::vector<matlab::data::Array *> &arrays_;

private:
  unsigned start_ = 0;
  unsigned end_ = 0;
  unsigned int i = 0;
  unsigned int j = 0;
  unsigned ntarget = 0;
  std::ostringstream &os_;

protected:
  // Callback functions:

  void on_units_4(const scanlib::units_4<iterator_type> &arg)
  {
    pointcloud::on_units_4(arg);
    (*arrays_[fields.size() - 1])[0] = (float)arg.time_unit;
  }

  // Called for each point that is decoded and transformed. Always on last in targets.
  void on_echo_transformed(echo_type echo)
  {
    pointcloud::on_echo_transformed(echo);
    scanlib::target t = targets[target_count - 1];
    if (ntarget >= start_ && ntarget < end_)
    {
      (*arrays_[0])[i] = t.time;
      (*arrays_[1])[i] = (unsigned int)echo;
      (*arrays_[2])[i][0] = t.vertex[0];
      (*arrays_[2])[i][1] = t.vertex[1];
      (*arrays_[2])[i][2] = t.vertex[2];
      (*arrays_[3])[i] = t.echo_range;
      (*arrays_[4])[i] = t.amplitude;
      (*arrays_[5])[i] = t.reflectance;
      (*arrays_[6])[i] = t.deviation;
      (*arrays_[7])[i] = t.background_radiation;
      (*arrays_[8])[i] = t.zone_index;
      (*arrays_[9])[i] = t.facet;
      (*arrays_[10])[i] = t.segment;
      (*arrays_[11])[i] = (unsigned short)target_count;
      (*arrays_[12])[i] = t.is_high_power;
      (*arrays_[13])[i] = t.is_pps_locked;
      (*arrays_[14])[i] = t.is_rising_edge;
      (*arrays_[15])[i] = t.is_sw;
      i++;
    }
    ntarget++;
  }

  // Decodes the GPS synchronization status and the PPS counter.
  // This function is only called on scanners WITHOUT higher resolution (hr) system time.
  // For scanners WITH higher resolution system time the function on_hk_gps_hr is called.
  void on_hk_gps_ts_status_dop_ucs(const scanlib::hk_gps_ts_status_dop_ucs<iterator_type> &arg)
  {
    pointcloud::on_hk_gps_ts_status_dop_ucs(arg);
    (*arrays_[16])[j] = (uint32_t)arg.TOWms;
    (*arrays_[17])[j] = (uint32_t)arg.systime;
    (*arrays_[18])[j] = (uint32_t)arg.PPS_CNT;
    (*arrays_[19])[j] = (uint32_t)arg.PPS_TIME;
    j++;
  }

  // Decodes the GPS synchronization status and the PPS counter.
  // This function is only called on scanners WITH higher resolution (hr) system time.
  // For scanners WITHOUT higher resolution (hr) system time the function on_hk_gps_ts_status_dop_ucs is called.
  void on_hk_gps_hr(const scanlib::hk_gps_hr<iterator_type> &arg)
  {
    pointcloud::on_hk_gps_hr(arg);
    (*arrays_[16])[j] = (uint32_t)arg.TOWms;
    (*arrays_[17])[j] = (uint32_t)arg.systime;
    (*arrays_[18])[j] = (uint32_t)arg.PPS_CNT;
    (*arrays_[19])[j] = (uint32_t)arg.PPS_TIME;
    j++;
  }
};

class MexFunction : public matlab::mex::Function
{
public:
  std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
  matlab::data::ArrayFactory factory;
  std::ostringstream stream;

  void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs)
  {
    std::chrono::steady_clock::time_point start_time_total = std::chrono::steady_clock::now();

    if (inputs[0].getType() != matlab::data::ArrayType::MATLAB_STRING)
      raiseError("The first input argument has invalid type. Expected Matlab string.");

    if (inputs[0].getNumberOfElements() != 1)
      raiseError("The first input argument has invalid size. Expected (1,1).");

    std::string file = std::string(inputs[0][0]);

    if (inputs[1].getType() != matlab::data::ArrayType::STRUCT)
      raiseError("Second input argument has invalid type. Expected Matlab struct.");

    if (inputs[1].getNumberOfElements() != 1)
      raiseError("Second inputs argument has invalid size. Expected (1,1).");

    matlab::data::StructArray optsArray = inputs[1];

    auto fieldNames = optsArray.getFieldNames();
    matlab::data::Struct opts = optsArray[0];

    unsigned start = 0;
    unsigned end = UINT32_MAX;
    unsigned verbose = 0;

    for (auto it = fieldNames.begin(); it != fieldNames.end(); it++)
    {
      std::string key = *it;
      matlab::data::Array value = opts[*it];
      matlab::data::ArrayType type = value.getType();
      std::size_t numel = value.getNumberOfElements();

      if (key == "Start")
      {
        if (type != matlab::data::ArrayType::UINT32)
          raiseError("Option " + key + " has invalid data type: expected uint32.");

        if (numel != 1)
          raiseError("Option " + key + " has invalid size. Expected scalar.");

        matlab::data::TypedArray<uint32_t> matStart = value;
        start = matStart[0];
      }

      if (key == "End")
      {
        if (type != matlab::data::ArrayType::UINT32)
          raiseError("Option " + key + " has invalid data type: expected uint32.");

        if (numel != 1)
          raiseError("Option " + key + " has invalid size. Expected scalar.");

        matlab::data::TypedArray<uint32_t> matEnd = value;
        end = matEnd[0];
      }

      if (key == "Verbose")
      {
        if (type != matlab::data::ArrayType::UINT32)
          raiseError("Option " + key + " has invalid data type: expected uint32.");

        if (numel != 1)
          raiseError("Option " + key + " has invalid size. Expected scalar.");

        matlab::data::TypedArray<uint32_t> matVerb = value;
        verbose = matVerb[0];
      }
    }

    RxpCounter point_counter(stream);
    try
    {
      std::shared_ptr<scanlib::basic_rconnection> connection;
      scanlib::buffer buffer;

      // Read thru once to get number of points
      connection = scanlib::basic_rconnection::create("file:" + file);
      connection->open();
      scanlib::decoder_rxpmarker decoder(connection);
      std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

      if (verbose > 0)
      {
        stream << "Counting number of points ..." << std::endl;
        show(stream);
      }

      for (decoder.get(buffer); !decoder.eoi(); decoder.get(buffer))
      {
        point_counter.dispatch(buffer.begin(), buffer.end());
      }
      connection->close();
      std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
      std::chrono::duration<double> duration = end_time - start_time;

      if (verbose > 0)
      {
        stream << "Number of points in file: " << point_counter.point_count << std::endl;
        stream << "First pass took: " << duration.count() << " seconds" << std::endl;
        show(stream);
      }

      unsigned NUM_POINTS = point_counter.point_count;
      const unsigned PPS_COUNT = point_counter.gps_count;

      if (start > end)
      {
        stream << "Start index: " << start << " cannot be greater than end index: " << end << "!" << std::endl;
        outputs[0] = std::move(factory.createArray<int>({1}));
        raiseError(stream.str());
        return;
      }

      start = std::min(std::max(start, 0U), NUM_POINTS);
      end = std::max(0U, std::min(NUM_POINTS, end));

      NUM_POINTS = end - start;

      if (verbose > 0)
      {
        stream << "Start: " << start << std::endl;
        stream << "End: " << end << std::endl;
        stream << "Reading a total of: " << NUM_POINTS << " points." << std::endl;
        show(stream);
      }

      // Create new connection and decoder for actual reading
      connection = scanlib::basic_rconnection::create("file:" + file);
      connection->open();
      scanlib::decoder_rxpmarker decoder_read(connection);

      // Create data arrays and struct

      matlab::data::ArrayDimensions dims = {NUM_POINTS, 1};

      matlab::data::StructArray mat_struct = factory.createStructArray({1, 1}, fields);
      matlab::data::TypedArray<double> time = factory.createArray<double>(dims);
      matlab::data::TypedArray<unsigned int> echo_type = factory.createArray<unsigned int>(dims);
      matlab::data::TypedArray<float> xyz = factory.createArray<float>({NUM_POINTS, 3});
      matlab::data::TypedArray<double> echo_range = factory.createArray<double>(dims);
      matlab::data::TypedArray<float> amplitude = factory.createArray<float>(dims);
      matlab::data::TypedArray<float> reflectance = factory.createArray<float>(dims);
      matlab::data::TypedArray<float> deviation = factory.createArray<float>(dims);
      matlab::data::TypedArray<float> radiation = factory.createArray<float>(dims);
      matlab::data::TypedArray<unsigned short> zone_index = factory.createArray<unsigned short>(dims);
      matlab::data::TypedArray<unsigned int> facet = factory.createArray<unsigned int>(dims);
      matlab::data::TypedArray<unsigned int> segment = factory.createArray<unsigned int>(dims);
      matlab::data::TypedArray<unsigned short> return_num = factory.createArray<unsigned short>(dims);
      matlab::data::TypedArray<bool> is_high_pow = factory.createArray<bool>(dims);
      matlab::data::TypedArray<bool> is_pps_locked = factory.createArray<bool>(dims);
      matlab::data::TypedArray<bool> is_rising_edge = factory.createArray<bool>(dims);
      matlab::data::TypedArray<bool> is_sw = factory.createArray<bool>(dims);
      matlab::data::TypedArray<unsigned int> gps_time = factory.createArray<unsigned int>({PPS_COUNT, 1});
      matlab::data::TypedArray<unsigned int> gps_systime = factory.createArray<unsigned int>({PPS_COUNT, 1});
      matlab::data::TypedArray<unsigned int> pps_count = factory.createArray<unsigned int>({PPS_COUNT, 1});
      matlab::data::TypedArray<unsigned int> pps_systime = factory.createArray<unsigned int>({PPS_COUNT, 1});

      matlab::data::TypedArray<float> time_unit = factory.createArray<float>({1});

      std::vector<matlab::data::Array *> arrays = {&time, &echo_type, &xyz, &echo_range,
                                                   &amplitude, &reflectance, &deviation,
                                                   &radiation, &zone_index, &facet,
                                                   &segment, &return_num, &is_high_pow, &is_pps_locked,
                                                   &is_rising_edge, &is_sw, &gps_time,
                                                   &gps_systime, &pps_count, &pps_systime,
                                                   &time_unit};

      if (arrays.size() != fields.size())
      {
        stream << "Number of arrays and fields do not match! Contact maintainer." << std::endl;
        stream << "Field size=" << fields.size() << " Number of arrays=" << arrays.size() << std::endl;
        show(stream);
        raiseError(stream.str());
      }

      if (verbose > 0)
      {
        stream << "Created data arrays successfully" << std::endl;
        show(stream);
      }

      Rxp2Mat rxp2mat(stream, arrays, start, end, false);

      if (verbose > 0)
      {
        stream << "Starting second pass to write data to Matlab arrays ..." << std::endl;
        show(stream);
      }

      start_time = std::chrono::steady_clock::now();
      // Read data and assign to output
      for (decoder_read.get(buffer); !decoder_read.eoi(); decoder_read.get(buffer))
      {
        rxp2mat.dispatch(buffer.begin(), buffer.end());
      }
      connection->close();

      if (verbose > 0)
      {
        end_time = std::chrono::steady_clock::now();
        duration = end_time - start_time;
        stream << "Finished writing point data to arrays, took: " << duration.count() << " seconds" << std::endl;
        stream << "Copying arrays to matlab struct ..." << std::endl;
        show(stream);
      }

      // Copy matlab arrays into the struct.
      for (int i = 0; i < fields.size(); i++)
      {
        mat_struct[0][fields[i]] = *arrays[i];
      }
      outputs[0] = std::move(mat_struct);
    }
    catch (const std::exception &e)
    {
      raiseError(e.what());
    }
    if (verbose > 0)
    {
      std::chrono::duration<double> total_dur = std::chrono::steady_clock::now() - start_time_total;
      stream << "Took a total of: " << total_dur.count() << " seconds." << std::endl;
      show(stream);
    }
  }

  // A helper function for raising errors inside matlab
  void raiseError(const std::string msg)
  {
    matlabPtr->feval("error", 0, {factory.createScalar(msg)});
  }

  /*
   * Some compilers don't output std::cout to Matlab Command Window, see:
   * https://se.mathworks.com/help/matlab/matlab_external/displaying-output-in-matlab-command-window.html
   */
  void show(std::ostringstream &stream)
  {
    // Pass stream content to MATLAB fprintf function
    matlabPtr->feval(u"fprintf", 0,
                     std::vector<matlab::data::Array>({factory.createScalar(stream.str())}));
    // Clear stream buffer
    stream.str("");
  }
};