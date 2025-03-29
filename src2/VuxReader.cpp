#include "Vux_reader.hpp"
#include <boost/filesystem.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace vux;

bool VuxAdaptor::setUpReader(const std::string &folder_path)
{
    try
    {
        boost::filesystem::path dir(folder_path);

        if (!boost::filesystem::exists(dir) || !boost::filesystem::is_directory(dir))
        {
            std::cerr << "Invalid folder path: " << folder_path << std::endl;
            return false;
        }

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
            return false;
        }

        // Sort the files alphabetically
        std::sort(rxp_files.begin(), rxp_files.end());
        std::cout << "SetUp VuxAdaptor: " << rxp_files.size() << " rxp files" << std::endl;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Unknown exception occurred." << std::endl;
        return false;
    }

    return true;
}

bool VuxAdaptor::timeAlign(const double &mls_tod)
{
    bool rv = false;
    /*
    --iterate the rxp vector
    --compute the time for each file
    --plot that time vector
    --take the difference with our point
    --plot that difference
    --take the file from the min diff point
    --remove all the files untill that
    */

    std::vector<double> time_diff_vect, ind;
    for (const auto &file : rxp_files)
    {
        std::cout << "\nprocessing file:" << file << std::endl;
        std::shared_ptr<basic_rconnection> rc_front;
        std::shared_ptr<decoder_rxpmarker> dec_front;
        try
        {
            rc_front = basic_rconnection::create("file:" + file);
            rc_front->open(); // open front file
            dec_front = std::make_shared<decoder_rxpmarker>(*rc_front);
            buffer buf;

            line_read = false;
            while (dec_front->get(buf), !dec_front->eoi())
            {
                if (!ros::ok())
                    break;

                dispatch(buf.begin(), buf.end());

                if (line_read && cloud_line->size() > 0)
                {
                    double tod = cloud_line->points[0].time;

                    auto diff_ = abs(mls_tod - tod);
                    ind.push_back(time_diff_vect.size());
                    time_diff_vect.push_back(diff_);

                    std::cout << "mls tod:" << mls_tod << ", vux tod:" << tod<<" time diff:"<< diff_ << std::endl;
                    if (dec_front->eoi())
                    {
                        std::cout << "End of file dec_front eoi()" << std::endl;
                        break;
                    }
                    break;
                }
            }
        }
        catch (std::exception &e)
        {
            std::cerr << "Error processing file " << file << ": " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception processing file " << file << std::endl;
        }
    }
    std::cout<<"\n\nFinished to read all the vux files - see the time distance to each of them in the plot\n"<<std::endl;

    auto min_it = std::min_element(time_diff_vect.begin(), time_diff_vect.end());
    size_t min_index = std::distance(time_diff_vect.begin(), min_it);

    std::cout << "Minimum time distance: " << *min_it << std::endl;
    std::cout << "Index of minimum value: " << min_index << std::endl;

    std::cout << "Removing files: " << min_index << std::endl;
    if(min_index == 0)
        return true;

    for (int i = 0; i < min_index - 1; i++)
    {
        std::cout << " " << i;
        rxp_files.pop_front();
    }
    std::cout << "rxp_files:" << rxp_files.size() << std::endl;

    plt::scatter(ind, time_diff_vect, 5, {{"label", "time_diff"}, {"color", "blue"}});
    plt::xlabel("Files");
    plt::ylabel("Difference");
    plt::title("Time Alignment");
    plt::grid(true);
    plt::legend();
    plt::show();

    //if (!decoder_init)
    {
        std::cout<<"Reinitialize the decoder"<<std::endl;
        if (rxp_files.empty())
        {
            std::cout << "No more VUX data" << std::endl;
            return rv;
        }
        const auto file = rxp_files.front();
        rxp_files.pop_front();
        try
        {
            rc = basic_rconnection::create("file:" + file);
            rc->open();
            // decoder_rxpmarker dec(rc);
            dec = std::make_shared<decoder_rxpmarker>(*rc);
            decoder_init = true;
            std::cout << "decoder_init for " << file << std::endl;
            line_read = false;
            gnss_measurements.clear();
            gnss_corrected_tow.clear();

            rv = true;
        }
        catch (std::exception &e)
        {
            std::cerr << "Error processing file " << file << ": " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception processing file " << file << std::endl;
        }
    }

    return rv;
}

bool VuxAdaptor::next(pcl::PointCloud<VUX_PointType>::Ptr &next_line)
{
    bool rv = false;
    if (!decoder_init)
    {
        if (rxp_files.empty())
        {
            std::cout << "No more VUX data" << std::endl;
            return rv;
        }
        const auto file = rxp_files.front();
        rxp_files.pop_front();
        try
        {
            rc = basic_rconnection::create("file:" + file);
            rc->open();
            // decoder_rxpmarker dec(rc);
            dec = std::make_shared<decoder_rxpmarker>(*rc);
            decoder_init = true;
            std::cout << "decoder_init for " << file << std::endl;
        }
        catch (std::exception &e)
        {
            std::cerr << "Error processing file " << file << ": " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception processing file " << file << std::endl;
            return rv;
        }
    }

    buffer buf;
    while (dec->get(buf), !dec->eoi())
    {
        if (!ros::ok())
            break;

        dispatch(buf.begin(), buf.end());

        if (line_read)
        {
            // std::cout << "finished to read the line" << std::endl;
            next_line = cloud_line;
            line_read = false;
            rv = true;
            break;
        }
    }

    if (dec->eoi())
    {
        std::cout << "End of file dec eoi()" << std::endl;
        decoder_init = false;
    }

    return rv;
};

bool VuxAdaptor::nextGNSS(V3D &out, uint32_t &correected_gnss_tow)
{
    if (gnss_measurements.empty())
    {
        return false;
    }

    out = gnss_measurements.front();
    correected_gnss_tow = gnss_corrected_tow.front();

    gnss_measurements.pop_front();
    gnss_corrected_tow.pop_front();

    return true;
}