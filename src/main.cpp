

#include "DataHandler.hpp"

int main(int argc, char **argv)
{   
    #ifdef MP_EN
        // NUM_THREADS = omp_get_max_threads();
        // NUM_THREADS = std::max(NUM_THREADS, 16);
        std::cout<<"NUM_THREADS:"<<NUM_THREADS<<std::endl;
        omp_set_num_threads(NUM_THREADS);
        omp_set_dynamic(0);  // prevent OpenMP from changing thread count
    #endif
    tbb::global_control tbb_limit(tbb::global_control::max_allowed_parallelism, NUM_THREADS);

    ros::init(argc, argv, "map_lio");
    ros::NodeHandle nh;
    std::shared_ptr<DataHandler> dh = std::make_shared<DataHandler>(nh);

    dh->Subscribe();
    
    ros::spin();
    ros::shutdown();
    return 0;
}
