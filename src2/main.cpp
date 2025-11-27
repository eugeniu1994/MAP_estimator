

#include "DataHandler_vux.hpp"

int main(int argc, char **argv)
{

    #ifdef _OPENMP
        omp_set_num_threads(NUM_THREADS);
        omp_set_dynamic(0);  // prevent OpenMP from changing thread count
    #endif
    // TBB 
    // Applies globally for all TBB algorithms
    
    tbb::global_control tbb_limit(tbb::global_control::max_allowed_parallelism, NUM_THREADS);
    //tbb::global_control tbb_limit(tbb::global_control::max_allowed_parallelism, 1);



    ros::init(argc, argv, "mls");
    ros::NodeHandle nh;
    std::shared_ptr<DataHandler> dh = std::make_shared<DataHandler>(nh);

    dh->Subscribe();
    
    ros::spin();
    ros::shutdown();
    return 0;
}
