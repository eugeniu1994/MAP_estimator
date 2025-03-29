

#include "DataHandler_vux.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mls");
    ros::NodeHandle nh;
    std::shared_ptr<DataHandler> dh = std::make_shared<DataHandler>(nh);

    dh->Subscribe();
    
    ros::spin();
    ros::shutdown();
    return 0;
}
