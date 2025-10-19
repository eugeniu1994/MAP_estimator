
#include "DataHandler.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mls");
    ros::NodeHandle nh;
    std::shared_ptr<DataHandler> dh = std::make_shared<DataHandler>(nh);


    int tmp_lidar_type = 0;
    nh.param<int>("preprocess/lidar_type", tmp_lidar_type, 0);
    //TODO - do it better here -------------------
    if(tmp_lidar_type == 1){ //its arvo
        //dh->BagHandler_Arvo();
    }else{
        dh->BagHandler();
    }
    
    
    ros::spin();
    ros::shutdown();
    return 0;
}