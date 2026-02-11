cmake_minimum_required(VERSION 3.0.2)
project(map_lio)

SET(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ") 

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")  #this will make the execution fast and compilation slow 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(GCC_COVERAGE_LINK_FLAGS "-lstdc++fs")


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rosbag
  rospy
  sensor_msgs
  std_msgs
  tf
  gps_common
  eigen_conversions
)

find_package(OpenMP REQUIRED) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
add_definitions(-DMP_EN)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED )
find_package(GeographicLib REQUIRED) #required for GNSS stuff


#todo remove TBB dependency 
find_package(TBB REQUIRED)
message(STATUS "TBB version: ${TBB_VERSION}")


option(USE_ALS "Enable ALS" OFF)

SET(MAIN_SOURCE_FILES src/IMU.cpp src/GNSS.cpp src/utils.cpp src/estimators/Estimator.cpp 
 src/DataHandler.cpp ) 

if(USE_ALS)
    SET(MAIN_SOURCE_FILES ${MAIN_SOURCE_FILES} src/estimators/ALS.cpp ) 
    add_definitions(-DUSE_ALS)
    find_package(libLAS REQUIRED) #required to read las files-----------------------------------
    message(libLAS_FOUND: ${libLAS_FOUND})
    message(STATUS "libLAS_FOUND is set to ${libLAS_FOUND}")

    # Manually specify paths for libLAS
    set(LIBLAS_INCLUDE_DIRS "/usr/local/include")
    set(LIBLAS_LIBRARY_DIRS "/usr/local/lib")
    set(LIBLAS_INCLUDE_DIR "/usr/local/include/liblas")
    set(LIBLAS_LIBRARY_DIR "/usr/local/lib")
    set(LIBLAS_LIBRARIES "las")

    link_directories(${LIBLAS_LIBRARY_DIRS})
endif()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${LIBLAS_INCLUDE_DIRS}
  include
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs 
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS  
)

SET(TARGET_LINK_LIBS ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${PCL_LIBRARIES} Sophus ${TBB_LIBRARIES} TBB::tbb 
${GCC_COVERAGE_LINK_FLAGS} ${GeographicLib_LIBRARIES} ${LIBLAS_LIBRARIES} OpenMP::OpenMP_CXX ) 

add_executable(estimate_map src/main.cpp ${MAIN_SOURCE_FILES} ) 
target_link_libraries(estimate_map ${TARGET_LINK_LIBS}) 

