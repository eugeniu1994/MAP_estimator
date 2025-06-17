cmake_minimum_required(VERSION 3.0.2)
project(s_mls)

SET(CMAKE_BUILD_TYPE "Release") #Release

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread ") #

#set( CMAKE_CXX_FLAGS " -O3 " ) #this will make the execution fast and compilation slow  

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
message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message("core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message("core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()


find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED )


find_package(TBB REQUIRED)
message(STATUS "TBB version: ${TBB_VERSION}")


find_package(GTSAM REQUIRED)     #if GTSAM is used only
#find_package(OpenCV 3 REQUIRED)  #only for range image 
find_package(PythonLibs 3.6 REQUIRED)
#find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h") #for plots
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS 
  "matplotlibcpp.h" 
  PATHS 
    ${CMAKE_SOURCE_DIR}/src/s_mls/include
    ${CMAKE_SOURCE_DIR}/include
    /home/eugeniu/catkin_ws/src/s_mls/include # Explicit full path
)
if (NOT MATPLOTLIB_CPP_INCLUDE_DIRS)
  message(FATAL_ERROR "Could not find matplotlibcpp.h in the specified directories.")
else()
  message(STATUS "Found matplotlibcpp.h in: ${MATPLOTLIB_CPP_INCLUDE_DIRS}")
endif()

find_package(GeographicLib REQUIRED) #required for GNSS stuff

find_package(libLAS REQUIRED) #required to read las files-----------------------------------
message(libLAS_FOUND: ${libLAS_FOUND})
message(STATUS "libLAS_FOUND is set to ${libLAS_FOUND}")

# Manually specify paths for libLAS
set(LIBLAS_INCLUDE_DIRS "/usr/local/include")
set(LIBLAS_LIBRARY_DIRS "/usr/local/lib")
set(LIBLAS_INCLUDE_DIR "/usr/local/include/liblas")
set(LIBLAS_LIBRARY_DIR "/usr/local/lib")
set(LIBLAS_LIBRARIES "las")

find_package(Ceres REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${MATPLOTLIB_CPP_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  #${OpenCV_INCLUDE_DIRS}
  ${TBB_INCLUDE_DIRS}
  ${LIBLAS_INCLUDE_DIRS}
  include

  ${CERES_INCLUDE_DIRS}
)

link_directories(
  ${GTSAM_LIBRARY_DIRS}
  #${OpenCV_LIBRARY_DIRS}
  ${LIBLAS_LIBRARY_DIRS}

  ${CERES_LIBRARY_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs 
  DEPENDS EIGEN3 GTSAM PCL Ceres
  INCLUDE_DIRS  
)

add_definitions(-DWITHOUT_NUMPY) #for matplotlib
#add_definitions(-DSAVE_DATA) #uncomment this to enable saving the data - note it will slow down the execution time

set(USE_EKF TRUE)
set(USE_STATIC_KDTREE TRUE) # or FALSE to disable

#add_definitions(-DADAPTIVE_KERNEL=1)
add_definitions(-DUSE_ALS)

#------------------------------------------------------------------------


SET(TARGET_LINK_LIBS ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${PCL_LIBRARIES} Sophus ${TBB_LIBRARIES} TBB::tbb 
${GCC_COVERAGE_LINK_FLAGS} ${GeographicLib_LIBRARIES} ${LIBLAS_LIBRARIES} OpenMP::OpenMP_CXX ) 

SET(MAIN_SOURCE_FILES src/IMU.cpp src/GNSS.cpp src/utils.cpp src/estimators/Estimator.cpp src/DataHandler.cpp)
SET(MAIN_SOURCE_FILES3 src/IMU.cpp src/GNSS.cpp src/utils.cpp src/estimators/Estimator.cpp)

if(USE_STATIC_KDTREE )
    message("Use the static KDTREE")
    add_definitions(-DUSE_STATIC_KDTREE=1)
else()
    message("Use the iKDTREE")
    add_definitions(-DUSE_STATIC_KDTREE=0)
    message(STATUS "Including ikd_Tree.cpp")
    set(MAIN_SOURCE_FILES ${MAIN_SOURCE_FILES} include/ikd-Tree/ikd_Tree.cpp)
    set(MAIN_SOURCE_FILES3 ${MAIN_SOURCE_FILES3} include/ikd-Tree/ikd_Tree.cpp)
endif()


if(USE_EKF)
    message("Build EKF registration.")
    add_definitions(-DUSE_EKF=1)
    #include the RIEKF part
    set(MAIN_SOURCE_FILES ${MAIN_SOURCE_FILES} src/estimators/RIEKF.cpp src/estimators/ALS_ekf.cpp)
    set(MAIN_SOURCE_FILES3 ${MAIN_SOURCE_FILES3} src/estimators/RIEKF.cpp src/estimators/ALS_ekf.cpp)
else()
    message("Build ICP registration.")
    #add the icp estimator
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/p2p/)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/p2p ${CMAKE_CURRENT_BINARY_DIR}/p2p)
    endif()
    set(TARGET_LINK_LIBS ${TARGET_LINK_LIBS} p2p::core)
    set(TARGET_LINK_LIBS ${TARGET_LINK_LIBS} gtsam)
    set(MAIN_SOURCE_FILES ${MAIN_SOURCE_FILES} src/estimators/ICP.cpp src/Graph.cpp src/estimators/ALS_icp.cpp)
    set(MAIN_SOURCE_FILES3 ${MAIN_SOURCE_FILES3} src/estimators/ICP.cpp src/Graph.cpp src/estimators/ALS_icp.cpp)
endif()

# message(STATUS "MAIN_SOURCE_FILES: ${MAIN_SOURCE_FILES}")
# add_executable(mls_mapping src/main.cpp ${MAIN_SOURCE_FILES} src/DataHandler_arvo.cpp) 
# target_link_libraries(mls_mapping ${TARGET_LINK_LIBS} ) 
# add_dependencies(mls_mapping ${catkin_EXPORTED_TARGETS})
# target_include_directories(mls_mapping PRIVATE ${PYTHON_INCLUDE_DIRS} ${MATPLOTLIB_CPP_INCLUDE_DIRS})




#vux data publisher 
#-----------------------------------------------------------------------------------
set(RIEGL_LIBRARY_DIR "/home/eugeniu/Apps/rivlib-2_7_2-x86_64-linux-gcc9")
set(CMAKE_PREFIX_PATH "/home/eugeniu/Apps/rivlib-2_7_2-x86_64-linux-gcc9/cmake")
include_directories(${RIEGL_LIBRARY_DIR}/include)
link_directories(${RIEGL_LIBRARY_DIR}/lib)
find_package(RiVLib COMPONENTS scanlib REQUIRED)
message(STATUS "RIEGL_LIBRARY_DIR: ${RIEGL_LIBRARY_DIR}")
SET(TARGET_LINK_LIBS2 ${catkin_LIBRARIES} RiVLib::scanlib ${PYTHON_LIBRARIES} ${PCL_LIBRARIES} Sophus ${TBB_LIBRARIES} TBB::tbb) 
SET(MAIN_SOURCE_FILES2 src2/Vux_publisher.cpp src/utils.cpp )

#add_executable(vux_publisher ${MAIN_SOURCE_FILES2})
#target_link_libraries(vux_publisher  ${TARGET_LINK_LIBS2})



#the code with vux
#-----------------------------------------------------------------------------------
SET(TARGET_LINK_LIBS3 ${TARGET_LINK_LIBS} RiVLib::scanlib gtsam)  
set(SOURCE_FILES_VUX ${MAIN_SOURCE_FILES3} src2/DataHandler_vux.cpp src2/clean_DataHandler_vux.cpp src2/VuxReader.cpp)

# set(SOURCE_FILES_VUX ${MAIN_SOURCE_FILES3} src3/test.cpp src2/clean_DataHandler_vux.cpp src2/VuxReader.cpp)



#for tsl map
# if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/p2p/)
#     add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/include/p2p ${CMAKE_CURRENT_BINARY_DIR}/p2p)
# endif()
# set(TARGET_LINK_LIBS3 ${TARGET_LINK_LIBS3} tsl::robin_map ) #p2p::core
# set(SOURCE_FILES_VUX ${MAIN_SOURCE_FILES3} src2/VoxelHash.cpp) #remove the voxel hash from here


add_executable(vux_main src2/main.cpp ${SOURCE_FILES_VUX})
target_link_libraries(vux_main ${TARGET_LINK_LIBS3} ${CERES_LIBRARIES})


#------------------------
#test tightly coupled
set(SOURCE_FILES_TEST ${MAIN_SOURCE_FILES3} src3/DataHandler_vux.cpp src2/clean_DataHandler_vux.cpp src2/VuxReader.cpp)
add_executable(test_fusion src2/main.cpp ${SOURCE_FILES_TEST})
target_link_libraries(test_fusion ${TARGET_LINK_LIBS3} ${CERES_LIBRARIES})