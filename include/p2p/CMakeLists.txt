
cmake_minimum_required(VERSION 3.16...3.26)
project(kiss_icp_cpp VERSION 0.2.9 LANGUAGES CXX)

message("Adding p2p ICP cmake.")

# Setup build options
option(USE_SYSTEM_EIGEN3 "Use system pre-installed Eigen" ON)
option(USE_SYSTEM_SOPHUS "Use system pre-installed Sophus" ON)
option(USE_SYSTEM_TSLMAP "Use system pre-installed tsl_robin" ON)
option(GTSAM_WITH_TBB "Use GTSAM_WITH_TBB" OFF)

# Set build type (repeat here for C++ only consumers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(3rdparty/find_dependencies.cmake)
include(cmake/CompilerOptions.cmake)

add_subdirectory(core)


