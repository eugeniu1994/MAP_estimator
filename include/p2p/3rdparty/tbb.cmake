
option(BUILD_SHARED_LIBS ON)
#option(BUILD_SHARED_LIBS OFF)
option(TBBMALLOC_BUILD ON)
option(TBB_EXAMPLES OFF)
option(TBB_STRICT OFF)
option(TBB_TEST OFF)

include(FetchContent)
FetchContent_Declare(tbb SYSTEM URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.8.0.tar.gz)

FetchContent_MakeAvailable(tbb)

message(STATUS " TBB is installing")

#if(${CMAKE_VERSION} VERSION_LESS 3.25)
  get_target_property(tbb_include_dirs tbb INTERFACE_INCLUDE_DIRECTORIES)
  set_target_properties(tbb PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${tbb_include_dirs}")
#endif()

message(STATUS "use TBB version: ${TBB_VERSION}")
