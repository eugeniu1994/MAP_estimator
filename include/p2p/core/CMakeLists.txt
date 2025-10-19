
add_library(core STATIC)
add_library(p2p::core ALIAS core)
target_sources(core PRIVATE Deskew.cpp Preprocessing.cpp Threshold.cpp VoxelHashMap.cpp Registration.cpp icp_util.cpp) 
target_link_libraries(core PUBLIC tsl::robin_map OpenMP::OpenMP_CXX TBB::tbb )
set_global_target_properties(core)


