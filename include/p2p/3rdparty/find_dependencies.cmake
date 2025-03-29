
if(USE_SYSTEM_TSLMAP)
  find_package(tsl-robin-map QUIET NO_MODULE)
endif()
if(NOT USE_SYSTEM_TSLMAP OR NOT TARGET tsl::robin_map)
  set(USE_SYSTEM_TSLMAP OFF)
  include(${CMAKE_CURRENT_LIST_DIR}/tsl_robin/tsl_robin.cmake)
endif()


message(STATUS "GTSAM_WITH_TBB is : ${GTSAM_WITH_TBB}")
if (GTSAM_WITH_TBB)
    message(STATUS "Installing GTSAM_WITH_TBB")    

    # Find TBB
    #find_package(TBB REQUIRED)
    find_package(TBB 4.4 COMPONENTS tbb tbbmalloc REQUIRED)

    message(STATUS "TBB_FOUND is set to ${TBB_FOUND}")

    # Set up variables if we're using TBB
    if(TBB_FOUND)
        set(GTSAM_USE_TBB 1)  # This will go into config.h

        if ((${TBB_VERSION_MAJOR} GREATER 2020) OR (${TBB_VERSION_MAJOR} EQUAL 2020))
            set(TBB_GREATER_EQUAL_2020 1)
        else()
            set(TBB_GREATER_EQUAL_2020 0)
        endif()
        # all definitions and link requisites will go via imported targets:
        # tbb & tbbmalloc
        list(APPEND GTSAM_ADDITIONAL_LIBRARIES TBB::tbb TBB::tbbmalloc)
    else()
        set(GTSAM_USE_TBB 0)  # This will go into config.h
    endif()

    message(STATUS "GTSAM_USE_TBB is set to ${GTSAM_USE_TBB}")

    ###############################################################################
    # Prohibit Timing build mode in combination with TBB
    if(GTSAM_USE_TBB AND (CMAKE_BUILD_TYPE  STREQUAL "Timing"))
        message(FATAL_ERROR "Timing build mode cannot be used together with TBB. Use a sampling profiler such as Instruments or Intel VTune Amplifier instead.")
    endif()

endif()







