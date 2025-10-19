
include(FetchContent)
FetchContent_Declare(tessil SYSTEM URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.0.1.tar.gz)
FetchContent_MakeAvailable(tessil)

FetchContent_MakeAvailable(tessil)

FetchContent_GetProperties(tessil)
if(NOT tessil_POPULATED)
  FetchContent_Populate(tessil)
endif()

# Only create the target if it doesn’t already exist
if(NOT TARGET tsl::robin_map)
  add_library(tsl_robin_map INTERFACE)
  add_library(tsl::robin_map ALIAS tsl_robin_map)
  target_include_directories(tsl_robin_map INTERFACE
    ${tessil_SOURCE_DIR}/include
  )
endif()