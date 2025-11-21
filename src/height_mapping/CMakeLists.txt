cmake_minimum_required(VERSION 3.14)
project(height_mapping)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -mtune=native)

# Profiling option
option(ENABLE_PROFILING "Enable profiling and debug outputs" OFF)
if(ENABLE_PROFILING)
  add_compile_definitions(HEIGHT_MAPPING_PROFILE)
endif()

# MuJoCo spoofing node option
option(BUILD_MUJOCO_SPOOF_NODE "Build height mapping spoof node with MuJoCo raycasting" OFF)
if(BUILD_MUJOCO_SPOOF_NODE)
  message(STATUS "Building MuJoCo spoof node enabled")
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

# Core library (no ROS deps)
add_library(height_mapping_core
  src/HeightMapping.cpp
)
target_include_directories(height_mapping_core PUBLIC include)
target_link_libraries(height_mapping_core PUBLIC ${OpenCV_LIBS})

# ROS 2 node
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

add_executable(height_mapping_node src/HeightMappingNode.cpp)
target_include_directories(height_mapping_node PUBLIC include ${PCL_INCLUDE_DIRS})
ament_target_dependencies(height_mapping_node
  rclcpp sensor_msgs nav_msgs tf2 tf2_ros cv_bridge pcl_conversions
)

# Use the PLAIN signature here to match ament_target_dependencies' usage
target_link_libraries(height_mapping_node
  height_mapping_core
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

# (optional) if PCL sets definitions on your platform
if(PCL_DEFINITIONS)
  add_definitions(${PCL_DEFINITIONS})
endif()

# Conditionally build MuJoCo spoof node
if(BUILD_MUJOCO_SPOOF_NODE)
  find_package(ament_index_cpp REQUIRED)

  # Try to find MuJoCo via find_package first
  find_package(mujoco QUIET)

  if(NOT mujoco_FOUND)
    # Fall back to manual configuration for prebuilt binaries
    message(STATUS "MuJoCo not found via find_package, using manual configuration")

    # Try common installation paths
    set(MUJOCO_SEARCH_PATHS
      /usr/local/mujoco
      $ENV{MUJOCO_DIR}
      ${CMAKE_PREFIX_PATH}
    )

    find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h
      PATHS ${MUJOCO_SEARCH_PATHS}
      PATH_SUFFIXES include
    )

    find_library(MUJOCO_LIBRARY
      NAMES mujoco
      PATHS ${MUJOCO_SEARCH_PATHS}
      PATH_SUFFIXES lib
    )

    if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
      message(FATAL_ERROR "MuJoCo not found. Please set MUJOCO_DIR environment variable or install MuJoCo to /usr/local/mujoco")
    endif()

    message(STATUS "Found MuJoCo: ${MUJOCO_LIBRARY}")
    message(STATUS "MuJoCo include: ${MUJOCO_INCLUDE_DIR}")
  endif()

  add_executable(height_mapping_spoof_node src/HeightMappingSpoofNode.cpp)

  target_include_directories(height_mapping_spoof_node PUBLIC
    include
    ${PCL_INCLUDE_DIRS}
    ${MUJOCO_INCLUDE_DIR}
  )

  ament_target_dependencies(height_mapping_spoof_node
    rclcpp
    sensor_msgs
    nav_msgs
    tf2
    tf2_ros
    cv_bridge
    pcl_conversions
    ament_index_cpp
  )

  target_link_libraries(height_mapping_spoof_node
    height_mapping_core
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${MUJOCO_LIBRARY}
  )

  install(TARGETS height_mapping_spoof_node DESTINATION lib/${PROJECT_NAME})
endif()

install(TARGETS height_mapping_core DESTINATION lib)
install(TARGETS height_mapping_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)
install(DIRECTORY rsc/ DESTINATION share/${PROJECT_NAME}/rsc)

ament_package()

option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
  add_executable(test_pole_occlusion test/test_pole_occlusion.cpp)
  target_link_libraries(test_pole_occlusion PRIVATE height_mapping_core ${OpenCV_LIBS})
  target_include_directories(test_pole_occlusion PRIVATE include)

  add_executable(test_stair_occlusion test/test_stair_occlusion.cpp)
  target_link_libraries(test_stair_occlusion PRIVATE height_mapping_core ${OpenCV_LIBS})
  target_include_directories(test_stair_occlusion PRIVATE include)

  add_executable(test_stair_occlusion_multipos test/test_stair_occlusion_multipos.cpp)
  target_link_libraries(test_stair_occlusion_multipos PRIVATE height_mapping_core ${OpenCV_LIBS})
  target_include_directories(test_stair_occlusion_multipos PRIVATE include)
endif()
