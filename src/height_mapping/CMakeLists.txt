cmake_minimum_required(VERSION 3.14)
project(height_mapping)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -mtune=native)

# Profiling option
option(ENABLE_PROFILING "Enable profiling and debug outputs" OFF)
if(ENABLE_PROFILING)
  add_compile_definitions(HEIGHT_MAPPING_PROFILE)
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

install(TARGETS height_mapping_core DESTINATION lib)
install(TARGETS height_mapping_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz)

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
