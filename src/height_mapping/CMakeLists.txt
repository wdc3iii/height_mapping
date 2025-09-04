cmake_minimum_required(VERSION 3.14)
project(largegrid_elevation_map)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -march=native -mtune=native)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)

# Core library (no ROS deps)
add_library(lem_core
  src/LargeGridElevationMap.cpp
)
target_include_directories(lem_core PUBLIC include)
target_link_libraries(lem_core PUBLIC ${OpenCV_LIBS})

# ROS 2 node
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

add_executable(lem_node src/lem_node.cpp)
target_include_directories(lem_node PUBLIC include)
ament_target_dependencies(lem_node
  rclcpp sensor_msgs nav_msgs tf2 tf2_ros cv_bridge pcl_conversions PCL
)
target_link_libraries(lem_node PRIVATE lem_core ${OpenCV_LIBS})

install(TARGETS lem_core DESTINATION lib)
install(TARGETS lem_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_package()
