cmake_minimum_required(VERSION 3.8)
project(lakibeam_driver_ros2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CURL REQUIRED)

add_executable(lakibeam_scan_node
  src/lakibeam_scan_node.cpp
  src/remote.cpp
)

target_include_directories(lakibeam_scan_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(lakibeam_scan_node rclcpp sensor_msgs)
target_link_libraries(lakibeam_scan_node CURL::libcurl)

target_compile_features(lakibeam_scan_node PUBLIC cxx_std_17)

install(TARGETS lakibeam_scan_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
