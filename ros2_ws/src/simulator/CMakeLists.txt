cmake_minimum_required(VERSION 3.8)
project(simulator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(mpc src/mpc.cpp)
ament_target_dependencies(mpc rclcpp)

add_executable(ekf src/ekf.cpp)
ament_target_dependencies(ekf rclcpp std_msgs nav_msgs geometry_msgs)

add_executable(object_sim src/object.cpp)
ament_target_dependencies(object_sim rclcpp geometry_msgs visualization_msgs)

install(TARGETS
  mpc
  ekf
  object_sim
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
