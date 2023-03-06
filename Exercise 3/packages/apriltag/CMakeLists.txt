cmake_minimum_required(VERSION 2.8.3)
project(apriltag)
find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  sensor_msgs
  duckietown_msgs
)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
