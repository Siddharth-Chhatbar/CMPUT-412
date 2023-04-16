cmake_minimum_required(VERSION 2.8.3)
project(breakdown)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  duckietown_msgs
)

catkin_package()