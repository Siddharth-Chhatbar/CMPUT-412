cmake_minimum_required(VERSION 2.8.3)
project(led_emitter)


find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)