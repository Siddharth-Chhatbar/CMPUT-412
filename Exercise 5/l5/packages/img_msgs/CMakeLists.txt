cmake_minimum_required(VERSION 2.8.3)
project(img_msgs)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
    duckietown_msgs
    sensor_msgs
    roscpp
    rospy
    message_generation
)

add_service_files(
    FILES
    DigitImage.srv
)

generate_messages(
    DEPENDENCIES
    sensor_msgs
    std_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)