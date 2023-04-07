#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

VEH="csc22918"      # Change this per bot
# launching app
dt-exec roslaunch duckiebot_detection duckiebot_detection_node.launch veh:=$VEH
dt-exec roslaunch lane_robot_follow lane_robot_follow_node.launch veh:=$VEH

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
