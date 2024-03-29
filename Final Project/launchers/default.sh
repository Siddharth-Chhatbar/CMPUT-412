#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
dt-exec roslaunch duckiebot_detection duckiebot_detection_node.launch veh:=csc22918
dt-exec roslaunch apriltag_detection apriltag_detection_node.launch veh:=csc22918
dt-exec roslaunch drive drive_node.launch veh:=csc22918
dt-exec roslaunch breakdown breakdown_node.launch veh:=csc22918

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
