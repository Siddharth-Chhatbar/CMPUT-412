#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Float32
from custom_msgs.srv import Shutdown, ShutdownResponse
from duckietown_msgs.msg import BoolStamped

class BreakdownNode(DTROS):
    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(BreakdownNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.node_name = node_name
        self.veh = rospy.get_param('~veh')

        self.current_distance = 0.5
        self.in_front = False

        # Publishers and Subscribers
        self.pub = rospy.Publisher(f"/{self.veh}/breakdown_node/breakdown", BoolStamped, queue_size=1)

        self.sub_distance = rospy.Subscriber(f"/{self.veh}/duckiebot_distance_node/distance", Float32, self.cb_distance)

        self.sub_detect = rospy.Subscriber(f"/{self.veh}/duckiebot_detection_node/detection", BoolStamped, self.cb_detect)

        self.loginfo("Initialized")

        rospy.on_shutdown(self.shutdown_hook)
    
    def cb_distance(self, msg):
        self.current_distance = round(msg.data,2)
    
    def cb_detect(self, msg):
        self.in_front = msg.data
        #print(self.current_distance)
        #print(self.in_front)
        if self.in_front and self.current_distance < 0.5:
            #print("BREAKDOWN")
            self.pub.publish(msg)
        else:
            #print("NO BREAKDOWN")
            self.pub.publish(msg)

    def shutdown_hook(self):
        rospy.loginfo("SHUTTING DOWN BREAKDOWN NODE")

if __name__ == '__main__':
    # create the node
    node = BreakdownNode(node_name='breakdown_node')
    # keep spinning
    rospy.spin()