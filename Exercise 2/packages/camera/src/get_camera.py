#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

class MySubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        botName = os.environ['HOSTNAME']
        self.sub = rospy.Subscriber(f"{botName}/camera_node/image/compressed", CompressedImage, self.callback)

    def callback(self, data):
        rospy.loginfo(f"Image size: {data.format}")

if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='camera_node')
    # keep spinning
    rospy.spin()