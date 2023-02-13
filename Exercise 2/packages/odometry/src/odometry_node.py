#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float32, Header
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped, WheelEncoderStamped
import os

class OdometryNode(DTROS):
    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super().__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
        botName = os.environ['HOSTNAME']
        self.length = 0.05 # meters
        self.radius = 0.0318 # meters
        self.init_ticks = [None, None] # initial ticks
        self.distances = [None, None] # left and right distance
        self.dist = 0.00

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(f"{botName}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_encoder_data, callback_args="left")
        self.sub_encoder_ticks_right = rospy.Subscriber(f"{botName}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.cb_encoder_data, callback_args="right")
        self.sub_executed_commands = rospy.Subscriber(f"{botName}/wheels_driver_node/wheels_cmd_executed", WheelsCmdStamped, self.cb_executed_commands)

        # Publishers
        self.pub_integrated_distance_left = rospy.Publisher(f"{botName}/odometry_node/integrated_distance_left", Float32, queue_size=1)
        self.pub_integrated_distance_right = rospy.Publisher(f"{botName}/odometry_node/integrated_distance_right", Float32, queue_size=1)
        self.pub_executed_commands = rospy.Publisher(f"{botName}/odometry_node/wheels_cmd", WheelsCmdStamped, queue_size=1)

        self.log("Initialized")

    def cb_encoder_data(self, msg, wheel):
        """ Update encoder distance information from ticks.
        """
        if wheel == "left":
            # self.pub_integrated_distance_left.publish(msg.data)
            if self.init_ticks[0] is None:
                self.init_ticks[0] = msg.data
            self.distances[0] = self.calculate_distance(msg.data - self.init_ticks[0])
        elif wheel == "right":
            # self.pub_integrated_distance_right.publish(msg.data)
            if self.init_ticks[1] is None:
                self.init_ticks[1] = msg.data
            self.distances[1] = self.calculate_distance(msg.data - self.init_ticks[1])

        if self.distances[0] and self.distances[1]:
            self.dist = sum(self.distances)/len(self.distances)
        elif not self.distances[0]:
            self.dist = self.distances[0]
        elif not self.distances[1]:
            self.dist = self.distances[1]
        print(f"Distance: {self.dist}")

        
    def cb_executed_commands(self, msg):
        """ Use the executed commands to determine the direction of travel of each wheel.
        """
        print(f"==============\n{msg}\n==============")
        self.pub_executed_commands.publish(msg)

    def calculate_distance(self, ticks):
        '''Calculate distance travelled fromt icks
        '''
        return (2*ticks*np.pi/135)*self.radius

if __name__ == '__main__':
    node = OdometryNode(node_name='my_odometry_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")
