#!/usr/bin/env python3
import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int32
from sensor_msgs.msg import CompressedImage
from img_msgs.srv import DigitImage, DigitImageResponse
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge
from duckietown_msgs.msg import Twist2DStamped
import numpy as np

LANE_COLOR = [(20, 60, 0), (50, 255, 255)]
# Mask resource: https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
STOP_COLOR = [(0, 100, 20), (10, 255, 255)]

DEBUG = False
ENGLISH = False

class DriveNode(DTROS):
    def __init__(self, node_name):
        super(DriveNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        ### Service proxy
        rospy.wait_for_service("/local/digit_class")
        self.get_pred = rospy.ServiceProxy("/local/digit_class", DigitImage)
        # rospy.loginfo(f"Service proxy to /local/digit_class is ready")

        rospy.wait_for_service("/local/shut_down")
        self.shutdown_local = rospy.ServiceProxy("/local/shut_down", DigitImage)

        ### Services
        self.srv_digit_img = rospy.Service(
            f"/{self.veh}/get_digit_img",
            DigitImage,
            self.cb_digit_img
        )

        ### Subscribers
        # Camera images
        self.sub = rospy.Subscriber(
            "/" + self.veh + "/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1,
            buff_size="20MB"
        )

        ### Publishers
        # Publish mask image for debug
        self.pub = rospy.Publisher(
            "/" + self.veh + "/output/image/mask/compressed",
            CompressedImage,
            queue_size=1
        )

        # Publish car command to control robot
        self.vel_pub = rospy.Publisher(
            "/" + self.veh + "/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        
        self.jpeg = TurboJPEG()

        if DEBUG:
            self.loginfo("Initialized")

        # Image related parameters
        self.width = None
        self.lower_thresh = 150

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220

        self.dummy = CvBridge().cv2_to_compressed_imgmsg(np.zeros((2, 2), np.uint8))

        self.velocity = 0.34
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.intersection_turn = None

        self.preds = np.zeros(10)

        # PID related terms
        self.P = 0.037
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()
        
        # Variables to track on
        self.is_stop = False
        self.is_stop_digit = False

        # Timer for stopping at the intersection
        self.t_stop = 5     # stop for this amount of seconds at intersections
        self.t_start = 0    # Measures the amount of time it stops

        # Timer for turning/going straight at the intersection
        self.turning = False 
        self.t_turn = 2
        self.t_turn_start = 0

        # Record for already-seen values
        self.seen = [0] * 10

        # Static location of apriltags
        # TODO: change keys to digits
        self.tag_locations = {
            200: [0.17, 0.17],
            201: [1.65, 0.17],
            94: [1.65, 2.84],
            93: [0.17, 2.84],
            153: [1.75, 1.252],
            133: [1.253, 1.755],
            58: [0.574, 1.259],
            62: [0.075, 1.755],
            169: [0.574, 1.755],
            162: [1.253, 1.253],
        }

        # Shutdown hook
        rospy.on_shutdown(self.hook)
    
    def cb_digit_img(self, req):
        """Callback for digit image service."""
        # Get prediction for the given image
        pred = self.get_pred(req.data, 1)

        if req.tag_id != self.intersection_turn:

            if self.intersection_turn is not None:
                cls = self.preds.argmax(axis=0)
                self.seen[cls] = 1
                # Print out the results
                rospy.loginfo(f"Prediction for the tag id: {self.intersection_turn} = {cls}")
                rospy.loginfo(f"Location: {self.tag_locations[self.intersection_turn]}")

            self.intersection_turn = req.tag_id
            self.preds = np.zeros(10)

        self.preds[pred.cls] += 1

        # return pred
        return pred
        
    def get_max_contour(self, contours):
        """Returns the index of contour with maximum area."""
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        return max_area, max_idx

    def get_contour_center(self, contours, idx):
        """Returns the center coordinate of specified contour."""
        x, y = -1, -1
        if idx != -1:
            M = cv2.moments(contours[idx])
            try:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            except:
                pass
        
        return x, y

    def compute_contour_location(
        self,
        cropped_img,
        contour_color
    ):
        """Get the location of contour based on image and color range."""
        # Convert corpped image to HSV and get mask corresponding to the
        # specified color
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, contour_color[0], contour_color[1])

        # Etract contours from mask
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        
        # Only get contour with maximum area, and compute center of it
        _, mx_idx = self.get_max_contour(contours)
        x, y = self.get_contour_center(contours, mx_idx)
        
        return x, y

    def callback(self, msg):
        """Callback for image message."""
        img = self.jpeg.decode(msg.data)

        if self.width == None:
            self.width = img.shape[1]
        
        # Get location of red intersection line
        red_crop = img[300:-1, 200:-200, :]
        rx, ry = self.compute_contour_location(red_crop, STOP_COLOR)

        # Update status of stoppiing or not
        if not self.is_stop and not self.turning and ry >= self.lower_thresh:
            if DEBUG:
                rospy.loginfo("Stopping at the intersection.")
            
            # Start a timer and turn the stop flag
            self.is_stop = True
            self.t_start = rospy.get_rostime().secs

        # Get location of the lane
        crop = img[300:-1, :, :]
        lx, ly = self.compute_contour_location(crop, LANE_COLOR)
        
        # Set proportional values if needed 
        if lx != -1 and ly != -1:
            self.proportional = lx - int(self.width / 2) + self.offset
        else:
            self.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def lane_follow(self):
        """Compute odometry values with PID"""
        if self.proportional is None:
            P = 0
            D = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            if DEBUG:
                self.loginfo(self.proportional, P, D, self.velocity, P+D)
            
        return self.velocity, P + D
        
    def stop(self):
        """Stop the vehicle completely."""
        return 0, 0
    
    def straight(self):
        """Move vehicle in forward direction."""
        return self.velocity, 0

    def turn(self, right=True):
        """Turn the car at the intersection according to the direction given."""
        if right:
            print("Turing right...")
            return self.velocity, -4.0
        else:
            print("Turining left...")
            return self.velocity, 3.0
    
    def drive(self):
        """Decide how to move based on the given information and flags."""
        if self.is_stop and ((rospy.get_rostime().secs - self.t_start) >= self.t_stop):
            # Move in a desired direction after stopping
            self.is_stop = False
            self.turning = True
            self.t_turn_start = rospy.get_rostime().secs
        elif self.turning and ((rospy.get_rostime().secs - self.t_turn_start) >= self.t_turn):
            # Switch from manual control to PID control (TURN to DRIVE)
            self.turning = False
            
        # Determine the velocity and angular velocity
        v, omega = 0, 0
        if self.is_stop and (rospy.get_rostime().secs - self.t_start) < self.t_stop:
            # Track the center of leader robot to decide the direction to turn
            v, omega = self.stop()
        elif self.turning and ((rospy.get_rostime().secs - self.t_turn_start) < self.t_turn_start):
            if self.intersection_turn in [162, 153]:
                v, omega = self.turn(False)
            elif self.intersection_turn in [58, 169]:
                v, omega = self.turn(True)
            else:
                v, omega = self.straight()
        else:
            v, omega = self.lane_follow()

        self.twist.v, self.twist.omega = v, omega
        
        # Publish the resultant control values
        self.vel_pub.publish(self.twist)

    def hook(self):
        """Hook for shutting down the entire system"""
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        
        # Service call to shutdown
        self.shutdown_local(self.dummy, 1)
        for i in range(8):
            self.vel_pub.publish(self.twist)

        # Shutdown the proxy service

        self.shutdown_local.shutdown()
        self.get_pred.shutdown()

        


if __name__ == "__main__":
    node = DriveNode("drive_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        # Periodically send the driving command
        node.drive()

        if sum(node.seen) == 10:
            rospy.signal_shutdown("SHUTTING DOWN")
        rate.sleep()