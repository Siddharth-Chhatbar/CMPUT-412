#!/usr/bin/env python3
import rospy
import subprocess

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import TagInfo
from custom_msgs.srv import Shutdown
from turbojpeg import TurboJPEG
import cv2
from cv_bridge import CvBridge
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
import numpy as np

LANE_COLOR = [(30, 60, 0), (50, 255, 255)]
# Mask resource: https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
STOP_COLOR = [(0, 100, 20), (10, 255, 255)]
DUCK_COLOR = [(10, 100, 0), (20, 255, 255)]
TAG_LEFT = 50
TAG_RIGHT = 48
TAG_STRAIGHT = 56
TAG_WALK = 163
TAG_PARKS = [207, 226, 228, 75]
TAG_PARK_ENTRANCE = 227

DEBUG = True
ENGLISH = False

class DriveNode(DTROS):
    """
    Initializes the drive node with PID control variables, Service Proxies, Subscribers, and Publishers.
    """
    def __init__(self, node_name):
        super(DriveNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC
        )
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")
        
        self.jpeg = TurboJPEG()

        # Parking lot ID
        self.lot_id = 4

        # Image related parameters
        self.width = None
        self.height = None
        self.lower_thresh = 150

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -180
        else:
            self.offset = 180

        self.dummy = CvBridge().cv2_to_compressed_imgmsg(np.zeros((2, 2), np.uint8))

        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        # Information related to apriltag
        self.cur_tag_id = -1
        self.cur_tag_area = -1
        self.cur_tag_aratio = -1

        # PID related terms
        self.P = 0.034
        self.D = -0.005
        self.last_error = 0
        self.last_time = rospy.get_time()
        
        # Variables to track on
        self.in_front = False
        self.is_stop = False
        self.is_stop_digit = False
        self.duck_front = False
        
        self.stage3 = False

        # Timer for stopping at the intersection
        self.t_stop = 5     # stop for this amount of seconds at intersections
        self.t_start = 0    # Measures the amount of time it stops

        # Timer for turning/going straight at the intersection
        self.turning = False 
        self.t_turn = 2
        self.t_turn_start = 0

        ### Service proxies
        # Service for shutting down the apriltag detection node
        rospy.wait_for_service(f"/{self.veh}/apriltag_detection_node/shutdown")
        self.shutdown_tag = rospy.ServiceProxy(
            f"/{self.veh}/apriltag_detection_node/shutdown",
            Shutdown
        )

        # Service for shutting down the duckiebot detection node
        rospy.wait_for_service(f"/{self.veh}/duckiebot_detection_node/shutdown")
        self.shutdown_det = rospy.ServiceProxy(
            f"/{self.veh}/duckiebot_detection_node/shutdown",
            Shutdown
        )

        # Service for shutting down the duckiebot distance node
        rospy.wait_for_service(f"/{self.veh}/duckiebot_distance_node/shutdown")
        self.shutdown_dist = rospy.ServiceProxy(
            f"/{self.veh}/duckiebot_distance_node/shutdown",
            Shutdown
        )
        
        ### Subscribers
        # Camera images
        self.sub_img = rospy.Subscriber(
            "/" + self.veh + "/camera_node/image/compressed",
            CompressedImage,
            self.cb_image,
            queue_size=1,
            buff_size="20MB"
        )

        self.sub_taginfo = rospy.Subscriber(
            f"/{self.veh}/apriltag_detection_node/taginfo",
            TagInfo,
            self.cb_tag,
            queue_size=1
        )

        self.sub_breakdown = rospy.Subscriber(
            f"/{self.veh}/breakdown_node/breakdown", 
            BoolStamped, 
            self.cb_breakdown,
            queue_size=1)

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

        # Shutdown hook
        rospy.on_shutdown(self.hook)
    
    def cb_tag(self, msg):
        """
        Store the currently detected tag information.
        """
        if msg.area != -1:
            self.cur_tag_id = msg.tag_id
            #rospy.loginfo(f"Tag ID: {self.cur_tag_id}, Area: {msg.area}, Area Ratio: {msg.area_ratio}")

        if not self.is_stop:
            self.cur_tag_area = msg.area
            self.cur_tag_aratio = msg.area_ratio
        else:
            self.cur_tag_area = -1
            self.cur_tag_aratio = -1
        
        if (self.cur_tag_id == TAG_WALK and self.cur_tag_aratio > 0.017) or \
           (self.cur_tag_id in TAG_PARKS and self.cur_tag_aratio > 0.03):
            # Prevent is_stop condition to be indefinitely true
            # by constantly updating the tag area (and its ratio)
            self.is_stop = True
            self.t_start = rospy.get_rostime().secs
    
    def get_max_contour(self, contours):
        """
        Returns the index of contour with maximum area.
        """
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        return max_area, max_idx

    def get_contour_center(self, contours, idx):
        """
        Returns the center coordinate of specified contour.
        """
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
        """
        Get the location of contour based on image and color range.
        """
        # Convert corpped image to HSV and get mask corresponding to the
        # specified color
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, contour_color[0], contour_color[1])

        # Etract contours from mask
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        
        # Only get contour with maximum area, and compute center of it
        mx_area, mx_idx = self.get_max_contour(contours)
        x, y = self.get_contour_center(contours, mx_idx)
        
        return x, y, mx_area

    def cb_image(self, msg):
        """
        Callback for image message.
        """
        img = self.jpeg.decode(msg.data)

        if self.width == None:
            self.height, self.width = img.shape[:2]
        
        # Get location of red intersection line
        red_crop = img[300:-1, 200:-200, :]
        _, ry, _ = self.compute_contour_location(red_crop, STOP_COLOR)

        # Update status of stopping or not
        if not self.is_stop and not self.turning and ry >= self.lower_thresh:
            # Start a timer and turn the stop flag
            self.is_stop = True
            self.t_start = rospy.get_rostime().secs

        # Get location of line
        crop = img[int(0.7 * self.height):-1, :450, :]
        lx, _, _ = self.compute_contour_location(crop, LANE_COLOR)
       
        # Set proportional values if needed 
        if lx != -1:
            self.proportional = lx - int(self.width / 2) + self.offset + (20 if self.stage3 else 0)
        else:
            self.proportional = None
        
        if self.is_stop and self.cur_tag_id == TAG_WALK:
            # Only check for the pedestrian (duck) when robot is stopping
            duck_crop = img[:, int(self.width * 0.2):int(self.width * 0.8), :]
            _, _, mx_area = self.compute_contour_location(duck_crop, DUCK_COLOR)
            
            # Indefinitely stop if the pedestrain is in front
            if mx_area > 1000:
                self.duck_front = True
            else:
                self.duck_front = False

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(img))
            self.pub.publish(rect_img_msg)
    
    def cb_breakdown(self, msg):
        """
        Callback for breakdown message.
        """
        if msg.data:
            self.in_front = True
        else:
            self.in_front = False

    def lane_follow(self):
        """
        Compute odometry values with PID.
        """
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
            
        return self.velocity, P + D
        
    def stop(self):
        """
        Stop the vehicle completely.
        """
        return 0, 0
    
    def straight(self):
        """
        Move vehicle in forward direction.
        """
        rospy.loginfo("Moving straight...")
        return self.velocity + 0.2, -0.7

    def turn(self, right=True, is_stage3=False, lot_id=None):
        """
        Turn the car at the intersection according to the direction given.
        """
        omega = 0.0
        if is_stage3:
            if lot_id == 1:
                omega = 2.2
            elif lot_id == 2:
                omega = 3.4
            elif lot_id == 3:
                omega = -3.5
            else:
                omega = -3.9
        else:
            if right:
                omega = -4.6
            else:
                omega = 2.3

        if DEBUG:
            rospy.loginfo(f"Turning {'right' if right else 'left'}...")
        return self.velocity, omega

    def breakdown_turn(self):
        """
        Turn the duckiebot around the broken duckiebot.
        """
        rospy.loginfo("Breakdown turn")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        rospy.sleep(2)
        
        # define range 
        omega = 0
        k = 2
        a = 1.7

        for v in np.linspace(0, 2*np.pi - (np.pi/4), num=10):
            s = np.sin(k * v) * k * a

            if s > 0:
                if v < np.pi/2:
                    omega = s
                else:
                    omega = 1.7*-s
            else:
                if v < (np.pi*3/2):
                    omega = 1.6*s
                else:
                    omega = -s

            self.twist.v = 0.3
            self.twist.omega = omega
            self.vel_pub.publish(self.twist)
            rospy.sleep(0.5)
        
        self.in_front = False
        rospy.loginfo("Breakdown turn done")
    
    def drive(self):
        """
        Decide how to move based on the given information and flags.
        """
        if self.is_stop and \
            not self.duck_front and \
            (rospy.get_rostime().secs - self.t_start) >= self.t_stop:
            # Move in a desired direction after stopping
            self.is_stop = False

            if self.cur_tag_id != TAG_WALK:
                # When the stop is not at the crosswalk, duckiebot requires turning
                # i.e. continue with PID when it is after the crosswalk
                self.turning = True
                if self.cur_tag_id == TAG_PARK_ENTRANCE:
                    # Transition to stage 3 (parking) and get the user input
                    self.stage3 = True
                self.t_turn_start = rospy.get_rostime().secs

        elif self.turning and ((rospy.get_rostime().secs - self.t_turn_start) >= self.t_turn):
            # Switch from manual control to PID control (TURN to DRIVE)
            self.turning = False

        # Determine the velocity and angular velocity
        v, omega = 0, 0
        if self.is_stop:
            # Track the center of leader robot to decide the direction to turn
            v, omega = self.stop()
            if self.cur_tag_id in TAG_PARKS:
                rospy.signal_shutdown("Done")

        elif self.turning and ((rospy.get_rostime().secs - self.t_turn_start) < self.t_turn):
            if self.cur_tag_id in [TAG_LEFT, TAG_RIGHT] or self.stage3:
                v, omega = self.turn(True if self.cur_tag_id == TAG_RIGHT else False, self.stage3, self.lot_id)
            else:
                v, omega = self.straight()
        else:
            v, omega = self.lane_follow()

        self.twist.v, self.twist.omega = v, omega
        
        # Publish the resultant control values
        self.vel_pub.publish(self.twist)
    
    def hook(self):
        """
        Hook for shutting down the entire system.
        """
        rospy.loginfo("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)

        self.shutdown_tag(0)
        self.shutdown_dist(0)
        self.shutdown_det(0)
        
        subprocess.run(["rosnode", "kill", "breakdown_node"])
        

if __name__ == "__main__":
    node = DriveNode("drive_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        # Periodically send the driving command
        if node.in_front:
            node.breakdown_turn()
        node.drive()
        rate.sleep()
