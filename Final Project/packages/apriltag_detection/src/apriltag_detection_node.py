#!/usr/bin/env python3
import yaml

import cv2
from cv_bridge import CvBridge
import numpy as np

import rospy

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int32
from custom_msgs.msg import TagInfo
from custom_msgs.srv import Shutdown, ShutdownResponse
from sensor_msgs.msg import CompressedImage
from dt_apriltags import Detector

DEBUG = True
LOWER_BLUE = np.array([80,120,0])
UPPER_BLUE = np.array([120,255,255])

def read_yaml_file(path):
    """Read in the yaml file as a dictionary format."""
    with open(path, 'r') as f:
        try:
            yaml_dict = yaml.safe_load(f)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(f"YAML syntax error. File: {path}. Exc: {exc}")
            rospy.signal_shutdown()
            return


def parse_calib_params(int_path=None, ext_path=None):
    # Load dictionaries from files
    int_dict, ext_dict = None, None
    if int_path:
        int_dict = read_yaml_file(int_path)
    if ext_path:
        ext_dict = read_yaml_file(ext_path)
    
    # Reconstruct the matrices from loaded dictionaries
    camera_mat, distort_coef, proj_mat = None, None, None
    hom_mat = None
    if int_dict:
        # Get all the matrices from intirnsic parameters
        camera_mat = np.array(list(map(np.float32, int_dict["camera_matrix"]["data"]))).reshape((3, 3))
        distort_coef = np.array(list(map(np.float32, int_dict["distortion_coefficients"]["data"]))).reshape((1, 5))
        proj_mat = np.array(list(map(np.float32, int_dict["projection_matrix"]["data"]))).reshape((3, 4))
    if ext_dict:
        # Get homography matrix from extrinsic parameters
        hom_mat = np.array(list(map(np.float32, ext_dict["homography"]))).reshape((3, 3))

    return (camera_mat, distort_coef, proj_mat, hom_mat)


class ApriltagDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ApriltagDetectionNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC,
        )
        
        # Static variables
        self.veh = rospy.get_param("~veh")
        self.int_path = rospy.get_param("~int_path")
        self.ext_path = rospy.get_param("~ext_path")

        # Utility instances
        self.bridge = CvBridge()
        self.apriltag_detector = Detector(families="tag36h11",
                                          nthreads=1)
        
        # Initialize all the transforms
        self.camera_mat, self.distort_coef, self.proj_mat, self.hom_mat = \
            parse_calib_params(self.int_path, self.ext_path)
        
        # Parameters
        self.c_params = [
            self.camera_mat[0, 0],
            self.camera_mat[1, 1],
            self.camera_mat[0, 2],
            self.camera_mat[1, 2],
        ]
        self.h = -1
        self.w = -1
        
        self.running = True

        self.prev_tag_id = -1

        self.meta_cnt = 0
        self.cnt = 0

        self.detected = False
        
        # Service provider
        self.srv_node_shutdown = rospy.Service(
            f"/{self.veh}/{node_name}/shutdown",
            Shutdown,
            self.cb_shutdown
        )

        # Subscriber
        self.counts = 0
        self.cur_value = None
        self.sub_cam = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_cam
        )

        # Publishers
        self.pub_info = rospy.Publisher(
            f"/{self.veh}/{node_name}/taginfo",
            TagInfo,
            queue_size=1
        )

        rospy.on_shutdown(self.shutdown_hook)

    def undistort(self, img):
        """Undistort the distorted image with precalculated parameters."""
        return cv2.undistort(img,
                             self.camera_mat,
                             self.distort_coef,
                             None,
                             self.camera_mat)
    
    def publish(self, tag_id, area, ratio):
        """Publish and send the service request."""
        # TODO: Consider adding distance from tag
        msg = TagInfo()
        msg.tag_id = tag_id
        msg.area = area
        msg.area_ratio = ratio

        self.pub_info.publish(msg)

    def extract_tag_corners(self, tag):
        """Reformat the tag corners to usable format."""
        (ptA, ptB, ptC, ptD) = tag.corners

        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        return (ptA, ptB, ptC, ptD)

    def get_max_tag(self, tags):
        """Get maximum sized tag"""
        mx_area = 0
        mx_tag = None
        for tag in tags:
            ptA, ptB, ptC, _ = self.extract_tag_corners(tag)

            area = abs(ptA[0]-ptB[0]) * abs(ptB[1]-ptC[1])
            if area > mx_area:
                mx_area = area
                mx_tag = tag
        
        return mx_area, mx_tag

    def draw_bbox(self, img, corners, col=(0, 0, 255)):
        """Draw apriltag bbox."""
        # NOTE: Unsure whether we are going to use this or not
        a, b, c, d = corners
        
        cv2.line(img, a, b, col, 2)
        cv2.line(img, b, c, col, 2)
        cv2.line(img, c, d, col, 2)
        cv2.line(img, d, a, col, 2)

        if self.cur_value is not None:
            cx, cy = (a[0] + c[0]) // 2, (a[1] + b[1]) // 2
            cv2.putText(img, self.cur_value, (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 0, 0), 2)
        return img
        
    def cb_cam(self, msg):
        """Callback for camera image."""
        if self.counts % 7 == 0:
            if not self.running:
                rospy.signal_shutdown("Shutdown requested")
            img = np.frombuffer(msg.data, np.uint8) 
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            # Undistort an image
            img = self.undistort(img)

            if self.h == -1 and self.w == -1:
                # Initialize the image size 
                self.h, self.w = img.shape[:2]

            # Convert image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect an apriltag
            tags = self.apriltag_detector.detect(gray_img, True, self.c_params, 0.065)
            
            # Get tag with maximum area
            mx_area, mx_tag = self.get_max_tag(tags)
            
            # Publish images
            if mx_tag is not None:
                self.publish(mx_tag.tag_id, mx_area, mx_area / (self.h * self.w))
            else:
                self.publish(-1, -1, -1)

            self.counts = 0

        self.counts += 1

    def cb_shutdown(self, req):
        """Service provider callback."""
        #rospy.signal_shutdown("Shutdown requested")
        self.running = False
        return ShutdownResponse(0)
    
    def shutdown_hook(self):
        self.srv_node_shutdown.shutdown()
        rospy.loginfo("SHUTTING DOWN APRILTAG DETECTION")


if __name__ == "__main__":
    node = ApriltagDetectionNode("apriltag_detection_node") 
    rospy.spin()