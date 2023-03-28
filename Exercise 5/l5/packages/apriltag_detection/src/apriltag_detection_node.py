#!/usr/bin/env python3
import yaml

import cv2
from cv_bridge import CvBridge
import numpy as np

import rospy

from duckietown.dtros import DTROS, NodeType
from img_msgs.srv import DigitImage
from std_msgs.msg import Int32
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

        self.prev_tag_id = -1

        self.meta_cnt = 0
        self.cnt = 0

        # Service proxies
        rospy.wait_for_service(f"/{self.veh}/get_digit_img")
        self.send_digit = rospy.ServiceProxy(f"/{self.veh}/get_digit_img", DigitImage)

        self.detected = False

        # Subscriber
        self.counts = 0
        self.cur_value = None
        self.sub_cam = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.cb_cam
        )

        # Publishers
        self.mx_area = 0
        self.mx_tag_id = 0
        self.pub_view = rospy.Publisher(
            f"/{self.veh}/{node_name}/image/compressed",
            CompressedImage,
            queue_size=1
        )

        self.pub_digit_img = rospy.Publisher(
            f"/{self.veh}/{node_name}/digit_img/compressed",
            CompressedImage,
            queue_size=1
        )

    def undistort(self, img):
        """Undistort the distorted image with precalculated parameters."""
        return cv2.undistort(img,
                             self.camera_mat,
                             self.distort_coef,
                             None,
                             self.camera_mat)

    def publish(self, marked_img=None, digit_img=None):
        """Publish and send the service request."""
        if marked_img is not None:
            msg = self.bridge.cv2_to_compressed_imgmsg(marked_img)
            self.pub_view.publish(msg)
        
        if digit_img is not None:
            msg = self.bridge.cv2_to_compressed_imgmsg(digit_img)
            x = self.send_digit(msg, self.mx_tag_id)

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

            area = abs(ptA[1]-ptB[1]) * abs(ptB[0] * ptC[0])
            if area > mx_area:
                mx_area = area
                mx_tag = tag
        
        if mx_tag is not None:
            self.mx_tag_id = mx_tag.tag_id
        return mx_area, mx_tag

    def draw_bbox(self, img, corners, col=(0, 0, 255)):
        """Draw apriltag bbox."""
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
        
    def get_digit_img(self, img, tag):
        """Get cropped image for digit"""
        # Enumerate through the detection results
        (ptA, ptB, ptC, ptD) = self.extract_tag_corners(tag)
        
        # Crop sufficiently large region of digit image
        x_lst = [ptA[0], ptB[0], ptC[0], ptD[0]]
        y_lst = [ptA[1], ptB[1], ptC[1], ptD[1]]
        
        mn_x, mx_x = min(x_lst), max(x_lst)
        mn_y, mx_y = min(y_lst), max(y_lst)

        corners = (
            (max(0, mn_x-20), max(0, mn_y-(mx_y-mn_y)-20)),
            (max(0, mn_x-20), mn_y),
            (min(mx_x+20, img.shape[1]), mn_y),
            (min(mx_x+20, img.shape[1]), max(0, mn_y-(mx_y-mn_y)-20)),
        )

        cropped = np.copy(img[corners[0][1]:corners[1][1], corners[0][0]:corners[2][0], :])
        
        # Extract the contour with blue (background of digit)
        cr_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(cr_hsv, LOWER_BLUE, UPPER_BLUE)
        contours, _ = cv2.findContours(msk,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)

        # Get a contour with maximum area
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        # Create mask to fill out the background
        fwd_mask = np.zeros_like(msk)
        cv2.fillPoly(fwd_mask, [contours[max_idx]], 255)
        bck_mask = fwd_mask != 255

        # Mask out the background
        msk[bck_mask] = 255
        
        # Denoise and make character more visible
        msk = cv2.dilate(msk, np.ones((3, 3), dtype=np.uint8))
        digit_img = 255 - msk

        return digit_img, corners 

    def cb_cam(self, msg):
        """Callback for camera image."""
        if self.counts % 5 == 0:
            img = np.frombuffer(msg.data, np.uint8) 
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            if self.h == -1 and self.w == -1:
                # Initialize the image size 
                self.h, self.w = img.shape[:2]
            
            # Undistort an image
            img = self.undistort(img)

            # Convert image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect an apriltag
            tags = self.apriltag_detector.detect(gray_img, True, self.c_params, 0.065)
            
            # Get tag with maximum area
            mx_area, mx_tag = self.get_max_tag(tags)
            
            digit_img, view_img = None, img
            if mx_area > (self.h * self.w * 0.3):
                # Plot the bbox of apriltag and crop the digit image if apriltag
                # bbox area is reasonably larage
                
                digit_img, corners = self.get_digit_img(img, mx_tag)
                view_img = self.draw_bbox(img, corners)

                if digit_img.sum() < 10:
                    digit_img = None
                # else:
                #     cv2.imwrite(f"/data/samples/{self.meta_cnt}_{self.cnt}.png", digit_img)
                #     self.cnt += 1
                
                self.prev_tag_id = mx_tag.tag_id
            
            # Publish images
            self.publish(view_img, digit_img)

            self.counts = 0

        self.counts += 1
        

if __name__ == "__main__":
    node = ApriltagDetectionNode("apriltag_detection_node") 
    rospy.spin()