#!/usr/bin/env python3

import cv2
import rospy
import numpy as np

from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from duckietown_msgs.srv import SetCustomLEDPattern
from duckietown_msgs.msg import LEDPattern
from duckietown.dtros import DTROS, NodeType
from dt_apriltags import Detector
from image_geometry import PinholeCameraModel
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import math
import os

class AprilTagDetector(DTROS):
    def __init__(self):
        super(AprilTagDetector, self).__init__(node_name="apriltag_detector_node", node_type=NodeType.PERCEPTION)

        self.bridge = CvBridge()
        self.detector = Detector(families="tag36h11")
        self.botName = os.environ['VEHICLE_NAME']
        #self._camera_parameters = [326.4579571121477,304.55075953611606,321.80115517357297,220.11422885592438]
        self._camera_parameters = [337.9774997168973,300.4345583102675,337.94862951764776,238.9711257300856]

        """
        None: White
        UofA[93,94,200,201]: Green
        T[58,62,133,153]: Blue
        Stop[162,169]: Red
        """
        self.count = 0

        self.colours = {
            93: "green",
            94: "green",
            200: "green",
            201: "green",
            143: "green",
            58: "blue",
            62: "blue",
            133: "blue",
            153: "blue",
            162: "red",
            169: "red",
        }

        self.image_sub = rospy.Subscriber(f"/{self.botName}/camera_node/image/compressed", CompressedImage, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher(f"/{self.botName}/apriltags/image/compressed", CompressedImage, queue_size=1)     

        self.br = tf2_ros.TransformBroadcaster()
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)


    def image_callback(self, data):
        # processing only every 30th frame
        if self.count == 29:
            self.count = 0

            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
                return

            grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            rospy.wait_for_service('/' + self.botName + '/led_emitter_node/set_custom_pattern')
            self.service = rospy.ServiceProxy('/' + self.botName + '/led_emitter_node/set_custom_pattern', SetCustomLEDPattern)

            results = self.detector.detect(grey, estimate_tag_pose=True, camera_params=self._camera_parameters, tag_size=0.065)

            for r in results:
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))

                cv2.line(cv_image, ptA, ptB, (0, 255, 0), 2)
                cv2.line(cv_image, ptB, ptC, (0, 255, 0), 2)
                cv2.line(cv_image, ptC, ptD, (0, 255, 0), 2)
                cv2.line(cv_image, ptD, ptA, (0, 255, 0), 2)

                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(cv_image, (cX, cY), 5, (0, 0, 255), -1)

                tagFamily = r.tag_family.decode("utf-8")
                
                cv2.putText(cv_image, str(r.tag_id), (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                r = results[0]

                print(f"[INFO] tag family: {r.tag_id}")

                self.transformation(r.pose_t, r.pose_R, r.tag_id)
                self.apply_to_static(r.tag_id)
                self.change_color(self.colours[r.tag_id] if r.tag_id in self.colours else "white")

            if not results:
                self.change_color("white")

            img_msg = CompressedImage()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.format = "jpg"
            img_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
            self.image_pub.publish(img_msg)

        self.count += 1
    
    def change_color(self, color):
        msg = LEDPattern()
        msg.color_list = [color] * 5
        msg.color_mask = [1, 1, 1, 1, 1]
        msg.frequency = 0
        msg.frequency_mask = [0, 0, 0, 0, 0]
        self.service(msg)
    
    def transformation(self, pose_t, pose_R, tag_id):
        
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = f"{self.botName}/camera_optical_frame"
        t.child_frame_id = "tag_" + str(tag_id)
        t.transform.translation.x = float(pose_t[0][0])
        t.transform.translation.y = float(pose_t[1][0])
        t.transform.translation.z = float(pose_t[2][0])

        yaw = math.atan2(pose_R[1][0], pose_R[0][0])
        pitch = math.atan2(-pose_R[2][0], math.sqrt(pose_R[2][1] ** 2 + pose_R[2][2] ** 2))
        roll = math.atan2(pose_R[2][1], pose_R[2][2])

        e = tf_conversions.transformations.quaternion_from_euler(roll, pitch, yaw)
        t.transform.rotation.x = e[0]
        t.transform.rotation.y = e[1]
        t.transform.rotation.z = e[2]
        t.transform.rotation.w = e[3]

        self.br.sendTransform(t)
    
    def apply_to_static(self, tag_id):
        """
        1. Get the transform from the apriltag location to your wheelbase in world frame
        2. Apply this transformation to the known location of the apriltag, this is your robot location.
        3. Teleport your robot to the position and rotation calculated by the transform in step 2"""
        try:
            transform = self.buffer.lookup_transform(f"tag_{tag_id}", f"{self.botName}/odom", rospy.Time())
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = f"_{tag_id}_"
            t.child_frame_id = "pridected_location"
            t.transform = transform.transform
            self.br.sendTransform(t)

            transform = self.buffer.lookup_transform(f"{self.botName}/world", f"_{tag_id}_", rospy.Time())
            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = f"{self.botName}/world"
            t.child_frame_id = f"_{tag_id}_"
            # send x and y and quaternion
            t.transform.translation.x = transform.transform.translation.x
            t.transform.translation.y = transform.transform.translation.y
            t.transform.translation.z = 0
            t.transform.rotation = transform.transform.rotation
            self.br.sendTransform(t)



        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("error")


    
if __name__ == '__main__':
    node = AprilTagDetector()
    rospy.spin()