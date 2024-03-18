#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
import message_filters
import cv_bridge
from tf2_ros import Buffer, TransformListener
from tf.transformations import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from thread_reconstruction.srv import GetGraspPoint, GetGraspPointResponse

import os
import cv2
import numpy as np

from segmenter import SAMSegmenter
from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from optim import optim
from utils import *

MM_TO_M = 1/1000

class ThreadReconstrNode:
    def __init__(self):
        # Set up camera params
        calib = os.path.dirname(__file__) + "/../../camera_calibration_sarah.yaml"
        cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
        K1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        K2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        ImageSize = cv_file.getNode("ImageSize").mat()
        img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
        new_size = (640, 480)

        R1, R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
        
        self.cam2img = self.P1[:,:-1]
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(K1, D1, R1, self.P1, new_size, cv2.CV_16SC2)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(K2, D2, R2, self.P2, new_size, cv2.CV_16SC2)

        # Segmenter
        self.segmenter = SAMSegmenter("cpu", "vit_h")

        # Set up transform listers
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf)

        # Set up grasp point pub
        self.grasp_pub = rospy.Publisher("/thread_reconstr/grasp_pose", PoseStamped, queue_size=10)

        # Set up rectified stereo img listener
        self.left = None
        self.right = None
        self.stamp = None
        self.bridge = cv_bridge.CvBridge()
        self.rectified_left_sub = message_filters.Subscriber("/stereo/left/rectified_downscaled_image", Image, queue_size=1)
        self.rectified_right_sub = message_filters.Subscriber("/stereo/right/rectified_downscaled_image", Image, queue_size=1)
        self.sync_stereo = message_filters.ApproximateTimeSynchronizer([self.rectified_left_sub, self.rectified_right_sub], queue_size=1, slop=0.2)
        self.sync_stereo.registerCallback(self.get_stereo_pair)
    
    def get_stereo_pair(self, left, right):
        try:
            self.left = self.bridge.imgmsg_to_cv2(left, desired_encoding="rgb8")
            self.right = self.bridge.imgmsg_to_cv2(right, desired_encoding="rgb8")
            self.stamp = left.header.stamp
        except Exception as e:
            rospy.logwarn("thread_reconstr_node: Image pair can't be saved: %s" % (e,))
    
    def reconstruct(self, request):
        if self.left is None or self.right is None or self.stamp is None:
            rospy.logwarn("thread_reconstr_node: No images received yet")
            return None
        
        img1 = cv2.remap(self.left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        img2 = cv2.remap(self.right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        
        mask1 = self.segmenter.segmentation(img1)
        mask2 = self.segmenter.segmentation(img2)

        stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
        img1 = np.where(stack_mask1>0, img1, 0)
        stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
        img2 = np.where(stack_mask2>0, img2, 0)
        
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        
        # Perform reconstruction
        img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, self.Q)
        img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
        spline = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, self.cam2img, self.P1, self.P2)

        # Get grasp pose in camera frame
        grasp_s = 0.5 # TODO Consult input for grasp point
        grasp_point = spline(grasp_s) * MM_TO_M
        # Choose default grasp pose aligned with thread, flat in xy-plane
        dspline = spline.derivative()
        grasp_z = dspline(grasp_s)
        cam_z = np.array([0, 0, 1]) 
        grasp_x = np.cross(cam_z, grasp_z)
        grasp_y = np.cross(grasp_z, grasp_x)
        # Represent as matrix
        R_grasp_ori = np.eye(4)
        R_grasp_ori[:3, 0] = grasp_x / np.linalg.norm(grasp_x)
        R_grasp_ori[:3, 1] = grasp_y / np.linalg.norm(grasp_y)
        R_grasp_ori[:3, 2] = grasp_z / np.linalg.norm(grasp_z)
        # Convert to PoseStamped
        grasp_quat_cam = quaternion_from_matrix(R_grasp_ori)
        grasp_pose = PoseStamped()
        grasp_pose.header.stamp = self.stamp
        grasp_pose.header.frame_id = "dvrk_cam"
        grasp_pose.pose.position.x = grasp_point[0]
        grasp_pose.pose.position.y = grasp_point[1]
        grasp_pose.pose.position.z = grasp_point[2]
        grasp_pose.pose.orientation.x = grasp_quat_cam[0]
        grasp_pose.pose.orientation.y = grasp_quat_cam[1]
        grasp_pose.pose.orientation.z = grasp_quat_cam[2]
        grasp_pose.pose.orientation.w = grasp_quat_cam[3]

        # Send grasp point
        self.grasp_pub.publish(grasp_pose)
        return GetGraspPointResponse(grasp_pose)

if __name__ == "__main__":
    rospy.init_node("thread_reconstr_node", anonymous=True)
    node = ThreadReconstrNode()
    service = rospy.Service('thread_reconstr_node/grasp', GetGraspPoint, node.reconstruct)
    rospy.spin()
