#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
from geometry_msgs.msg import PoseStamped
from thread_reconstruction.srv import Grasp, GraspResponse

import numpy as np
import time

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control.psm_control import PsmControl

from tf.transformations import quaternion_matrix

CAPTURE = 0
GUIDE = 1
GRASP = 2

class ExecGraspNode:
    def __init__(self, psm_control):
        self.psm_control = psm_control
        self.grasp_sub = rospy.Subscriber("/thread_reconstr/grasp_pose", PoseStamped, self.execute_grasp, queue_size=1)
    
    def execute_grasp(self, request):
        PSM = 2
        pose = request.graspPoint
        
        # Set up approach on capture
        if request.primitive == CAPTURE:
            R = quaternion_matrix([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w
            ])
            approach_len = 0.02
            grasp_point = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
            approach_point = grasp_point - R[:3, 1] * approach_len
            approach_pose_cam_ree = np.array([
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                approach_point[0],
                approach_point[1],
                approach_point[2]
            ])

            self.psm_control.openGripper(PSM)
            self.psm_control.controlPoseReeInCam(PSM, approach_pose_cam_ree)

        # Move to goal
        goal_pose_cam_ree = np.array([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z
        ])
        self.psm_control.controlPoseReeInCam(PSM, goal_pose_cam_ree)

        # Set final gripper state
        if request.primitive == CAPTURE:
            self.gripper_capture()
        elif request.primitive == GRASP:
            self.psm_control.closeGripper(PSM)
            time.sleep(0.5)
            self.psm_control.openGripper(PSM)
        
        return GraspResponse()


    def gripper_capture(self):
        self.psm_control._setGripper(
                self.psm_control.set_gripper2_pub, 
                end_pos=20, 
            )
        time.sleep(1.)

if __name__ == '__main__': 
    ros_dvrk = ROSdVRK( # Calls "init_node" internally
        control=True, 
        joints=True, 
    )
    psm_control = PsmControl(ros_dvrk)
    node = ExecGraspNode(psm_control)
    service = rospy.Service('exec_grasp_node/grasp', Grasp, node.execute_grasp)
    rospy.spin()
