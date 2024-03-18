#! /home/autosurg/anaconda3/envs/thread_ros/bin/python

import rospy
from geometry_msgs.msg import PoseStamped

import numpy as np

from read_dvrk_msg.ros_dvrk import ROSdVRK
from psm_control.psm_control import PsmControl


class GraspThreadNode:
    def __init__(self, psm_control):
        self.psm_control = psm_control
        self.grasp_sub = rospy.Subscriber("/thread_reconstr/grasp_pose", PoseStamped, self.execute_grasp, queue_size=1)
    
    def execute_grasp(self, pose):
        goal_pose_cam_ree = np.array([ #TODO double check order
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z
        ])
        self.psm_controller.openGripper(1)
        self.psm_controller.controlPoseReeInCam(1, goal_pose_cam_ree)
        self.psm_controller.closeGripper(1)

if __name__ == '__main__': 
    ros_dvrk = ROSdVRK( # Calls "init_node" internally
        control=True, 
        joints=True, 
    )
    psm_control = PsmControl(ros_dvrk)
    node = GraspThreadNode(psm_control)
    rospy.spin()
