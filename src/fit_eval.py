import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.interpolate as interp
import cv2
import os
import torch

from segmenter import SAMSegmenter, UNetSegmenter
from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from optim import optim
from utils import *

"""
img1: RGB left camera image (np array)
img2: RGB right camera image (np array)
calib: filename for camera calibration file (string)
segmenter: segmentation object (see segmenter.py)
"""
def fit_eval(img1, img2, calib, segmenter):
    # Read in camera matrix
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

    # Rectify image and store necessary matrices
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
    cam2img = P1[:,:-1]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, new_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, new_size, cv2.CV_16SC2)
    img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    # Segment stereo images
    mask1 = segmenter.segmentation(img1)
    mask2 = segmenter.segmentation(img2)
    stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
    img1 = np.where(stack_mask1>0, img1, 0)
    stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
    img2 = np.where(stack_mask2>0, img2, 0)
    
    # Convert from btyes to float
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Perform reconstruction
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    spline, reliability = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, cam2img, P1, P2)
    return spline, reliability