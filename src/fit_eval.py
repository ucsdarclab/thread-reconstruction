import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import scipy.stats
import cv2
import sys

from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from curve_fit import curve_fit

sys.path.append("/Users/neelay/ARClabXtra/thread_reconstruction/src/Bo_Lu")
from Bo_Lu.pixel_ordering import order_pixels
from Bo_Lu.ssp_reconstruction import ssp_reconstruction

SIMULATION = False
ERODE = False

def fit_eval(img1, img2, calib, left_start, right_start, gt_tck=None):
    # Read in camera matrix
    cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    
    # Our method
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, calib)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    final_tck = curve_fit(img1, img_3D, keypoints, grow_paths, order)
    final_tck.c = change_coords(final_tck.c, K1)

    # Bo Lu's method
    # ord1, ord2, steps1, steps2 = order_pixels(img1, img2, left_start, right_start)
    # X_cloud = ssp_reconstruction(ord1, ord2, steps1, steps2, calib, folder_num, file_num)
    # ord1 = ord1.T
    # ord2 = ord2.T


    final_spline = final_tck(np.linspace(final_tck.t[0], final_tck.t[-1], 150))
    if gt_tck is not None:
        gt_spline = gt_tck(np.linspace(0, 1, 150))

    plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.plot(
        final_spline[:, 0],
        final_spline[:, 1],
        final_spline[:, 2]
    )
    if gt_tck is not None:
        ax1.plot(
            gt_spline[:, 0],
            gt_spline[:, 1],
            gt_spline[:, 2],
            c="g")
    set_axes_equal(ax1)

    # plt.figure(2)
    # ax2 = plt.axes(projection='3d')
    # min_len = min(ord1.shape[0], ord2.shape[0])
    # X_cloud = X_cloud.T
    # for row in X_cloud:
    #     row_cat = np.stack((ord1[:min_len, 0], ord1[:min_len, 1], row), axis=1)
    #     rescaled = change_coords(row_cat, K1)
    #     ax2.scatter(rescaled[:, 0], rescaled[:, 1], rescaled[:, 2])
    # ax2.plot(
    #     gt_spline[:, 0],
    #     gt_spline[:, 1],
    #     gt_spline[:, 2],
    #     c="g")
    # set_axes_equal(ax2)
    plt.show()

def change_coords(pts, K1):
    pts[:, 0], pts[:, 1] = pts[:, 1].copy(), pts[:, 0].copy()
    depths = pts[:, 2:].copy()
    pts[:, 2] = np.ones(pts.shape[0])
    pts_c = depths * (np.linalg.inv(K1) @ pts.copy().T).T
    return pts_c

"""
Source code here: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
"""
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == "__main__":
    if SIMULATION:
        folder_num = 1
        file_num = 4
        fileb = "../Blender_imgs/blend%d/blend%d_%d.jpg" % (folder_num, folder_num, file_num)
        calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
        imgb = cv2.imread(fileb)
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
        img1 = imgb[:, :640]
        img2 = imgb[:, 640:]
        img1 = np.where(img1>=200, 255, img1)
        img2 = np.where(img2>=200, 255, img2)

        if ERODE:
            img1_dig = np.where(img1==255, 0, 1).astype("uint8")
            img2_dig = np.where(img2==255, 0, 1).astype("uint8")
            kernel = np.ones((2, 2))
            img1_dig = cv2.erode(img1_dig, kernel, iterations=1)
            img2_dig = cv2.erode(img2_dig, kernel, iterations=1)
            img1 = np.where(img1_dig==1, img1, 255)
            img2 = np.where(img2_dig==1, img2, 255)

        left_starts = np.load("../Blender_imgs/blend%d/left%d.npy" % (folder_num, folder_num))
        right_starts = np.load("../Blender_imgs/blend%d/right%d.npy" % (folder_num, folder_num))

        gt_b = np.load("/Users/neelay/ARClabXtra/Blender_imgs/blend%d/blend%d_%d.npy" % (folder_num, folder_num, file_num))
        cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml", cv2.FILE_STORAGE_READ)
        # K1 = cv_file.getNode("K1").mat()
        # m2pix = K1[0, 0] / 50e-3
        # gt_pix = np.matmul(K1, gt_b.T).T
        # gt_b[:, :2] = gt_pix[:, :2] / gt_pix[:, 2:]
        gk = 3
        gt_knots = np.concatenate(
            (np.repeat(0, gk),
            np.linspace(0, 1, gt_b.shape[0]-gk+1),
            np.repeat(1, gk))
        )
        gt_tck = interp.BSpline(gt_knots, gt_b, gk)
        gt_spline = gt_tck(np.linspace(0, 1, 150))

        # ax = plt.axes(projection='3d')
        # ax.plot(
        #     gt_spline[:, 0],
        #     gt_spline[:, 1],
        #     gt_spline[:, 2],
        #     c="g")
        # set_axes_equal(ax)
        # plt.show()
        fit_eval(img1, img2, calib, left_starts[file_num-1], right_starts[file_num-1], gt_tck)
    else:
        folder_num = 2
        file_num = 159
        file1 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_left_%d_final.jpg" % (folder_num, folder_num, file_num)
        file2 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_right_%d_final.jpg" % (folder_num, folder_num, file_num)
        calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration.yaml"
        img1 = cv2.imread(file1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(file2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if ERODE:
            img1_dig = np.where(img1==255, 0, 1).astype("uint8")
            img2_dig = np.where(img2==255, 0, 1).astype("uint8")
            kernel = np.ones((2, 2))
            img1_dig = cv2.erode(img1_dig, kernel, iterations=1)
            img2_dig = cv2.erode(img2_dig, kernel, iterations=1)
            img1 = np.where(img1_dig==1, img1, 255)
            img2 = np.where(img2_dig==1, img2, 255)

        #TODO implement
        left_starts = np.zeros((2, 2))#np.load("../Blender_imgs/blend%d/left%d.npy" % (folder_num, folder_num))
        right_starts = np.zeros((2, 2))#np.load("../Blender_imgs/blend%d/right%d.npy" % (folder_num, folder_num))

        gt_tck = None

        # ax = plt.axes(projection='3d')
        # ax.plot(
        #     gt_spline[:, 0],
        #     gt_spline[:, 1],
        #     gt_spline[:, 2],
        #     c="g")
        # set_axes_equal(ax)
        # plt.show()

        fit_eval(img1, img2, calib, left_starts, right_starts)