import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import scipy.stats
import cv2
import sys
import csv

from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from curve_fit import curve_fit

sys.path.append("/Users/neelay/ARClabXtra/thread_reconstruction/src/Bo_Lu")
from Bo_Lu.pixel_ordering import order_pixels
from Bo_Lu.ssp_reconstruction import ssp_reconstruction

SIMULATION = False
ERODE = False
COMPARE = False
STORE = False

def fit_eval(img1, img2, calib, left_start=None, right_start=None, gt_tck=None):
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

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY)

    if not SIMULATION:
        map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
        map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

        # plt.figure(1)
        # plt.imshow(img1, cmap="gray")
        # plt.figure(2)
        # plt.imshow(img2, cmap="gray")
        # plt.show()
        img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
        img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
        # plt.figure(1)
        # plt.imshow(img1, cmap="gray")
        # plt.figure(2)
        # plt.imshow(img2, cmap="gray")
        # plt.show()
        # return

        img1 = np.where(img1==0, 255, img1)
        img2 = np.where(img2==0, 255, img2)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Our method
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    final_tck = curve_fit(img1, img_3D, keypoints, grow_paths, order)
    final_tck.c = change_coords(final_tck.c, P1[:, :3])


    final_spline = final_tck(np.linspace(final_tck.t[0], final_tck.t[-1], 150))
    if gt_tck is not None:
        gt_spline = gt_tck(np.linspace(0, 1, 150))

        ours_len, gt_len, diff = length_error(final_tck, gt_tck)
        print("Lengths: ours %f, gt %f, diff %f" % (ours_len, gt_len, diff))

        num_eval_pts = int(gt_len*10)
        errors1, spots1, e_mean1, e_max1 = curve_error(final_tck, gt_tck, num_eval_pts)
        errors2, spots2, e_mean2, e_max2 = curve_error(gt_tck, final_tck, num_eval_pts)
        e_mean = (e_mean1 + e_mean2)/2
        e_max = max(e_max1, e_max2)
        print("Curve error: mean %f, max %f" % (e_mean, e_max))
    else:
        num_eval_pts = 200
        left, left_max = reprojection_error(final_tck, img1, P1, num_eval_pts)
        right, right_max = reprojection_error(final_tck, img2, P2, num_eval_pts)
        print("Reprojection error: mean left %f, max left %f, mean right %f, max right %f" \
            % (left, left_max, right, right_max))

    plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.tick_params(labelsize=8)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_zlabel("$z$")
    ax1.plot(
        final_spline[:, 0],
        final_spline[:, 1],
        final_spline[:, 2],
        c="r",
        label="Our Result"
    )
    np.save("spline174.npy", final_spline)
    if gt_tck is not None:
        ax1.plot(
            gt_spline[:, 0],
            gt_spline[:, 1],
            gt_spline[:, 2],
            label="Ground Truth",
            c="g")
        ax1.legend()
        # gt_pts = np.concatenate((gt_tck(np.linspace(0, 1, num_eval_pts)), gt_tck(spots2)))
        # ours_pts = np.concatenate((final_tck(spots1), final_tck(np.linspace(final_tck.t[0], final_tck.t[-1], num_eval_pts))))
        # ax1.scatter(
        #     gt_pts[:, 0],
        #     gt_pts[:, 1],
        #     gt_pts[:, 2]
        # )
        # ax1.scatter(
        #     ours_pts[:, 0],
        #     ours_pts[:, 1],
        #     ours_pts[:, 2]
        # )
    set_axes_equal(ax1)

    # Bo Lu's method
    if COMPARE:
        ord1, ord2, steps1, steps2 = order_pixels(img1, img2, left_start, right_start)
        X_cloud = ssp_reconstruction(ord1, ord2, steps1, steps2, calib, folder_num, file_num)
        ord1 = ord1.T
        ord2 = ord2.T

        plt.figure(2)
        ax2 = plt.axes(projection='3d')
        min_len = min(ord1.shape[0], ord2.shape[0])
        X_cloud = X_cloud.T
        for row in X_cloud:
            row_cat = np.stack((ord1[:min_len, 0], ord1[:min_len, 1], row), axis=1)
            rescaled = change_coords(row_cat, P1[:, :3])
            ax2.scatter(rescaled[:, 0], rescaled[:, 1], rescaled[:, 2])
        ax2.plot(
            gt_spline[:, 0],
            gt_spline[:, 1],
            gt_spline[:, 2],
            c="g")
        set_axes_equal(ax2)
    if not STORE:
        plt.show()
    if SIMULATION:
        return ours_len, gt_len, diff, e_mean, e_max
    else:
        return left, left_max, right, right_max

def length_error(ours, gt):
    ours_der = ours.derivative()
    gt_der = gt.derivative()

    def integrand(u, dspline):
        return np.linalg.norm(dspline(u))
    
    ours_len = scipy.integrate.quad(integrand, ours.t[0], ours.t[-1], args=(ours_der))[0]
    gt_len = scipy.integrate.quad(integrand, gt.t[0], gt.t[-1], args=(gt_der))[0]
    return ours_len, gt_len, ours_len - gt_len

def curve_error(ours, gt, num_eval_pts):
    def objective(u, gt_pt):
        return np.linalg.norm(gt_pt - ours(u))
    
    # Find direction of ordering
    to_start = np.linalg.norm(gt(gt.t[0]) - ours(ours.t[0]))
    to_end = np.linalg.norm(gt(gt.t[-1]) - ours(ours.t[0]))
    aligned = True if to_start < to_end else False
    
    errors = np.zeros(num_eval_pts)
    spots = np.zeros(num_eval_pts)
    gt_pts = gt(np.linspace(gt.t[0], gt.t[-1], num_eval_pts))
    slider = ours.t[0] if aligned else ours.t[-1]
    for i, gt_pt in enumerate(gt_pts):
        bounds = [(slider, ours.t[-1]) if aligned else (ours.t[0], slider)]
        res1 = scipy.optimize.shgo(
            objective,
            bounds=bounds,
            # method="bounded",
            args=(gt_pt,)
        )
        res2 = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
            # method="bounded",
            args=(gt_pt,)
        )
        if objective(res1.x, gt_pt) < objective(res2.x, gt_pt):
            best = res1.x
        else:
            best = res2.x
        # slider = best
        spots[i] = best
        errors[i] = objective(best, gt_pt)
    return errors, spots, np.mean(errors), np.max(errors)

def reprojection_error(ours, img, P, num_eval_pts):
    segpix = np.argwhere(img<=250)
    u = np.linspace(ours.t[0], ours.t[-1], num_eval_pts)
    pts = ours(u)
    aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    proj_pts = (P @ aug_pts.T).T
    proj_pts /= proj_pts[:, 2:].copy() + 1e-7
    # plt.imshow(img, cmap="gray")
    # plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r")
    # plt.show()
    pixs = proj_pts[:, :2]
    pixs[:, 0], pixs[:, 1] = pixs[:, 1].copy(), pixs[:, 0].copy()
    errors = np.zeros(pts.shape[0])
    for i, pix in enumerate(pixs):
        pix = np.expand_dims(pix, 0)
        diffs = np.linalg.norm(pix - segpix, axis=1)
        errors[i] = np.min(diffs)
    return np.mean(errors), np.max(errors)


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
        # folder_num = 9
        # file_num = 3
        data = []
        header = ["file", "ours_len", "gt_len", "diff", "e_mean", "e_max"]
        footer = ["Failed"]
        for folder_num in range(1,2):
            for file_num in range(1,2):
                if folder_num < 5:
                    fileb = "../Blender_imgs/blend%d/blend%d_%d.jpg" % (folder_num, folder_num, file_num)
                    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
                else:
                    fileb = "../Blender_imgs/blend%d/blend%d_%d.png" % (folder_num, folder_num, file_num)
                    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration_new.yaml"
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

                left_start = None
                right_start = None
                if COMPARE:
                    left_starts = np.load("../Blender_imgs/blend%d/left%d.npy" % (folder_num, folder_num))
                    right_starts = np.load("../Blender_imgs/blend%d/right%d.npy" % (folder_num, folder_num))
                    left_start = left_starts[file_num-1]
                    right_start = right_starts[file_num-1]

                gt_b = np.load("/Users/neelay/ARClabXtra/Blender_imgs/blend%d/blend%d_%d.npy" % (folder_num, folder_num, file_num))
                cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml", cv2.FILE_STORAGE_READ)
                gk = 3
                gt_knots = np.concatenate(
                    (np.repeat(0, gk),
                    np.linspace(0, 1, gt_b.shape[0]-gk+1),
                    np.repeat(1, gk))
                )
                gt_tck = interp.BSpline(gt_knots, gt_b, gk)
                gt_spline = gt_tck(np.linspace(0, 1, 150))

                try:
                    out = list(fit_eval(img1, img2, calib, left_start, right_start, gt_tck))
                    out = ["%d_%d" % (folder_num, file_num)] + out
                    data.append(out)
                except:
                    footer.append("%d_%d" % (folder_num, file_num))

        if STORE:
            with open("results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)

                
    else:
        # folder_num = 2
        # file_num = 187
        files = [(1, 99), (1, 119), (2, 59), (2, 72), (2, 116), \
            (2, 149), (2, 159), (2,174), (2, 187), (2, 209)]
        data = []
        header = ["file", "left mean err", "left max error", "right mean err", "right max err"]
        footer = ["Failed"]
        for folder_num, file_num in files[-1:]:
            file1 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_left_%d_final.png" % (folder_num, folder_num, file_num)
            file2 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_right_%d_final.png" % (folder_num, folder_num, file_num)
            # TODO convert back if needed
            calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
            img1 = cv2.imread(file1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(file2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # plt.figure(1)
            # plt.imshow(img1, cmap="gray")
            # plt.figure(2)
            # plt.imshow(img2, cmap="gray")
            # plt.show()

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
            # try:
            out = list(fit_eval(img1, img2, calib, left_starts, right_starts))
            out = ["%d_%d" % (folder_num, file_num)] + out
            data.append(out)
            # except:
            #     footer.append("%d_%d" % (folder_num, file_num))
        if STORE:
            with open("results_real.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)