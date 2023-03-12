import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import cv2
import os

from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from curve_fit import curve_fit

# Set to "True" if evaluating on simulated dataset
#   "False" for real dataset
SIMULATION = True
# Set to "True" to store results in csv file without visualization
#   "False" to visualize results without storing in csv file
STORE = False

DATA = os.path.dirname(
        os.path.abspath(__file__)
    ) + "/../data/"
SIMDATA = DATA + "simulated/"
REALDATA = DATA + "real/"

# Evaluate the fit of our reconstruction
# img1 and img2 are the stereo images
# calib is the camera calibration file
# gt_tck is the ground truth spline (only provided for simulated curves)
def fit_eval(img1, img2, calib, gt_tck=None):
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

    # Stereorectify the images
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY)

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Perform reconstruction
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    final_tck = curve_fit(img1, img_3D, keypoints, grow_paths, order)
    final_tck.c = change_coords(final_tck.c, P1[:, :3])
    final_spline = final_tck(np.linspace(final_tck.t[0], final_tck.t[-1], 150))

    # Evaluate reconstruction accuracy
    if SIMULATION:
        gt_spline = gt_tck(np.linspace(0, 1, 150))

        # Length error
        ours_len, gt_len, diff = length_error(final_tck, gt_tck)
        print("Lengths: ours %f, gt %f, diff %f" % (ours_len, gt_len, diff))

        num_eval_pts = int(gt_len*10)
        errors1, spots1, e_mean1, e_max1 = curve_error(final_tck, gt_tck, num_eval_pts)
        errors2, spots2, e_mean2, e_max2 = curve_error(gt_tck, final_tck, num_eval_pts)
        # Mean and max curve error
        e_mean = (e_mean1 + e_mean2)/2
        e_max = max(e_max1, e_max2)
        print("Curve error: mean %f, max %f" % (e_mean, e_max))
    else:
        num_eval_pts = 200
        # Reprojection error
        left, left_max = reprojection_error(final_tck, img1, P1, num_eval_pts)
        right, right_max = reprojection_error(final_tck, img2, P2, num_eval_pts)
        print("Reprojection error: mean left %f, max left %f, mean right %f, max right %f" \
            % (left, left_max, right, right_max))

    if not STORE:
        # Visualize the result
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
        # Also show ground truth if available
        if gt_tck is not None:
            ax1.plot(
                gt_spline[:, 0],
                gt_spline[:, 1],
                gt_spline[:, 2],
                label="Ground Truth",
                c="g")
            ax1.legend()
        set_axes_equal(ax1)
        plt.show()
    if SIMULATION:
        return ours_len, gt_len, diff, e_mean, e_max
    else:
        return left, left_max, right, right_max

# Compute length error
# ours is computed reconstruction
# gt is ground truth
def length_error(ours, gt):
    ours_der = ours.derivative()
    gt_der = gt.derivative()

    def integrand(u, dspline):
        return np.linalg.norm(dspline(u))
    
    ours_len = scipy.integrate.quad(integrand, ours.t[0], ours.t[-1], args=(ours_der))[0]
    gt_len = scipy.integrate.quad(integrand, gt.t[0], gt.t[-1], args=(gt_der))[0]
    return ours_len, gt_len, ours_len - gt_len

# Compute curve error, averaged over curve
# ours is computed reconstruction
# gt is ground truth
# num_eval_pts is number of points at which curve error is evaluated
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
            args=(gt_pt,)
        )
        res2 = scipy.optimize.differential_evolution(
            objective,
            bounds=bounds,
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

# Compute reprojection error, averaged over curve
# ours is computed reconstruction
# img is image to project onto
# P is projection matrix
# num_eval_pts is number of points at which reprojection error is evaluated
def reprojection_error(ours, img, P, num_eval_pts):
    segpix = np.argwhere(img<=250)
    u = np.linspace(ours.t[0], ours.t[-1], num_eval_pts)
    pts = ours(u)
    aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    proj_pts = (P @ aug_pts.T).T
    proj_pts /= proj_pts[:, 2:].copy() + 1e-7
    
    pixs = proj_pts[:, :2]
    pixs[:, 0], pixs[:, 1] = pixs[:, 1].copy(), pixs[:, 0].copy()
    errors = np.zeros(pts.shape[0])
    for i, pix in enumerate(pixs):
        pix = np.expand_dims(pix, 0)
        diffs = np.linalg.norm(pix - segpix, axis=1)
        errors[i] = np.min(diffs)
    return np.mean(errors), np.max(errors)

# Convert from image-depth (u, v, d) frame to camera frame (x, y, z)
def change_coords(pts, K1):
    pts[:, 0], pts[:, 1] = pts[:, 1].copy(), pts[:, 0].copy()
    depths = pts[:, 2:].copy()
    pts[:, 2] = np.ones(pts.shape[0])
    pts_c = depths * (np.linalg.inv(K1) @ pts.copy().T).T
    return pts_c

"""
Helper function for preventing distortion when displaying reconstruction
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
    # Run reconstruction on datasets
    # Simulated Dataset
    if SIMULATION:
        data = []
        header = ["file", "ours_len", "gt_len", "diff", "e_mean", "e_max"]
        footer = ["Failed"]
        for folder_num in range(1,11):
            for file_num in range(1,5):
                # Choose correct calibration matrix
                if folder_num < 5:
                    fileb = SIMDATA + "blend%d/blend%d_%d.jpg" % (folder_num, folder_num, file_num)
                    calib = SIMDATA + "blend_calibration.yaml"
                else:
                    fileb = SIMDATA + "blend%d/blend%d_%d.png" % (folder_num, folder_num, file_num)
                    calib = SIMDATA + "blend_calibration_new.yaml"
                
                # Extract and color segment left and right images
                imgb = cv2.imread(fileb)
                imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
                img1 = imgb[:, :640]
                img2 = imgb[:, 640:]
                img1 = np.where(img1>=200, 255, img1)
                img2 = np.where(img2>=200, 255, img2)

                gt_b = np.load(SIMDATA + "blend%d/blend%d_%d.npy" % (folder_num, folder_num, file_num))
                gk = 3
                gt_knots = np.concatenate(
                    (np.repeat(0, gk),
                    np.linspace(0, 1, gt_b.shape[0]-gk+1),
                    np.repeat(1, gk))
                )
                gt_tck = interp.BSpline(gt_knots, gt_b, gk)

                # Can use try catch to avoid errors while storing results
                # try:
                out = list(fit_eval(img1, img2, calib, gt_tck))
                out = ["%d_%d" % (folder_num, file_num)] + out
                data.append(out)
                # except:
                #     footer.append("%d_%d" % (folder_num, file_num))
        # Store results conveniently in csv file
        if STORE:
            with open(SIMDATA + "results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)
    # Real dataset        
    else:
        files = [(1, 99), (1, 119), (2, 59), (2, 72), (2, 116), \
            (2, 149), (2, 159), (2,174), (2, 187), (2, 209)]
        data = []
        header = ["file", "left mean err", "left max error", "right mean err", "right max err"]
        footer = ["Failed"]
        for folder_num, file_num in files:
            file1 = REALDATA + "thread_%d_seg/thread%d_left_%d_final.png" % (folder_num, folder_num, file_num)
            file2 = REALDATA + "thread_%d_seg/thread%d_right_%d_final.png" % (folder_num, folder_num, file_num)
            calib = REALDATA + "camera_calibration_sarah.yaml"
            img1 = cv2.imread(file1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(file2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Can use try catch to avoid errors while storing results
            # try:
            out = list(fit_eval(img1, img2, calib))
            out = ["%d_%d" % (folder_num, file_num)] + out
            data.append(out)
            # except:
            footer.append("%d_%d" % (folder_num, file_num))
        if STORE:
            with open(REALDATA + "results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)