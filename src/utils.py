import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import scipy.optimize
import scipy.integrate

def change_coords(pts, cam2img):
    pts[:, 0], pts[:, 1] = pts[:, 1].copy(), pts[:, 0].copy()
    depths = pts[:, 2:].copy()
    pts[:, 2] = np.ones(pts.shape[0])
    pts_c = depths * (np.linalg.inv(cam2img) @ pts.copy().T).T
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

"""
Source code here: https://github.com/opencv/opencv/issues/22120
"""
def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction // 2
    return P

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

def reprojection_error(ours, mask, P, num_eval_pts):
    segpix = np.argwhere(mask>0)
    u = np.linspace(ours.t[0], ours.t[-1], num_eval_pts)
    pts = ours(u)
    aug_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    proj_pts = (P @ aug_pts.T).T
    proj_pts /= proj_pts[:, 2:].copy() + 1e-7
    plt.imshow(mask, cmap="gray")
    plt.scatter(proj_pts[:, 0], proj_pts[:, 1], c="r")
    plt.show()
    pixs = proj_pts[:, :2]
    pixs[:, 0], pixs[:, 1] = pixs[:, 1].copy(), pixs[:, 0].copy()
    errors = np.zeros(pts.shape[0])
    for i, pix in enumerate(pixs):
        pix = np.expand_dims(pix, 0)
        diffs = np.linalg.norm(pix - segpix, axis=1)
        errors[i] = np.min(diffs)
    return np.mean(errors), np.max(errors)