import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import scipy.stats
import cv2
from stereo_matching import stereo_match
from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
import sys

def curve_fit(img1, img_3D, keypoints, grow_paths, order):
    # Gather more points between keypoints to get better data for curve initialization
    init_pts = []
    segpix1 = np.argwhere(img1<250)

    size_thresh = segpix1.shape[0] // 100
    interval_floor = size_thresh // 2
    keypoint_idxs = []
    for key_ord, key_id in enumerate(order[:-1]):
        # Find segmented points between keypoints
        keypoint_idxs.append(len(init_pts))
        init_pts.append(keypoints[key_id])
        curr_growth = grow_paths[key_id]
        next_id = order[key_ord+1]
        next_growth = grow_paths[next_id]
        btwn_pts = curr_growth.intersection(next_growth)

        # gather extra points if keypoint distance is large
        if len(btwn_pts) > size_thresh:
            btwn_pts = np.array(list(btwn_pts))
            btwn_depths = img_3D[btwn_pts[:, 0], btwn_pts[:, 1], 2]
            num_samples = btwn_pts.shape[0] // size_thresh
            # remove outliers
            quartiles = np.percentile(btwn_depths, [25, 75])
            iqr = quartiles[1] - quartiles[0]
            low_clip = quartiles[0]-1.5*iqr < btwn_depths
            up_clip = btwn_depths < quartiles[1]+1.5*iqr
            mask = np.logical_and(low_clip, up_clip)
            mask_idxs = np.squeeze(np.argwhere(mask))
            if mask_idxs.shape[0] < num_samples:
                continue
            filtered_pix = btwn_pts[mask_idxs]
            filtered_depths = btwn_depths[mask_idxs]
            filtered_pts = np.concatenate((filtered_pix, np.expand_dims(filtered_depths, 1)), axis=1)
            
            # project filtered points onto 2D line between keypoints
            p1 = keypoints[key_id, :2]
            p2 = keypoints[next_id, :2]
            p1p2 = p2 - p1
            p1pt = filtered_pix - np.expand_dims(p1, 0)
            proj = np.dot(p1pt, p1p2) / np.linalg.norm(p1p2)

            # Choose evenly spaced points, based on projections
            pt2ord = np.argsort(proj)
            floor = interval_floor if interval_floor<np.max(proj) else np.min(proj)
            p1d = keypoints[key_id, 2]
            p2d = keypoints[next_id, 2]
            p1p2d = p2d - p1d
            intervals = np.linspace(interval_floor, np.max(proj), num_samples)
            int_idx = 0
            for pt_idx in pt2ord:
                if int_idx >= num_samples or \
                    (filtered_pix[pt_idx] == keypoints[next_id, :2]).all():
                    break
                if proj[pt_idx] >= intervals[int_idx]:
                    # Use linearly interpolated depth between keypoints
                    init_pts.append(filtered_pts[pt_idx])
                    int_idx += 1
    keypoint_idxs.append(len(init_pts))
    init_pts.append(keypoints[order[-1]])
    init_pts = np.array(init_pts)

    # Construct bounds
    lower = np.zeros((len(order), 3))
    upper = np.zeros((len(order), 3))
    fit_rad = keypoints.shape[0] // 10
    bound_rads = []
    bound_thresh = 1e-5
    lowest_nonzero = np.inf
    for key_ord, key_id in enumerate(order):
        # Fit line to range of points
        start = max(key_ord-fit_rad, 0)
        end = min(key_ord+fit_rad, len(order)-1)
        if start == 0:
            end += fit_rad//2
        elif end == len(order)-1:
            start -= fit_rad//2
        x = np.arange(keypoint_idxs[start], keypoint_idxs[end]+1)
        data = init_pts[x, 2]
        slope, intercept, *_ = scipy.stats.linregress(x, data)

        # Construct bound radius from current point
        line_pts = slope * x + intercept
        curr_line_pt = slope * keypoint_idxs[key_ord] + intercept
        line_std = np.mean((data - line_pts)**2) ** (1/2)
        bound_rad = np.abs(keypoints[key_id, 2] - curr_line_pt) * 1.5
        bound_rad = max(bound_rad, line_std)
        bound_rads.append(bound_rad)
        if bound_rad > bound_thresh:
            lowest_nonzero = min(lowest_nonzero, bound_rad)
    for key_ord, (key_id, bound_rad) in enumerate(zip(order, bound_rads)):
        if bound_rad <= bound_thresh:
            bound_rad = lowest_nonzero
        lower[key_ord] = keypoints[key_id] - np.array([[0, 0, bound_rad]])
        upper[key_ord] = keypoints[key_id] + np.array([[0, 0, bound_rad]])

    # ax = plt.axes(projection="3d")
    # ax.set_xlim(0, 480)
    # ax.set_ylim(0, 640)
    # ax.set_zlim(5, 15)
    # ax.plot(
    #     gt_spline[:, 1],
    #     gt_spline[:, 0],
    #     gt_spline[:, 2],
    #     c="g")
    # ax.scatter(
    #     keypoints[:, 0],
    #     keypoints[:, 1],
    #     keypoints[:, 2],
    #     s=10, c="r"
    # )
    # ax.plot(lower[:, 0], lower[:, 1], lower[:, 2], c="orange")
    # ax.plot(upper[:, 0], upper[:, 1], upper[:, 2], c="orange")
    # u = np.arange(init_pts.shape[0])
    # plt.plot(u, gt_tck(np.linspace(0, 1, u.shape[0]))[:, 2], c="g")
    # plt.scatter(keypoint_idxs, keypoints[order, 2], c="r")
    # plt.plot(keypoint_idxs, lower[:, 2], c="turquoise")
    # plt.plot(keypoint_idxs, upper[:, 2], c="turquoise")
    # plt.show()
    # return

    "Set up optimization"
    # initialize 3D spline
    d = 4
    num_ctrl = 15#init_pts.shape[0] // (2*d)
    print(num_ctrl)

    dists = np.linalg.norm(init_pts[1:, :2] - init_pts[:-1, :2], axis=1)
    dists /= np.sum(dists)
    u = np.zeros(init_pts.shape[0])
    u[1:] = np.cumsum(dists) * dists.shape[0]
    u[-1] = dists.shape[0]
    # u = np.arange(0, init_pts.shape[0])

    key_weight = init_pts.shape[0] / keypoints.shape[0]
    w = np.ones_like(u)
    w[keypoint_idxs] = key_weight
    knots = np.concatenate(
        (np.repeat(0, d),
        np.linspace(0, u[-1], num_ctrl),
        np.repeat(u[-1], d))
    )

    low_constr = interp.interp1d(u[keypoint_idxs], lower[:, 2])
    high_constr = interp.interp1d(u[keypoint_idxs], upper[:, 2])
    center = (low_constr(u) + high_constr(u))/2
    init_pts[:, 2] = center

    # Get endpoint trends
    endpts = []
    slopes = []
    endpt_ids = [order[0], order[-1]]
    endpt_ords = [0, len(order)-1]
    for key_ord, key_id in zip(endpt_ords, endpt_ids):
        # Fit line to range of points
        start = max(key_ord-fit_rad, 0)
        end = min(key_ord+fit_rad, len(order)-1)
        if start == 0:
            end += fit_rad//2
        elif end == len(order)-1:
            start -= fit_rad//2
        x = np.arange(keypoint_idxs[start], keypoint_idxs[end]+1)
        data = init_pts[x, 2]
        slope, intercept, *_ = scipy.stats.linregress(x, data)
        if key_ord == 0:
            endpts.append(intercept)
            slopes.append(slope)
        elif key_ord == len(order) - 1:
            endpts.append(slope*x[-1] + intercept)
            slopes.append(slope)

    tck, *_ = interp.splprep(init_pts.T, w=w, u=u, k=d, task=-1, t=knots)

    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    tck = interp.BSpline(t, c, k)
    init_spline = tck(np.linspace(0, u[-1], 150))

    # plt.imshow(img1, cmap="gray")
    # plt.plot(init_pts[:, 1], init_pts[:, 0])
    # plt.plot(init_spline[:, 1], init_spline[:, 0])
    # plt.scatter(init_pts[:, 1], init_pts[:, 0], c="r")
    # plt.show()
    # return

    # ax = plt.axes(projection="3d")
    # ax.set_xlim(0, 480)
    # ax.set_ylim(0, 640)
    # ax.set_zlim(5, 15)
    # ax.plot(
    #     gt_spline[:, 1],
    #     gt_spline[:, 0],
    #     gt_spline[:, 2],
    #     c="g")
    # ax.plot(
    #     init_spline[:, 0],
    #     init_spline[:, 1],
    #     init_spline[:, 2],
    #     c="b")
    # plt.show()
    # plt.plot(u, tck(u)[:, 2], c="b")
    # plt.scatter(keypoint_idxs, keypoints[order, 2], c="r")
    # plt.plot(keypoint_idxs, lower[:, 2], c="turquoise")
    # plt.plot(keypoint_idxs, upper[:, 2], c="turquoise")
    # plt.show()

    b = c[:, 2]
    spline_db, dspline_db, d2spline_db, d3spline_db = dspline_grads(b, knots, d)

    constraints = []
    for p in u:
        constraints.append(
            {
                "type":"ineq",
                "fun":lower_bound,
                "jac":lower_grad,
                "args":(knots, d, low_constr, p, spline_db)
            }
        )
        constraints.append(
            {
                "type":"ineq",
                "fun":upper_bound,
                "jac":upper_grad,
                "args":(knots, d, high_constr, p, spline_db)
            }
        )
    sides = [0, -1, 0, -1]
    constrs = [endpts[0], endpts[1], slopes[0], slopes[-1]]
    ders = [0, 0, 1, 1]
    for side, constr, der in zip(sides, constrs, ders):
        constraints.append(
            {
                "type":"eq",
                "fun":endpt_constr,
                # "jac":endpt_grad,
                "args": (knots, d, side, constr, der)
            }
        )
    
    
    final = scipy.optimize.minimize(
        objective,
        b,
        method = 'SLSQP',
        jac=gradient,
        args=(knots, d, dspline_db, d2spline_db, d3spline_db),
        constraints=constraints,
        options= {"maxiter" : 200}
    )
    print("success:", final.success)
    print("status:", final.status)
    print("message:", final.message)
    print("num iter:", final.nit)
    b = np.array([val for val in final.x])
    c[:, 2] = b
    tck = interp.BSpline(t, c, k)
    final_spline = tck(np.linspace(0, u[-1], 150))

    return tck

    # plt.figure(1)
    # ax1 = plt.axes(projection="3d")
    # ax1.set_xlim(0, 480)
    # ax1.set_ylim(0, 640)
    # # ax1.set_zlim(5, 15)
    # ax1.plot(
    #     gt_spline[:, 1],
    #     gt_spline[:, 0],
    #     gt_spline[:, 2],
    #     c="g")
    # ax1.plot(
    #     init_spline[:, 0],
    #     init_spline[:, 1],
    #     init_spline[:, 2],
    #     c="b")
    # plt.figure(2)
    ax2 = plt.axes(projection="3d")
    # ax2.set_xlim(0, 480)
    # ax2.set_ylim(0, 640)
    # ax2.set_zlim(5, 15)
    # ax2.plot(
    #     gt_spline[:, 1],
    #     gt_spline[:, 0],
    #     gt_spline[:, 2],
    #     c="g")
    ax2.plot(
        init_spline[:, 0],
        init_spline[:, 1],
        init_spline[:, 2],
        c="turquoise")
    # ax2.plot(
    #     init_pts[:, 0],
    #     init_pts[:, 1],
    #     init_pts[:, 2],
    #     c="turquoise")
    ax2.plot(
        final_spline[:, 0],
        final_spline[:, 1],
        final_spline[:, 2],
        c="b")
    ax2.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c="r")
    ax2.plot(lower[:, 0], lower[:, 1], lower[:, 2], c="orange")
    ax2.plot(upper[:, 0], upper[:, 1], upper[:, 2], c="orange")
    plt.show()


def objective(args, knots, d, grad1_spl, grad2_spl, grad3_spl):
    b = np.array([val for val in args])
    spline = interp.BSpline(knots, b, d)
    dspline = spline.derivative()
    d2spline = dspline.derivative()
    d3spline = d2spline.derivative()

    def integrand(x):
        ds = dspline(x)
        d2s = d2spline(x)
        d3s = d3spline(x)
        dKB = (d3s*(1+ds**2)**(3/2) - 3/2*(1+ds**2)**(1/2)*(2*ds)*d2s**2) \
            / (1+ds**2)**3
        return dKB**2 / (1+ds**2)**(1/2)
    
    u = np.linspace(knots[0], knots[-1], 100)
    curve = [integrand(x) for x in u]
    return scipy.integrate.simpson(curve, u)

# def start_constr(args, knots, d, start, der, A):
#     b = np.array([val for val in args])
#     tck = interp.BSpline(knots, b, d)
#     spline = interp.splev([start], tck, der)
#     return spline - A

# def end_constr(args, knots, d, end, der, B):
#     b = np.array([val for val in args])
#     tck = interp.BSpline(knots, b, d)
#     spline = interp.splev([end], tck, der)
#     return spline - B

def endpt_constr(args, knots, d, side, constr, der):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    spline = interp.splev([knots[side]], tck, der)
    return spline - constr

# def endpt_grad(args, side, der, constr):
#     grad = np.zeros(len(args))
#     grad[side] = 1
#     return grad

def lower_bound(args, knots, d, constr, x, grad):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    dist = tck(x) - constr(x)
    return dist

def lower_grad(args, knots, d, constr, x, grad):
    return [der(x) for der in grad]

def upper_bound(args, knots, d, constr, x, grad):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    dist = constr(x) - tck(x)
    return dist

def upper_grad(args, knots, d, constr, x, grad):
    return [-1*der(x) for der in grad]

def dspline_grads(b, knots, d):
    n = b.shape[0]
    # Organize useful coefficients conveniently
    Q1 = np.ones((n-1, n))
    for j in range(Q1.shape[1]):
        Q1[:, j] = d / (knots[d+1:-1] - knots[1:-1*(d+1)])
    Q2 = np.ones((n-2, n))
    for j in range(Q2.shape[1]):
        Q2[:, j] = (d-1) / (knots[d+1:-2] - knots[2:-1*(d+1)])
    Q3 = np.ones((n-3, n))
    for j in range(Q3.shape[1]):
        Q3[:, j] = (d-2) / (knots[d+1:-3] - knots[3:-1*(d+1)])

    J0 = np.eye(n)
    J1 = Q1 * (J0[1:] - J0[:-1])
    J2 = Q2 * (J1[1:] - J1[:-1])
    J3 = Q3 * (J2[1:] - J2[:-1])

    grad0 = []
    grad1 = []
    grad2 = []
    grad3 = []
    for j in range(n):
        tck0 = interp.BSpline(knots, J0[:, j], d)
        grad0.append(tck0)
        tck1 = interp.BSpline(knots[1:-1], J1[:, j], d-1)
        grad1.append(tck1)
        tck2 = interp.BSpline(knots[2:-2], J2[:, j], d-2)
        grad2.append(tck2)
        tck3 = interp.BSpline(knots[3:-3], J3[:, j], d-3)
        grad3.append(tck3)
        
    return grad0, grad1, grad2, grad3

def gradient(args, knots, d, grad1_spl, grad2_spl, grad3_spl):
    b = np.array([val for val in args])
    spline = interp.BSpline(knots, b, d)
    dspline = spline.derivative()
    d2spline = dspline.derivative()
    d3spline = d2spline.derivative()

    def integrand(x, j):
        ds = dspline(x)
        d2s = d2spline(x)
        d3s = d3spline(x)
        Gs = grad1_spl[j](x)#[der(x) for der in grad1_spl]
        G2s = grad2_spl[j](x)#[der(x) for der in grad2_spl]
        G3s = grad3_spl[j](x)#[der(x) for der in grad3_spl]

        dK = (d3s*(1+ds**2)**(3/2) - 3/2*(1+ds**2)**(1/2)*(2*ds)*d2s**2) \
            / (1+ds**2)**3
        dK_db = calc_dK_db(ds, d2s, d3s, Gs, G2s, G3s)

        res = (2*dK*dK_db*(1 + ds**2)**(1/2) \
            - (dK)**2 * (1/2)*(1+ds**2)**(-1/2) * 2*ds*Gs) \
            / (1+ds**2)
        return res
    
    def calc_dK_db(ds, d2s, d3s, Gs, G2s, G3s):
        top = d3s*(1 + ds**2) - 3*ds*(d2s)**2
        dtop_db = G3s*(1 + ds**2) + d3s*(2*ds)*Gs \
            - 3*(Gs*d2s**2 + ds*(2*d2s)*G2s)
        bottom = (1+ds**2)**(5/2)
        dbottom_db = 5/2*(1 + ds**2)**(3/2) * (2*ds) * Gs
        dK_db = (dtop_db*bottom - top*dbottom_db) / bottom**2
        return dK_db
    
    u = np.linspace(knots[0], knots[-1], 100)
    curves = [[integrand(x, j) for x in u] for j in range(b.shape[0])]
    return [scipy.integrate.simpson(curve, u) for curve in curves]


if __name__ == "__main__":
    # file1 = "../Sarah_imgs/thread_3_left_final.jpg"#sys.argv[1]
    # file2 = "../Sarah_imgs/thread_3_right_final.jpg"#sys.argv[2]
    # img1 = cv2.imread(file1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(file2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # img_3D, keypoints, grow_paths, order = keypt_selection(img1, img2)
    fileb = "../Blender_imgs/blend5/blend5_1.png"
    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
    imgb = cv2.imread(fileb)
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    img1 = imgb[:, :640]
    img2 = imgb[:, 640:]
    img1 = np.where(img1>=205, 255, img1)
    img2 = np.where(img2>=205, 255, img2)
    plt.figure(1)
    plt.imshow(img1, cmap="gray")
    plt.figure(2)
    plt.imshow(img2, cmap="gray")
    plt.show()
    assert False
    # test()
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, calib)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)

    curve_fit(img1, img_3D, keypoints, grow_paths, order)