import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from prob_cloud import prob_cloud
import scipy.interpolate as interp
import scipy.optimize
import scipy.integrate

def curve_fit(img1, img_3D, keypoints, grow_paths, order):
    # Gather more points between keypoints to get better data for curve initialization
    init_pts = []
    segpix1 = np.argwhere(img1<250)
    # TODO Base this off of max_size of clusters from prob_cloud
    size_thresh = 20
    interval_floor = 8
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
            filtered_pix = btwn_pts[mask_idxs]
            filtered_depths = btwn_depths[mask_idxs]
            filtered_pts = np.concatenate((filtered_pix, np.expand_dims(filtered_depths, 1)), axis=1)
            
            # project filtered points onto line between keypoints
            p1 = keypoints[key_id, :2]
            p2 = keypoints[next_id, :2]
            p1p2 = p2 - p1
            p1pt = filtered_pix - np.expand_dims(p1, 0)
            proj = np.dot(p1pt, p1p2) / np.linalg.norm(p1p2)

            # Choose evenly spaced points, based on projections
            pt2ord = np.argsort(proj)
            floor = interval_floor if interval_floor<np.max(proj) else np.min(proj)
            intervals = np.linspace(interval_floor, np.max(proj), num_samples)
            int_idx = 0
            for pt_idx in pt2ord:
                if int_idx >= num_samples:
                    break
                if proj[pt_idx] >= intervals[int_idx]:
                    init_pts.append(filtered_pts[pt_idx])
                    int_idx += 1
    keypoint_idxs.append(len(init_pts))
    init_pts.append(keypoints[order[-1]])
    init_pts = np.array(init_pts)
    
    # initialize 3D spline
    d = 5
    num_ctrl = init_pts.shape[0] // d

    u = np.arange(0, init_pts.shape[0])
    key_weight = 1#init_pts.shape[0] / keypoints.shape[0]
    w = np.ones_like(u)
    w[keypoint_idxs] = key_weight
    knots = np.concatenate(
        (np.repeat(0, d),
        np.linspace(0, u[-1], num_ctrl),
        np.repeat(u[-1], d))
    )
    tck, *_ = interp.splprep(init_pts.T, w=w, u=u, k=d, task=-1, t=knots)

    t = tck[0]
    c = np.array(tck[1]).T
    k = tck[2]
    tck = interp.BSpline(t, c, k)
    init_spline = tck(np.linspace(0, u[-1], 150))

    constraints = []
    for key_idx in keypoint_idxs:
        for axis in range(3):
            constraints.append({
                "type" : "ineq",
                "fun" : keypt_constr,
                "args" : (knots, d, init_pts[key_idx], key_idx, axis)
            })

    final = scipy.optimize.minimize(
        shape_loss,
        c,
        method = 'SLSQP',
        args=(knots, d),
        constraints=constraints,
        options= {"maxiter" : 2}
    )
    print("success:", final.success)
    print("status:", final.status)
    print("message:", final.message)
    print("num iter:", final.nit)
    b = np.array([val for val in final.x]).reshape((-1, 3))
    tck = interp.BSpline(knots, b, d)
    final_spline = tck(np.linspace(0, u[-1], 150))
    plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.view_init(0, 0)
    ax1.set_zlim(0, 500)
    ax1.scatter(
        segpix1[:, 0],
        segpix1[:, 1],
        img_3D[segpix1[:, 0], segpix1[:, 1], 2],
        s=1, c="r", alpha=0.1)
    # ax.scatter(
    #     keypoints[:, 0],
    #     keypoints[:, 1],
    #     keypoints[:, 2],
    #     c="r"
    # )
    ax1.plot(
        init_spline[:, 0],
        init_spline[:, 1],
        init_spline[:, 2],
        c="b")
    plt.figure(2)
    ax2 = plt.axes(projection='3d')
    ax2.view_init(0, 0)
    ax2.set_zlim(0, 500)
    ax2.scatter(
        segpix1[:, 0],
        segpix1[:, 1],
        img_3D[segpix1[:, 0], segpix1[:, 1], 2],
        s=1, c="r", alpha=0.1)
    # ax.scatter(
    #     keypoints[:, 0],
    #     keypoints[:, 1],
    #     keypoints[:, 2],
    #     c="r"
    # )
    ax2.plot(
        final_spline[:, 0],
        final_spline[:, 1],
        final_spline[:, 2],
        c="b")
    plt.show()

    # plt.imshow(img1, cmap="gray")
    # plt.scatter(init_pts[:, 1], init_pts[:, 0], c=np.arange(0, init_pts.shape[0]), cmap="hot")
    # plt.scatter(keypoints[:, 1], keypoints[:, 0], c="r")
    # plt.show()

def shape_loss(args, knots, d):
    b = np.array([val for val in args]).reshape((-1, 3))
    spline = interp.BSpline(knots, b, d)
    dspline = spline.derivative()
    d2spline = dspline.derivative()
    d3spline = d2spline.derivative()
    d4spline = d3spline.derivative()

    def integrand(u):
        # precompute spline derivatives
        dsp = dspline(u)
        d2sp = d2spline(u)
        d3sp = d3spline(u)
        d4sp = d4spline(u)
        speed = np.linalg.norm(dsp)

        # compute arc-length derivative of curvature
        h = np.cross(dsp, d2sp)
        dh = np.cross(dsp, d3sp)
        f = np.linalg.norm(h)
        df = np.dot(h, dh) / f
        g = speed ** 3
        dg = 3 * speed * np.dot(dsp, d2sp)
        dkappa = (df*g - f*dg)/(g**2)
        dkappa_ds = dkappa**2 / speed

        # compute arc-length derivative of torsion
        j = np.dot(h, d3sp)
        dj = np.dot(h, d4sp)
        k = np.dot(h, h)
        dk = 2*np.dot(h, dh)
        dtau = (dj*k - j*dk)/(k**2)
        dtau_ds = dtau**2 / speed

        return dkappa_ds + dtau_ds
    val = scipy.integrate.quad(integrand, knots[0], knots[-1])[0]
    print(val)
    return val#scipy.integrate.quad(integrand, knots[0], knots[-1])[0]

DIST_CONSTR = [2, 2, 6]
def keypt_constr(args, knots, d, keypt, key_idx, axis):
    b = np.array([val for val in args]).reshape((-1, 3))
    tck = interp.BSpline(knots, b, d)
    splpt = tck(key_idx)
    dist = np.abs(keypt[axis] - splpt[axis])
    if dist > DIST_CONSTR[axis]:
        return -1*dist
    else:
        return 1/(dist + 1e-5)


if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_3_left_final.jpg"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_3_right_final.jpg"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_3D, keypoints, grow_paths, order = prob_cloud(img1, img2)
    curve_fit(img1, img_3D, keypoints, grow_paths, order)