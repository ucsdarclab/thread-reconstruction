import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.interpolate as interp
import cv2
from stereo_matching import stereo_match
from prob_cloud import prob_cloud
from pixel_ordering import order_pixels
import sys

def curve_fit(img1, img_3D, keypoints, bounds_rads, grow_paths, order):
    seg_pix = np.argwhere(img1 <= 250)
    lower = keypoints[order].copy()
    lower[:, 2] -= bounds_rads
    upper = keypoints[order].copy()
    upper[:, 2] += bounds_rads

    # Ground truth, for testing
    gt_b = np.load("/Users/neelay/ARClabXtra/Blender_imgs/blend1_pos.npy")
    cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Blender_imgs/blend1_calibration.yaml", cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    gt_pix = np.matmul(K1, gt_b.T).T
    gt_b[:, :2] = gt_pix[:, :2] / gt_pix[:, 2:]
    gk = 3
    gt_knots = np.concatenate(
        (np.repeat(0, gk),
        np.linspace(0, 1, gt_b.shape[0]-gk+1),
        np.repeat(1, gk))
    )
    gt_tck = interp.BSpline(gt_knots, gt_b, gk)
    gt_spline = gt_tck(np.linspace(0, 1, 150))

    ax = plt.axes(projection="3d")
    ax.plot(
        gt_spline[:, 1],
        gt_spline[:, 0],
        gt_spline[:, 2],
        c="g")
    ax.plot(lower[:, 0], lower[:, 1], lower[:, 2], c="r")
    ax.plot(upper[:, 0], upper[:, 1], upper[:, 2], c="b")
    keypts = np.array([0, 17, 27, 40, 49, 64, 72, 87, 100, 108, 115, 125, 133])
    # ax.scatter(depth_bounds[keypts, 0], depth_bounds[keypts, 1], (depth_bounds[keypts, 2] + depth_bounds[keypts, 3])/2, c="r")
    # ax.scatter(depth_bounds[17:27, 0], depth_bounds[17:27, 1], depth_bounds[17:27, 3], c="orange")
    # ax.set_zlim(0, 1000)
    plt.show()
    return
    "constructing xy part of 3D spline"
    # TODO Refine
    sample_idxs = np.arange(0, 134)
    xy_iknots = np.concatenate((np.repeat(0, 4), np.linspace(0, sample_idxs[-1], 30), np.repeat(sample_idxs[-1], 4)))
    xy_tck, *_ = interp.splprep(
        depth_bounds[sample_idxs, :2].T, u=sample_idxs, k=4, task=-1, t=xy_iknots
    )
    xy_t = xy_tck[0]
    xy_c = np.array(xy_tck[1]).T
    xy_k = xy_tck[2]
    xy_tck = interp.BSpline(xy_t, xy_c, xy_k)
    # plt.imshow(img1)
    # xy_spline = xy_tck(np.linspace(0, sample_idxs[-1], 150))
    # plt.plot(xy_spline[:, 1], xy_spline[:, 0], c="r")
    
    init_splines = []
    final_splines = []
    # start = 17
    # end = 27-1
    for start, end in zip(keypts[:-1], keypts[1:]):

        # Initialize spline params
        num_bases = 7
        d = 4
        # Initialize knots to take on range [start, end]
        knots = np.linspace(start, end, num_bases+1 - (d+1) + 1)
        knots_start = np.ones((d,)) * start
        knots_end = np.ones((d,)) * end
        knots = np.concatenate((knots_start, knots, knots_end))
        t_star = np.array([np.sum(knots[k+1:k+d+1])/d for k in range(num_bases)]) #Greville abscissae

        low_constr = interp.interp1d(np.arange(start, end+1), depth_bounds[start:end+1, 2])
        high_constr = interp.interp1d(np.arange(start, end+1), depth_bounds[start:end+1, 3])
        center = (low_constr(t_star) + high_constr(t_star))/2
        tck0 = interp.splrep(t_star, center, k=4, t=knots[5:-5]) #interp.BSpline(knots, b0, d)
        tck0 = interp.BSpline(tck0[0], tck0[1][:num_bases], tck0[2])
        init_splines.append(tck0)
        b0 = tck0.c

        "Plotting initial spline"
        # x = np.linspace(start, end, 50)
        # spline = interp.splev(x, tck0)
        # spline_b = interp.splev(t_star, tck0)
        # e_low, e_top = get_envelope(tck0, t_star)
        # plt.plot(x, spline)
        # # plt.plot(t_star, e_low, c="r")
        # # plt.plot(t_star, e_top, c="r")
        # idxs = np.arange(start, end+1)
        # plt.plot(idxs, depth_bounds[start:end+1, 2], c="g")
        # plt.plot(idxs, depth_bounds[start:end+1, 3], c="g")
        # plt.title("initial")

        A = [interp.splev([start], tck0, der) for der in range(4)]
        B = [interp.splev([end], tck0, der) for der in range(4)]
        t_all = np.arange(start, end+1)#np.sort(np.concatenate((t_star[1:-1], np.arange(start, end+1))))

        constraints = []
        for der, (Ai, Bi) in enumerate(zip(A[:1], B[:1])):
            constraints.append({"type":"eq", "fun":start_constr, "args":(knots, d, start, der, Ai)})
            constraints.append({"type":"eq", "fun":end_constr, "args":(knots, d, end, der, Bi)})
        for p1, p2 in zip(t_all[:-1], t_all[1:]):
            constraints.append(
                {
                    "type":"ineq",
                    "fun":lower_bound,
                    "args":(knots, d, t_star, low_constr, p1, p2)
                }
            )
            constraints.append(
                {
                    "type":"ineq",
                    "fun":upper_bound,
                    "args":(knots, d, t_star, high_constr, p1, p2)
                }
            )

        plt.figure(2)
        final = scipy.optimize.minimize(objective, b0, method = 'SLSQP', args=(knots, d, t_star), constraints=constraints)
        print("success:", final.success)
        print("status:", final.status)
        print("message:", final.message)
        print("num iter:", final.nit)
        # print("max violation:", final.maxcv)
        b = np.array([val for val in final.x])
        tck = interp.BSpline(knots, b, d)
        final_splines.append(tck)
        "Plotting final spline"
        # x = np.linspace(start, end, 50)
        # spline = interp.splev(x, tck)
        # spline_b = interp.splev(t_star, tck)
        # e_low, e_top = get_envelope(tck, t_star)
        # plt.plot(x, spline)
        # # plt.plot(t_star, e_low, c="r")
        # # plt.plot(t_star, e_top, c="r")
        # idxs = np.arange(start, end+1)
        # plt.plot(idxs, depth_bounds[start:end+1, 2], c="g")
        # plt.plot(idxs, depth_bounds[start:end+1, 3], c="g")
        # plt.title("final")
        # plt.show()
    
    plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.scatter(seg_pix[:, 0], seg_pix[:, 1], img_3D[seg_pix[:, 0], seg_pix[:, 1], 2], s=1)
    ax.set_zlim(0, 1000)
    plt.title("image-based reconstruction")
    
    plt.figure(2)
    plt.scatter(keypts, (depth_bounds[keypts, 2] + depth_bounds[keypts, 3])/2, c="b")
    plt.plot(sample_idxs, depth_bounds[:, 2], c="turquoise")
    plt.plot(sample_idxs, depth_bounds[:, 3], c="turquoise")
    for itck, p1, p2 in zip(init_splines, keypts[:-1], keypts[1:]):
        params = np.arange(p1, p2+1)
        plt.plot(params, itck(params), c="r")
    plt.ylim(0, 1000)
    plt.title("initial spline, depth vs order")
    # ax1 = plt.axes(projection="3d")
    # ax1.scatter(depth_bounds[keypts, 0], depth_bounds[keypts, 1], (depth_bounds[keypts, 2] + depth_bounds[keypts, 3])/2, c="b")
    # ax1.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 2], c="turquoise")
    # ax1.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 3], c="turquoise")
    # for itck, p1, p2 in zip(init_splines, keypts[:-1], keypts[1:]):
    #     params = np.arange(p1, p2+1)
    #     xy = xy_tck(params)
    #     z = itck(params)
    #     ax1.plot(xy[:, 0], xy[:, 1], z, c="r")
    # ax1.set_zlim(0, 1000)
    # plt.title("initial spline")

    plt.figure(3)
    plt.scatter(keypts, (depth_bounds[keypts, 2] + depth_bounds[keypts, 3])/2, c="b")
    plt.plot(sample_idxs, depth_bounds[:, 2], c="turquoise")
    plt.plot(sample_idxs, depth_bounds[:, 3], c="turquoise")
    for itck, p1, p2 in zip(final_splines, keypts[:-1], keypts[1:]):
        params = np.arange(p1, p2+1)
        plt.plot(params, itck(params), c="r")
    plt.ylim(0, 1000)
    plt.title("final spline, depth vs order")
    # ax2 = plt.axes(projection="3d")
    # ax2.scatter(depth_bounds[keypts, 0], depth_bounds[keypts, 1], (depth_bounds[keypts, 2] + depth_bounds[keypts, 3])/2, c="b")
    # ax2.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 2], c="turquoise")
    # ax2.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 3], c="turquoise")
    # for itck, p1, p2 in zip(final_splines, keypts[:-1], keypts[1:]):
    #     params = np.arange(p1, p2+1)
    #     xy = xy_tck(params)
    #     z = itck(params)
    #     ax2.plot(xy[:, 0], xy[:, 1], z, c="r")
    # ax2.set_zlim(0, 1000)
    # plt.title("final spline")
    plt.show()




def objective(args, knots, d, t_star):
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
    
    return scipy.integrate.quad(integrand, knots[0], knots[-1])[0]

def get_envelope(spline, t_star):
    spline_b = spline(t_star)
    b = spline.c
    e_low = np.where(b<spline_b, b, spline_b)
    e_top = np.where(b>spline_b, b, spline_b)
    return e_low, e_top

def start_constr(args, knots, d, start, der, A):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    spline = interp.splev([start], tck, der)
    return spline - A

def end_constr(args, knots, d, end, der, B):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    spline = interp.splev([end], tck, der)
    return spline - B

def lower_bound(args, knots, d, t_star, constr, start, end):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    # e_low, _ = get_envelope(tck, t_star)
    # e_interp = interp.interp1d(t_star, e_low)
    dists = tck([start, end]) - constr([start, end])#e_interp([start, end]) - constr([start, end])
    neg_only = np.where(dists<0, dists, 0)
    neg_sum = np.sum(neg_only)
    if neg_sum < 0:
        return neg_sum
    else:
        return np.sum(dists)

def upper_bound(args, knots, d, t_star, constr, start, end):
    b = np.array([val for val in args])
    tck = interp.BSpline(knots, b, d)
    # _, e_high = get_envelope(tck, t_star)
    # e_interp = interp.interp1d(t_star, e_high)
    dists = constr([start, end]) - tck([start, end])#e_interp([start, end])
    neg_only = np.where(dists<0, dists, 0)
    neg_sum = np.sum(neg_only)
    if neg_sum < 0:
        return neg_sum
    else:
        return np.sum(dists)
# def lower_bound(args, knots, d, t_star, constr, start, end):
#     b = np.array([val for val in args])
#     tck = interp.BSpline(knots, b, d)
#     e_low, _ = get_envelope(tck, t_star)
#     e_interp = interp.interp1d(t_star, e_low)
#     dists = e_interp(np.arange(start, end+1)) - constr
#     neg_only = np.where(dists<0, dists, 0)
#     neg_sum = np.sum(neg_only)
#     if neg_sum < 0:
#         return neg_sum
#     else:
#         return np.sum(dists)

# def upper_bound(args, knots, d, t_star, constr, start, end):
#     b = np.array([val for val in args])
#     tck = interp.BSpline(knots, b, d)
#     _, e_high = get_envelope(tck, t_star)
#     e_interp = interp.interp1d(t_star, e_high)
#     dists = constr - e_interp(np.arange(start, end+1))
#     neg_only = np.where(dists<0, dists, 0)
#     neg_sum = np.sum(neg_only)
#     if neg_sum < 0:
#         return neg_sum
#     else:
#         return np.sum(dists)

    


if __name__ == "__main__":
    # file1 = "../Sarah_imgs/thread_1_left_final.jpg"#sys.argv[1]
    # file2 = "../Sarah_imgs/thread_1_right_final.jpg"#sys.argv[2]
    # img1 = cv2.imread(file1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(file2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img_3D, keypoints, grow_paths, order = prob_cloud(img1, img2)
    fileb = "../Blender_imgs/blend_thread_1.jpg"
    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend1_calibration.yaml"
    imgb = cv2.imread(fileb)
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    img1 = imgb[:, :640]
    img2 = imgb[:, 640:]
    img1 = np.where(img1>=200, 255, img1)
    img2 = np.where(img2>=200, 255, img2)
    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(img2, cmap="gray")
    # plt.show()
    # assert False
    # test()
    img_3D, keypoints, bounds_rads, grow_paths, order = prob_cloud(img1, img2, calib)

    curve_fit(img1, img_3D, keypoints, bounds_rads, grow_paths, order)