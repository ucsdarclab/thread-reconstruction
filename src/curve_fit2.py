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

def curve_fit2(img1, img2):
    img_3D = stereo_match(img1, img2)
    thresh = 226
    seg_pix = np.argwhere(img1 <= thresh)

    # TODO update method of getting depth_bounds if cloud changes
    cloud = prob_cloud(img1, img2)
    cloud = cloud.reshape(-1, 5, 3)
    cloud_bounds = cloud[:, ::4, :]

    # Reshape bounds into (row, col, thresh_idx, depth), where thresh_idx is 0 for low and 1 for high
    # TODO sanity check?
    bound_map = np.zeros((img1.shape[0], img1.shape[1], 2, 1))
    bound_map[np.int32(cloud_bounds[:, 0, 0]), np.int32(cloud_bounds[:, 0, 1]), 0] = cloud_bounds[:, 0, 2:]
    bound_map[np.int32(cloud_bounds[:, 1, 0]), np.int32(cloud_bounds[:, 1, 1]), 1] = cloud_bounds[:, 1, 2:]

    order, _, steps = order_pixels(img1, img2)
    cstep = 3
    # organized as list of [row, col, lower, upper] arrays
    depth_bounds = np.zeros((len(steps)//cstep+1, 4))
    ord_idx = 0
    for i in range(0, len(steps), cstep):
        j = min(i+cstep, len(steps))
        # squared sum of steps = number of pixels in current interval
        interval = sum([val**2 for val in steps[i:j]])
        pixels = order[:, ord_idx:ord_idx+interval]
        depths = bound_map[pixels[0], pixels[1]]
        mean = np.mean(depths, axis=(0, 2))
        depth_bounds[i//cstep] = np.array([
            pixels[0, interval//2],
            pixels[1, interval//2],
            mean[0],
            mean[1]
        ])

        ord_idx += interval
    # ax = plt.axes(projection="3d")
    # cloud_bounds = cloud_bounds.reshape(-1, 3)
    # ax.scatter(seg_pix[:, 0], seg_pix[:, 1], img_3D[seg_pix[:, 0], seg_pix[:, 1], 2], s=1)
    # ax.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 2], c="b")
    # ax.plot(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 3], c="b")
    # print(depth_bounds.shape)
    # lim = slice(17, 27)
    # ax.scatter(depth_bounds[17:27, 0], depth_bounds[17:27, 1], depth_bounds[17:27, 2], c="orange")
    # ax.scatter(depth_bounds[17:27, 0], depth_bounds[17:27, 1], depth_bounds[17:27, 3], c="orange")
    # ax.set_zlim(0, 1000)
    # plt.show()
    
    start = 17
    end = 27-1

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
    b0 = tck0.c

    x = np.linspace(start, end, 50)
    spline = interp.splev(x, tck0)
    spline_b = interp.splev(t_star, tck0)
    e_low, e_top = get_envelope(tck0, t_star)
    plt.plot(x, spline)
    # plt.plot(t_star, e_low, c="r")
    # plt.plot(t_star, e_top, c="r")
    idxs = np.arange(start, end+1)
    plt.plot(idxs, depth_bounds[start:end+1, 2], c="g")
    plt.plot(idxs, depth_bounds[start:end+1, 3], c="g")
    plt.title("initial")

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
    x = np.linspace(start, end, 50)
    spline = interp.splev(x, tck)
    spline_b = interp.splev(t_star, tck)
    e_low, e_top = get_envelope(tck, t_star)
    plt.plot(x, spline)
    # plt.plot(t_star, e_low, c="r")
    # plt.plot(t_star, e_top, c="r")
    idxs = np.arange(start, end+1)
    plt.plot(idxs, depth_bounds[start:end+1, 2], c="g")
    plt.plot(idxs, depth_bounds[start:end+1, 3], c="g")
    plt.title("final")
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
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    curve_fit2(img1, img2)