import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
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
    

    # Initialize knots to take on range [start, end]
    num_bases = 7
    d = 4
    knots = np.linspace(start, end, num_bases+1 - (d+1) + 1)
    delta = knots[1] - knots[0]
    knots_start = np.ones((d,)) * start
    knots_end = np.ones((d,)) * end
    knots = np.concatenate((knots_start, knots, knots_end))

    # Initialize other spline parameters
    t_star = np.array([np.sum(knots[k+1:k+d+1])/d for k in range(num_bases)]) #Greville abscissae
    # l = np.zeros_like(t_star, dtype=np.int32)
    # knot_idx = d
    # for i, pt in enumerate(t_star):
    #     while knots[knot_idx+1] < pt:
    #         knot_idx += 1
    #     l[i] = knot_idx
    # k_low = l[:-1] - (d - 1)
    # k_high = l[1:]
    # I_star = np.array([[i for i in range(l[k]-d+1, l[k])] for k in range(num_bases)])
    # Bki = np.zeros_like(I_star)
    # for k in range(num_bases):
    #     for i in I_star[k]:
    #         if i > k:
    #             t = knots[i:k_high[k]+1 + d+1]
    #             c = np.array([t_star[j] - t_star[i] for j in range(i, k_high[k]+1)])
    #         else:
    #             t = knots[k_low[k]:i+1 + d+1]
    #             c = np.array([t_star[i] - t_star[j] for j in range(k_low[k], i+1)])
    #         Bki[k, i] = interp.BSpline(t, c, d)
    b = np.array([0, 2, 2, 0, 0, 1, 1])
    tck = interp.BSpline(knots, b, d)
    x = np.linspace(start, end, 50)
    x_b = t_star#np.linspace(start, end, num_bases)
    spline = interp.splev(x, tck)
    spline_b = interp.splev(x_b, tck)
    e_top = np.where(b>spline_b, b, spline_b)
    e_low = np.where(b<spline_b, b, spline_b)
    plt.plot(x, spline)
    plt.plot(x_b, e_top, c="r")
    plt.plot(x_b, e_low, c="r")
    plt.scatter(x_b, b, c="g")
    plt.show()

    

    # Set endpoint equality constraints
    # TODO Make sure chosen depth for keypoint is correct
    start_d = (depth_bounds[start, 2] + depth_bounds[start, 3]) / 2
    # TODO add extra checks to make sure no OOB errors occur
    # TODO Choose different numerical derivative calculation?
    post_start_d = (depth_bounds[start+1, 2] + depth_bounds[start+1, 3]) / 2
    pre_start_d = (depth_bounds[start-1, 2] + depth_bounds[start-1, 3]) / 2
    post2_start_d = (depth_bounds[start+2, 2] + depth_bounds[start+1, 3]) / 2
    a0 = start_d
    a1 = post_start_d - a0
    a2 = a1 - (a0 - pre_start_d)
    next_2nd = (post2_start_d - post_start_d) - a1
    a3 = next_2nd - a2
    
    # TODO As above
    end_d = (depth_bounds[end, 2] + depth_bounds[end, 3]) / 2
    post_end_d = (depth_bounds[end+1, 2] + depth_bounds[end+1, 3]) / 2
    pre_end_d = (depth_bounds[end-1, 2] + depth_bounds[end-1, 3]) / 2
    post2_end_d = (depth_bounds[end+2, 2] + depth_bounds[end+1, 3]) / 2
    b0 = end_d
    b1 = post_end_d - b0
    b2 = b1 - (b0 - pre_end_d)
    next_2nd = (post2_end_d - post_end_d) - b1
    b3 = next_2nd - b2


    def objective(x):
        pass



if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    curve_fit2(img1, img2)