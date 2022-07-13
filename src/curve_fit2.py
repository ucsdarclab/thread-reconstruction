import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
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
    print(ord_idx, order.shape[1])
    ax = plt.axes(projection="3d")
    cloud_bounds = cloud_bounds.reshape(-1, 3)
    ax.scatter(cloud_bounds[:, 0], cloud_bounds[:, 1], cloud_bounds[:, 2], s=2)
    ax.scatter(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 2], c="r")
    ax.scatter(depth_bounds[:, 0], depth_bounds[:, 1], depth_bounds[:, 3], c="r")
    # ax.set_zlim(0, 1000)
    plt.show()



if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    curve_fit2(img1, img2)