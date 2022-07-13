import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
import cv2
from stereo_matching import stereo_match
import sys

def prob_cloud(img1, img2):
    img_3D = stereo_match(img1, img2)
    thresh = 226
    seg_pix = np.argwhere(img1 <= thresh)

    rng = np.random.default_rng()
    sample = seg_pix[rng.choice(seg_pix.shape[0], seg_pix.shape[0]//10)]
    # test = np.ones_like(img1) * 255
    # test[sample[:, 0], sample[:, 1]] = 0
    # plt.imshow(test)
    # plt.show()
    rad = 6
    # TODO try sampling
    # means = np.zeros_like(img1)
    # varians = np.zeros_like(means)
    cloud = np.zeros((5*seg_pix.shape[0], 3))
    varians = np.zeros_like(img1)
    for i, pix in enumerate(seg_pix):
        r0 = max(0, pix[0]-rad)
        r1 = min(img1.shape[0]-1, pix[0]+rad+1)
        c0 = max(0, pix[1]-rad)
        c1 = min(img1.shape[1]-1, pix[1]+rad+1)
        roi = img1[r0:r1, c0:c1]
        roi_seg = np.argwhere(roi <= thresh) + np.expand_dims(pix, 0) - rad
        roi_depths = img_3D[roi_seg[:, 0], roi_seg[:, 1], 2]
        roi_depths = roi_depths[roi_depths != np.inf]
        roi_depths = roi_depths[roi_depths > 0]
        mean = np.mean(roi_depths)
        std = np.std(roi_depths)
        varians[pix[0], pix[1]] = std**2
        cloud[5*i] = np.array([pix[0], pix[1], mean-2*std])
        cloud[5*i+1] = np.array([pix[0], pix[1], mean-std])
        cloud[5*i+2] = np.array([pix[0], pix[1], mean])
        cloud[5*i+3] = np.array([pix[0], pix[1], mean+std])
        cloud[5*i+4] = np.array([pix[0], pix[1], mean+2*std])
    # var_var = np.zeros((seg_pix.shape[0]*5))
    # for i, pix in enumerate(seg_pix):
    #     roi = img1[pix[0]-1:pix[0]+2, pix[1]-1:pix[1]+2]
    #     roi_seg = np.argwhere(roi <= thresh) + np.expand_dims(pix, 0) - 1
    #     roi_vars = varians[roi_seg[:, 0], roi_seg[:, 1]]
    #     var_var[5*i:5*i+5] = 0 if varians[pix[0], pix[1]]<1e-7 else \
    #         np.log(np.var(roi_vars) / varians[pix[0], pix[1]]+1)
    ax = plt.axes(projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2)#, c=var_var)
    # ax.scatter(
    #     seg_pix[:, 0],
    #     seg_pix[:, 1],
    #     img_3D[seg_pix[:, 0], seg_pix[:, 1], 2],
    #     s=2, c="r")
    ax.set_zlim(0, 1000)
    # plt.show()
    return cloud


if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    prob_cloud(img1, img2)