import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
from pixel_ordering import order_pixels
from curve_fit import fit_2D_curves

OPENCV_MATCHING = False

def match_curves():
    # Get normal and segmented images
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img1 = cv2.imread(img_dir + "thread_1_left.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img_dir + "thread_1_right.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1seg = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img1seg = cv2.cvtColor(img1seg, cv2.COLOR_BGR2GRAY)
    img2seg = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img2seg = cv2.cvtColor(img2seg, cv2.COLOR_BGR2GRAY)

    # Read in camera calibration values
    cv_file = cv2.FileStorage(img_dir + "camera_calibration_fei.yaml", cv2.FILE_STORAGE_READ)
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
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

    img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    curve_l, curve_r = fit_2D_curves()

    n = 30
    indicies = [(i * curve_l.size(1)) // n for i in range(n)]
    proj_ctrl_l = torch.stack(
        (
            torch.tensor([curve_l[0, i] for i in indicies]),
            torch.tensor([curve_l[1, i] for i in indicies])
        )
    )
    # proj_ctrl_r = torch.stack(
    #     (
    #         torch.tensor([curve_r[0, i] for i in indicies]),
    #         torch.tensor([curve_r[1, i] for i in indicies])
    #     )
    # )
    disp = torch.zeros(curve_l.size(1))
    scan_window = 10
    c = 0.3
    for i in range(curve_l.size(1)):
        pt_l = curve_l[:, i]
        min_off = max(0, i-scan_window)
        max_off = min(curve_l.size(1), i + scan_window)
        y_dist = np.array([abs(curve_r[0, off] - pt_l[0]) + c * abs(i - off) for off in range(min_off, max_off)])
        # if (np.min(y_dist) > 4):
        #     continue
        best_idx = np.argmin(y_dist) + min_off
        disp[i] = pt_l[1] - curve_r[1, best_idx]

    disp = curve_l[1] - curve_r[1]
    curve_3D_l = torch.stack([curve_l[0], curve_l[1], disp])

    ax = plt.axes(projection='3d')
    ax.view_init(0, 0)
    ax.scatter(
        curve_3D_l[0],
        curve_3D_l[1],
        curve_3D_l[2]
    )
    # ax.set_zlim(-5, 400)
    plt.show()

if __name__ == "__main__":
    match_curves()