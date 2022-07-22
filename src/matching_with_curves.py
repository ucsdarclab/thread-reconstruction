import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
from pixel_ordering import order_pixels
from curve_fit import fit_2D_curves
import scipy.interpolate as interp
import scipy.integrate

OPENCV_MATCHING = False

def match_curves(img1, img2):
    # Read in camera calibration values
    cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml", cv2.FILE_STORAGE_READ)
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

    ord1, ord2, steps1, steps2 = order_pixels(img1, img2)
    pts1 = np.zeros((len(steps1), 2))
    ord_idx = 0
    for i, step in enumerate(steps1):
        pts1[i] = np.mean(ord1[:, ord_idx:ord_idx+step**2], axis=1)
        ord_idx += step**2
    pts2 = np.zeros((len(steps2), 2))
    ord_idx = 0
    for i, step in enumerate(steps2):
        pts2[i] = np.mean(ord2[:, ord_idx:ord_idx+step**2], axis=1)
        ord_idx += step**2
    
    idxs1 = np.arange(0, pts1.shape[0])
    # knots1 = np.concatenate((np.repeat(0, 4), np.linspace(0, idxs1[-1], 30), np.repeat(idxs1[-1], 4)))
    tck1, *_ = interp.splprep(
        pts1.T, u=idxs1, k=4#, task=-1, t=knots1
    )
    t1 = tck1[0]
    c1 = np.array(tck1[1]).T
    k1 = tck1[2]
    tck1 = interp.BSpline(t1, c1, k1)
    # plt.imshow(img1, cmap="gray")
    # spline1 = tck1(idxs1)
    # plt.plot(spline1[:, 1], spline1[:, 0])
    # plt.show()

    idxs2 = np.arange(0, pts2.shape[0])
    # knots2 = np.concatenate((np.repeat(0, 4), np.linspace(0, idxs2[-1], 30), np.repeat(idxs2[-1], 4)))
    tck2, *_ = interp.splprep(
        pts2.T, u=idxs2, k=4#, task=-1, t=knots2
    )
    t2 = tck2[0]
    c2 = np.array(tck2[1]).T
    k2 = tck2[2]
    tck2 = interp.BSpline(t2, c2, k2)
    
    g1_unnormed = tck1.derivative()
    g2_unnormed = tck2.derivative()

    # len1, _ = scipy.integrate.quad(lambda x : np.linalg.norm(g1_unnormed(x)), idxs1[0], idxs1[-1])
    # len2, _ = scipy.integrate.quad(lambda x : np.linalg.norm(g2_unnormed(x)), idxs2[0], idxs2[-1])
    lens1 = np.linalg.norm(pts1[1:] - pts1[:-1], axis=1)
    total_len1 = np.sum(lens1)
    lens1 *= 100
    lens1 /= total_len1
    lens2 = np.linalg.norm(pts2[1:] - pts2[:-1], axis=1)
    total_len2 = np.sum(lens2)
    lens2 *= 100
    lens2 /= total_len2

    disp = np.zeros(pts1.shape[0])
    disp[0] = pts1[0, 1] - pts2[0, 1]
    len1 = lens1[0]
    len2 = lens2[0]
    idx2 = 1
    for idx1 in range(1, pts1.shape[0]):
        while len2 < len1 - 1e-5:
            len2 += lens2[idx2]
            idx2 += 1
        len_diff = max(0, len2 - len1)
        ratio = len_diff / lens2[idx2-1]
        interp_pt = pts2[idx2] - (pts2[idx2] - pts2[idx2-1])*ratio
        disp[idx1] = pts1[idx1, 1] - interp_pt[1]
        if idx1 < lens1.shape[0]:
            len1 += lens1[idx1]
    
    plt.figure(1)
    disp_map = np.zeros_like(img1)
    disp_map[np.int32(pts1[:, 0]), np.int32(pts1[:, 1])] = disp
    heatmap = plt.pcolor(disp_map)
    plt.colorbar(heatmap)
    plt.gca().invert_yaxis()

    # compare to opencv
    sgbm_win_size = 5
    sgbm = cv2.StereoSGBM.create(
        numDisparities=((img_size[0]//8) + 15) & -16,
        blockSize=sgbm_win_size,
        P1=8*1*sgbm_win_size*sgbm_win_size,
        P2=32*1*sgbm_win_size*sgbm_win_size,
        disp12MaxDiff=1,
        # preFilterCap=63,
        # uniquenessRatio=10,
        # speckleWindowSize=100,
        # speckleRange=10
    )
    disp_cv = sgbm.compute(img1, img2)
    disp_cv = np.float32(disp_cv) / 16.0
    img_3D = cv2.reprojectImageTo3D(disp_cv, Q)

    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.view_init(0, 0)
    ax.set_zlim(0, 1000)
    depth = np.matmul(
        Q,
        np.stack((np.int32(pts1[:, 0]), np.int32(pts1[:, 1]), disp, np.ones_like(disp)))
    )
    depth /= np.expand_dims(depth[3].copy(), 0)
    ax.scatter(
        np.int32(pts1[:, 0]),
        np.int32(pts1[:, 1]),
        depth[2],
        s=1
    )
    ax.scatter(
        np.int32(pts1[:, 0]),
        np.int32(pts1[:, 1]),
        img_3D[np.int32(pts1[:, 0]), np.int32(pts1[:, 1]), 2],
        c="r",
        s=1
    )
    plt.show()
        

    # TODO make this less hard-coded
    num_samples = 200


    # curve_l, curve_r = fit_2D_curves()

    # n = 30
    # indicies = [(i * curve_l.size(1)) // n for i in range(n)]
    # proj_ctrl_l = torch.stack(
    #     (
    #         torch.tensor([curve_l[0, i] for i in indicies]),
    #         torch.tensor([curve_l[1, i] for i in indicies])
    #     )
    # )
    # # proj_ctrl_r = torch.stack(
    # #     (
    # #         torch.tensor([curve_r[0, i] for i in indicies]),
    # #         torch.tensor([curve_r[1, i] for i in indicies])
    # #     )
    # # )
    # disp = torch.zeros(curve_l.size(1))
    # scan_window = 10
    # c = 0.3
    # for i in range(curve_l.size(1)):
    #     pt_l = curve_l[:, i]
    #     min_off = max(0, i-scan_window)
    #     max_off = min(curve_l.size(1), i + scan_window)
    #     y_dist = np.array([abs(curve_r[0, off] - pt_l[0]) + c * abs(i - off) for off in range(min_off, max_off)])
    #     # if (np.min(y_dist) > 4):
    #     #     continue
    #     best_idx = np.argmin(y_dist) + min_off
    #     disp[i] = pt_l[1] - curve_r[1, best_idx]

    # disp = curve_l[1] - curve_r[1]
    # curve_3D_l = torch.stack([curve_l[0], curve_l[1], disp])

    # ax = plt.axes(projection='3d')
    # ax.view_init(0, 0)
    # ax.scatter(
    #     curve_3D_l[0],
    #     curve_3D_l[1],
    #     curve_3D_l[2]
    # )
    # # ax.set_zlim(-5, 400)
    # plt.show()

if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    match_curves(img1, img2)