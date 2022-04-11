import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d

def stereo_match():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img1 = cv2.imread(img_dir + "thread_1_left.jpg")
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img_dir + "thread_1_right.jpg")
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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
    # plt.imshow(img1, cmap="gray")
    # plt.show()
    # plt.imshow(img2, cmap="gray")
    # plt.show()

    sgbm_win_size = 3
    sgbm = cv2.StereoSGBM.create(
        numDisparities=((img_size[0]//8) + 15) & -16,
        blockSize=5,
        # P1=8*9,
        # P2=32*9,
        # disp12MaxDiff=1,
        # preFilterCap=63,
        # uniquenessRatio=10,
        # speckleWindowSize=100,
        speckleRange=10
    )
    disp = sgbm.compute(img1, img2)
    # max_disp = max(abs(np.min(disp)), abs(np.max(disp)))
    # disp = cv2.normalize(disp, disp, alpha=255,
    #                           beta=0, norm_type=cv2.NORM_MINMAX)
    disp = np.float32(disp) / 16.0
    # plt.imshow(disp, cmap="gray")
    # plt.show()
    img_3D = cv2.reprojectImageTo3D(disp, Q)
    # img_3D = np.where(np.abs(img_3D) == np.inf, 0, img_3D)
    img_3D = np.clip(img_3D, -30, 30)
    img_3D -= img_3D.min()
    
    # img_3D = cv2.normalize(img_3D, img_3D, alpha=255,
    #                             beta=0, norm_type=cv2.NORM_MINMAX)
    img_3D *= 255 / np.abs(img_3D).max()
    plt.imshow(np.uint8(img_3D))
    # img_3D = img_3D.reshape((1, 480*640, 3)).squeeze(0)
    # ax = plt.axes(projection='3d')
    # ax.scatter(img_3D[..., 0], img_3D[..., 1], img_3D[..., 2])
    plt.show()
    


if __name__ == "__main__":
    stereo_match()