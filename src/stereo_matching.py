import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def stereo_match():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img1 = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

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

    sgbm_win_size = 3
    sgbm = cv2.StereoSGBM.create(
        numDisparities=((img_size[1]//8) + 15) & -16,
        P1=8*9,
        P2=32*9,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disp = sgbm.compute(img1, img2)
    # max_disp = max(abs(np.min(disp)), abs(np.max(disp)))
    disp = cv2.normalize(disp, disp, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
    disp = np.uint8(disp)
    plt.imshow(disp, cmap="gray")
    plt.show()
    


if __name__ == "__main__":
    stereo_match()