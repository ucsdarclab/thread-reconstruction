import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from pixel_ordering import order_pixels

def ssp_reconstruction(ord1, ord2, steps1, steps2, calib):
    # Read in camera calibration values
    cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
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
    
    ord1 = ord1.T
    ord2 = ord2.T
    min_len = min(ord1.shape[0], ord2.shape[0])
    X_cloud = np.zeros((min_len, 9))
    for phi in range(min_len):
        for i_s in range(9):
            sp = np.random.randn()
            ord_s = sp + phi
            while ord_s >= ord2.shape[0]-1:
                ord_s = np.random.randn() + phi
            ord_fl = np.int32(np.floor(ord_s))
            ord_cl = np.int32(np.ceil(ord_s))
            pix1 = ord2[ord_fl]
            pix2 = ord2[ord_cl]
            interp_pix = pix1 + (pix2 - pix1)*(ord_s - ord_fl)
            disp = ord1[phi, 1] - interp_pix[1]
            depth_calc = np.matmul(Q, np.array([ord1[phi, 0], ord1[phi, 1], disp, 1]))
            depth_calc /= depth_calc[3].copy() + 1e-7
            X_cloud[phi, i_s] = depth_calc[2]
    
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 500)
    X_cloud = X_cloud.T
    for row in X_cloud:
        ax.scatter(ord1[:min_len, 0], ord1[:min_len, 1], row)
    plt.show()

    



if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_3_left_final.jpg"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_3_right_final.jpg"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # plt.imshow(img1)
    # plt.show()
    # plt.clf()
    # plt.imshow(img2)
    # plt.show()
    # assert False
    # fileb = "../Blender_imgs/blend3/blend3_3.jpg"
    # calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
    # imgb = cv2.imread(fileb)
    # imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    # img1 = imgb[:, :640]
    # img2 = imgb[:, 640:]
    # img1 = np.where(img1>=200, 255, img1)
    # img2 = np.where(img2>=200, 255, img2)

    ord1, ord2, steps1, steps2 = order_pixels(img1, img2)
    # plt.figure(1)
    # plt.imshow(img1)
    # plt.scatter(ord1[1], ord1[0], c=np.arange(ord1.shape[1]), cmap="hot")
    # plt.figure(2)
    # plt.imshow(img2)
    # plt.scatter(ord2[1], ord2[0], c=np.arange(ord2.shape[1]), cmap="hot")
    # plt.show()
    print(ord1.shape)
    print(ord2.shape)
    ssp_reconstruction(ord1, ord2, steps1, steps2, calib)