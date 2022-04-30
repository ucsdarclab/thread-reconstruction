import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
from pixel_ordering import order_pixels

OPENCV_MATCHING = True
TESTING = False

def stereo_match():
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

    # TODO remove if experimentation fails
    img1 = img1seg
    img2 = img2seg

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

    if OPENCV_MATCHING:
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
        disp = sgbm.compute(img1, img2)
        disp = np.float32(disp) / 16.0
        img_3D = cv2.reprojectImageTo3D(disp, Q)
        ordering, _ = order_pixels()
        points_3D = img_3D[ordering[1], ordering[0]]#np.stack([
        #     ordering[1],
        #     ordering[0],
        #     img_3D[ordering[1], ordering[0], 2]
        # ])
        if TESTING:
            ordering, ordering2 = order_pixels()
            # ordering_0 = np.int64([ordering[0, i] * 480/433 + off 
            #     for i in range(ordering.shape[1]) for off in [-1,0,1,-1,0,1,-1,0,1]])
            # ordering_1 = np.int64([ordering[1, i] * 640/577+ off 
            #     for i in range(ordering.shape[1]) for off in [-1,-1,-1,0,0,0,1,1,1]])
            camera_3D = img_3D[ordering[1, 0], ordering[0, 0]]
            camera_3D = np.array([camera_3D[0], camera_3D[1], camera_3D[2], 1])
            camera_proj = np.matmul(P2, camera_3D)
            camera_proj /= camera_proj[-1]
            print(camera_proj)
            print(ordering2[..., 0])
            return

            ax = plt.axes(projection='3d')
            ax.view_init(0, 0)
            # ax.set_xlim(1, 500)
            # ax.set_ylim(1, 500)
            # ax.set_zlim(100, 300)
            ax.scatter(
                ordering[1],#img_3D[ordering[1], ordering[0], 0],
                ordering[0],#img_3D[ordering[1], ordering[0], 1],
                img_3D[ordering[1], ordering[0], 2]
            )
            plt.show()
        return P1, P2, points_3D
    else:
        # get orderings
        ord1, ord2 = order_pixels()
        thresh = 10
        min_disp = 0#15
        max_disp = 40
        # numerator for disparity calculation
        num = np.linalg.norm(T) * np.sqrt(K1[0, 0]**2 + K1[1, 1]**2) # cm * pixels ? TODO Check units
        ord_3D = []

        ord2mat = np.zeros((433, 577))
        # get better pix2 lookup. TODO Optimize
        for i in range(ord2.shape[1]):
            pix2 = (ord2[1, i], ord2[0, i])
            ord2mat[pix2] = img2seg[pix2]#np.mean(img2[pix2[0], pix2[1], :])
        kernel = np.ones((3, 3))
        ord2mat = cv2.dilate(ord2mat, kernel, iterations=1)

        prev_disp = 0
        lam = 0.5
        for i in range(ord1.shape[1]):
            pix1 = (ord1[1, i], ord1[0, i])
            energy = np.array([(img1seg[pix1] - ord2mat[pix1[0], pix1[1] - off])**2 for off in range(min_disp, max_disp)])
            # factor in smoothness
            if i != 0:
                # print(energy.shape)
                # print()
                # print(np.linspace(prev_disp - min_disp, max_disp - prev_disp, endpoint=False, num=energy.shape[0]))
                # print(np.abs(np.arange(prev_disp - min_disp, max_disp - prev_disp)).shape)
                energy += lam * np.abs(np.array([prev_disp - min_disp - i for i in range(energy.shape[0])]))
            if np.min(energy) > thresh:
                continue
            disp = np.argmin(energy) + min_disp
            if disp == 0:
                continue
            ord_3D.append([pix1[0], pix1[1], num / disp])
            prev_disp = disp

        ord_3D = np.array(ord_3D)
        # ordMed = np.array([
        #     [ord_3D[i, 0], ord_3D[i, 1], np.median(ord_3D[i-5:i+6, 2])] for i in range(5, ord_3D.shape[0] - 5)
        # ])
        # ord_3D = ordMed
        # plt.plot(ord_3D[:, 2])
        ax = plt.axes(projection='3d')
        ax.view_init(0, 0)
        ax.scatter(
            ord_3D[:, 0],
            ord_3D[:, 1],
            ord_3D[:, 2]
        )
        ax.set_zlim(-5, 400)
        plt.show()
            

    


if __name__ == "__main__":
    stereo_match()