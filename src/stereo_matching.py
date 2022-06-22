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
        disp_cv = sgbm.compute(img1, img2)
        disp_cv = np.float32(disp_cv) / 16.0
        img_3D = cv2.reprojectImageTo3D(disp_cv, Q)
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
        # return P1, P2, points_3D #TODO Add this back
    if True: # TODO remove
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        thresh = 40
        pix_thresh = 210
        max_disp = 40
        # numerator for 3D calculation
        num = np.linalg.norm(T) * np.sqrt(K1[0, 0]**2 + K1[1, 1]**2) # cm * pixels ? TODO Check units

        pixels1 = np.argwhere(img1<=pix_thresh)
        pix_to_idx = {(int(pix[0]), int(pix[1])):i for i, pix in enumerate(pixels1)}
        disps = np.zeros(pixels1.shape[0])
        reliab = np.zeros(pixels1.shape[0])
        affins = np.zeros((pixels1.shape[0], 9))

        # to_r = 1e5 #prevent floating-point cut-off
        rad = 2
        c_data = 5
        c_slope = 8
        c_shift = 0.8
        # TODO deal with out-of-bounds conditions
        for i in range(pixels1.shape[0]):
            pix1 = (pixels1[i][0], pixels1[i][1])
            chunk = img1[pix1[0]-rad:pix1[0]+rad+1, pix1[1]-rad:pix1[1]+rad+1]
            seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(pixels1[i], 0) - rad

            # energy = np.array([(img1[pix1] - img2[pix1[0], pix1[1] - off])**2 for off in range(max_disp)])
            energy = np.array([
                np.sum(
                    (img1[seg[:,0], seg[:,1]] - img2[seg[:,0], seg[:,1] - off])**2
                ) for off in range(max_disp)
            ])
            
            best = np.min(energy)#to_r/(np.min(energy) + 1e-7)
            disp = np.argmin(energy)
            energy2 = np.delete(energy, slice(disp-1, disp+2))
            next_best = np.min(energy2)#to_r/(np.min(energy) + 1e-7)
            
            disps[i] = disp
            x = (next_best - best)/((best + 1e-7)*c_data)
            reliab[i] = 1/(1+np.exp(-1*c_slope*(x-c_shift)))

            roi = img1[pix1[0]-1:pix1[0]+2, pix1[1]-1:pix1[1]+2].copy()
            # ignore current pixel
            roi[1, 1] = pix_thresh+1
            neighs = np.argwhere(roi<=pix_thresh) + np.expand_dims(pixels1[i], 0) - 1
            valn = np.array([img1[pixn[0], pixn[1]] for pixn in neighs])
            if neighs.size == 0:
                continue
            varn = np.var(valn)
            curr_affins = np.exp(np.clip(-1*(img1[pix1] - valn)**2 / (2*varn + 1e-7), -87, None))
            curr_affins /= np.sum(curr_affins) + 1e-7
            for j, pixn in enumerate(neighs):
                idx = 3*(pixn[0]-pix1[0]+1) + pixn[1]-pix1[1]+1
                affins[i, idx] = curr_affins[j]
        
        # Perform laplacian propagation
        F_hat = disps
        P = np.diag(reliab)
        W = np.zeros_like(P)
        for i, pix in enumerate(pixels1):
            for j, affin in enumerate(affins[i]):
                if affin > 1e-7:
                    r = j//3 + pix[0] - 1
                    c = j%3 + pix[1] - 1
                    idx = pix_to_idx[(r, c)]
                    W[i, idx] = affin
        F = np.linalg.solve(
            P + np.eye(len(pixels1)) - W,
            np.matmul(P, F_hat)
        )
        
        disp_map = np.zeros_like(img1)
        disp_map[pixels1[:, 0], pixels1[:, 1]] = F#disps
        # reliab_map = np.zeros_like(img1)
        # reliab_map[pixels1[:, 0], pixels1[:, 1]] = reliab
        img2rgb = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        for i in range(pixels1.shape[0]):
            pix1 = (pixels1[i][0], pixels1[i][1])
            if disp_map[pix1]:
                img2rgb[pix1[0], int(pix1[1] - disp_map[pix1]), 0] -= 60
                img2rgb[pix1[0], int(pix1[1] - disp_map[pix1]), 2] -= 60
        img2rgb = np.uint8(np.clip(img2rgb, 0, 255))
        plt.imshow(img2rgb)
        plt.figure(2)
        heatmap = plt.pcolor(disp_map)
        plt.colorbar(heatmap)
        plt.gca().invert_yaxis()
        # plt.figure(3)
        # heatmap_r = plt.pcolor(reliab_map)
        # plt.colorbar(heatmap_r)
        # plt.gca().invert_yaxis()
        # plt.figure(3)
        # ax = plt.axes(projection='3d')
        # ax.view_init(0, 0)
        # ax.set_zlim(0, 1000)
        # ax.scatter(
        #     pixels1[:, 0],
        #     pixels1[:, 1],
        #     num/F
        # )
        # ax.scatter(
        #     pixels1[:, 0],
        #     pixels1[:, 1],
        #     img_3D[pixels1[:, 0], pixels1[:, 1], 2],
        #     c="r"
        # )
        plt.figure(3)
        disp_cv_map = np.zeros_like(disp_cv)
        disp_cv_map[pixels1[:, 0], pixels1[:, 1]] += disp_cv[pixels1[:, 0], pixels1[:, 1]]
        diff = disp_map - disp_cv_map
        heatmap = plt.pcolor(diff, vmin=-5, vmax=5)
        plt.colorbar(heatmap)
        plt.gca().invert_yaxis()
        plt.show()
            

    


if __name__ == "__main__":
    stereo_match()