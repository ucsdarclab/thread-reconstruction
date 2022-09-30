import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
import sys

OPENCV_MATCHING = True
TESTING = False

"""
TODO Remove altogether?
"""

gt = np.array([
    [259, 337, 314],
    [256, 327, 304],
    [256, 319, 296],
    [259, 312, 289],
    [263, 305, 282],
    [270, 299, 275.5],
    [277, 296, 272],
    [288, 294, 270],
    [298, 296, 272],
    [310, 304, 279],
    [318, 311, 286],
    [326, 321, 296],
    [331, 329, 303.5],
    [336, 340, 314],
    [340, 352, 325],
    [341, 357, 331],
    [342, 362, 333.5],
    [342, 369, 342.5],
    [341, 373, 345],
    [342, 387, 363],
    [341, 392, 366.5],
    [339, 404, 378.5],
    [334, 422, 396],
    [328, 434, 408],
    [319, 445, 418.5],
    [314, 449, 422],
    [305, 455, 428],
    [296, 459, 432.5],
    [278, 464, 437.5],
    [267, 465, 439],
    [251, 465, 440],
    [234, 463, 437.5],
    [224, 461, 436],
    [212, 457, 432],
    [207, 455, 430.5],
    [198, 451, 427],
    [187, 445, 421],
    [177, 438, 414.5],
    [168, 430, 407],
    [160, 421, 399],
    [155, 415, 393],
    [148, 406, 384],
    [142, 399, 377],
    [133, 388, 366.5],
    [129, 381, 360],
    [127, 375, 355.5],
    [122, 357, 338],
    [117, 343, 323.5],
    [113, 334, 314],
    [110, 327, 307.5],
    [105, 316, 297],
    [101, 307, 288],
    [96, 294, 276],
    [93, 283, 264],
    [91, 276, 258],
    [89, 265, 248],
    [88, 258, 242],
    [89, 251, 229.5],
    [88, 240, 214],
    [91, 230, 208.5],
    [100, 220, 198.5],
    [106, 215, 193],
    [115, 209, 187],
    [122, 205, 184],
    [131, 200, 178.5],
    [138, 196, 174],
    [149, 191, 169],
    [157, 188, 166],
    [165, 185, 163.5],
    [174, 183, 161],
    [183, 181, 159],
    [190, 179, 157],
    [199, 177, 154.5],
    [212, 173, 150],
    [223, 170, 147],
    [231, 168, 145],
    [242, 166, 143],
    [249, 165, 142],
    [259, 163, 140],
    [269, 161, 138],
    [279, 159, 135],
    [289, 156, 132],
    [299, 153, 129],
    [315, 148, 124],
    [332, 143, 118.5],
    [345, 139, 114],
    [358, 135, 110],
    [365, 133, 107.5],
    [375, 129, 103.5],
    [381, 127, 101],
    [390, 124, 98],
    [396, 123, 96.5],
    [400, 123, 96],
    [404, 124, 97],
    [403, 128, 101],
    [399, 131, 104.5],
    [393, 134, 107.5],
    [387, 136, 109.5],
    [382, 137, 110],
    [379, 137, 110],
    [376, 134, 108],
    [372, 134, 108],
    [369, 132, 106]
])

def stereo_match(img1, img2, calib):
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
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)

    img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

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
        return disp_cv, img_3D, Q, img1, img2
    else:
        img1 = np.float32(img1)
        img2 = np.float32(img2)
        thresh = 40
        pix_thresh = 210
        max_disp = 40
        
        pixels1 = np.argwhere(img1<=pix_thresh)
        pix_to_idx = {(int(pix[0]), int(pix[1])):i for i, pix in enumerate(pixels1)}
        disps = np.zeros(pixels1.shape[0])
        reliab = np.zeros(pixels1.shape[0])
        affins = np.zeros((pixels1.shape[0], 9))

        # ORDERING
        # ord1, ord2 = order_pixels()

        # ordmap1 = np.ones_like(img1) * -100
        # ordmap2 = np.ones_like(img2) * -100

        # ordmap1[ord1[0], ord1[1]] = np.indices([ord1.shape[1]])/ord1.shape[1]*100
        # ordmap2[ord2[0], ord2[1]] = np.indices([ord2.shape[1]])/ord2.shape[1]*100

        rad = 2
        c_data = 5
        c_slope = 8
        c_shift = 0.8
        # TODO deal with out-of-bounds conditions
        for i in range(pixels1.shape[0]):
            pix1 = (pixels1[i][0], pixels1[i][1])
            chunk = img1[pix1[0]-rad:pix1[0]+rad+1, pix1[1]-rad:pix1[1]+rad+1]
            seg = np.argwhere(chunk<=pix_thresh) + np.expand_dims(pixels1[i], 0) - rad

            energy = np.array([
                np.sum(
                    (img1[seg[:,0], seg[:,1]] - img2[seg[:,0], seg[:,1] - off])**2 #+ 
                    # (ordmap1[seg[:,0], seg[:,1]] - ordmap2[seg[:,0], seg[:,1] - off])**2
                ) for off in range(max_disp)
            ])
            
            best = np.min(energy)#to_r/(np.min(energy) + 1e-7)
            disp = np.argmin(energy)
            energy2 = np.delete(energy, slice(disp-1, disp+2))
            next_best = np.min(energy2)#to_r/(np.min(energy) + 1e-7)
            
            disps[i] = disp
            x = (next_best - best)/((best + 1e-7)*c_data)
            reliab[i] = 1/(1+np.exp(np.clip(-1*c_slope*(x-c_shift), -87, None)))

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
        # RELIABILITY
        # plt.figure(3)
        # heatmap_r = plt.pcolor(reliab_map)
        # plt.colorbar(heatmap_r)
        # plt.gca().invert_yaxis()
        plt.figure(3)
        ax = plt.axes(projection='3d')
        ax.view_init(0, 0)
        ax.set_zlim(0, 1000)
        depth = np.matmul(
            Q,
            np.stack((pixels1[:, 0], pixels1[:, 1], F, np.ones_like(F)))
        )
        depth /= np.expand_dims(depth[3].copy(), 0)
        ax.scatter(
            pixels1[:, 0],
            pixels1[:, 1],
            depth[2],
            s=1
        )
        ax.scatter(
            pixels1[:, 0],
            pixels1[:, 1],
            img_3D[pixels1[:, 0], pixels1[:, 1], 2],
            c="r",
            s=1
        )
        gt_depth = np.matmul(
            Q,
            np.stack((gt[:, 0], gt[:, 1], gt[:, 1]-gt[:, 2], np.ones_like(gt[:, 0])))
        )
        gt_depth /= np.expand_dims(gt_depth[3].copy(), 0)
        ax.scatter(
            gt[:, 0],
            gt[:, 1],
            gt_depth[2],
            c="g"
        )
        # DIFFERENCES
        # plt.figure(3)
        # disp_cv_map = np.zeros_like(disp_cv)
        # disp_cv_map[pixels1[:, 0], pixels1[:, 1]] += disp_cv[pixels1[:, 0], pixels1[:, 1]]
        # diff = disp_map - disp_cv_map
        # heatmap = plt.pcolor(diff, vmin=-5, vmax=5)
        # plt.colorbar(heatmap)
        # plt.gca().invert_yaxis()
        plt.show()
            

    


if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_3_left.jpg"
    file2 = "../Sarah_imgs/thread_3_right.jpg"
    file3 = "../Sarah_imgs/thread_3_left_seg.jpg"
    file4 = "../Sarah_imgs/thread_3_right_seg.jpg"

    ref1 = cv2.imread(file1)
    ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2GRAY)
    ref2 = cv2.imread(file2)
    ref2 = cv2.cvtColor(ref2, cv2.COLOR_BGR2GRAY)
    seg1 = cv2.imread(file3)
    seg1 = cv2.cvtColor(seg1, cv2.COLOR_BGR2GRAY)
    seg2 = cv2.imread(file4)
    seg2 = cv2.cvtColor(seg2, cv2.COLOR_BGR2GRAY)

    # plt.figure(1)
    # plt.imshow(seg1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(seg2, cmap="gray")
    # plt.show()

    segpix1 = np.argwhere(seg1<230)
    segpix2 = np.argwhere(seg2<235)
    img1 = np.ones_like(ref1)*255
    img2 = np.ones_like(ref2)*255
    img1[segpix1[:, 0], segpix1[:, 1]] = ref1[segpix1[:, 0], segpix1[:, 1]]
    img2[segpix2[:, 0], segpix2[:, 1]] = ref2[segpix2[:, 0], segpix2[:, 1]]

    # plt.figure(1)
    # plt.imshow(img1, cmap="gray")
    # plt.figure(2)
    # plt.imshow(img2, cmap="gray")
    # plt.show()

    cv2.imwrite("../Sarah_imgs/thread_3_left_final.jpg", img1)
    cv2.imwrite("../Sarah_imgs/thread_3_right_final.jpg", img2)
    # file1 = "../Sarah_imgs/thread_1_left_final.jpg"
    # file2 = "../Sarah_imgs/thread_1_right_final.jpg"
    # img1 = cv2.imread(file1)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.imread(file2)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # stereo_match(img1, img2)