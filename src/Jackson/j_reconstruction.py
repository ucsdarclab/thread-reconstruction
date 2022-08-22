import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
# from pixel_ordering import order_pixels
import scipy.interpolate as interp
from heapq import heappush, heappop

def j_reconstruction(img1, img2, calib, left_start, right_start, folder_num, file_num):
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
    
    frontier1 = [(0,tuple(left_start))]
    frontier2 = [(0,tuple(right_start))]
    explored1 = np.zeros_like(img1)
    explored2 = np.zeros_like(img1)
    matched = {}
    prev_matched1 = None
    prev_matched2 = None
    match_dist = 5
    DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                           [1, 1], [-1, -1], [-1, 1], [1, -1]])
    pix_thresh = 250
    while len(frontier1) > 0 and len(frontier2) > 0:
        cost1, curr1 = heappop(frontier1)
        cost2, curr2 = heappop(frontier2)
        curr1 = np.array(curr1)
        curr2 = np.array(curr2)
        if (prev_matched1 is None or np.linalg.norm(prev_matched1 - curr1) > match_dist) and \
            (prev_matched2 is None or np.linalg.norm(prev_matched2 - curr2) > match_dist) and \
            curr1[0] - curr2[0] < 2:
            matched[tuple(curr1)] = curr1[1] - curr2[1]
            prev_matched1 = curr1
            prev_matched2 = curr2
            cost1 = 0
            cost2 = 0
        explored1[curr1[0], curr1[1]] = 1
        explored2[curr2[0], curr2[1]] = 1

        for d in DIRECTIONS:
            neigh1 = curr1 + d
            neigh2 = curr2 + d
            # Check OOB condition
            if not (neigh1[0] < 0 or neigh1[0] >= img1.shape[0] or \
                neigh1[1] < 0 or neigh1[1] >= img1.shape[1]):
                # if neighbor is segmented, unvisited
                val = img1[neigh1[0], neigh1[1]]
                if val <= pix_thresh and \
                    explored1[neigh1[0], neigh1[1]] == 0:
                    cost = cost1 + val / (255 - val)**2.3
                    cost *= np.linalg.norm(d)
                    heappush(frontier1, (cost.item(), tuple(neigh1)))
            # Check OOB condition
            if not (neigh2[0] < 0 or neigh2[0] >= img2.shape[0] or \
                neigh2[1] < 0 or neigh2[1] >= img2.shape[1]):
                # if neighbor is segmented, unvisited
                val = img2[neigh2[0], neigh2[1]]
                if val <= pix_thresh and \
                    explored2[neigh2[0], neigh2[1]] == 0:
                    cost = cost2 + val / (255 - val)**2.3
                    cost *= np.linalg.norm(d)
                    heappush(frontier2, (cost.item(), tuple(neigh2)))
        
    depths = np.zeros((len(matched), 3))
    for i, (pix, disp) in enumerate(matched.items()):
        depth_calc = np.array([pix[0], pix[1], disp, 1])
        depth_calc = np.matmul(Q, depth_calc)
        depth_calc[2] /= depth_calc[3].copy() + 1e-7
        depths[i] = depth_calc[:3]
    gt_b = np.load("/Users/neelay/ARClabXtra/Blender_imgs/blend%d/blend%d_%d.npy" % (folder_num, folder_num, file_num))
    cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml", cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    m2pix = K1[0, 0] / 50e-3
    gt_pix = np.matmul(K1, gt_b.T).T
    gt_b[:, :2] = gt_pix[:, :2] / gt_pix[:, 2:]
    gk = 3
    gt_knots = np.concatenate(
        (np.repeat(0, gk),
        np.linspace(0, 1, gt_b.shape[0]-gk+1),
        np.repeat(1, gk))
    )
    gt_tck = interp.BSpline(gt_knots, gt_b, gk)
    gt_spline = gt_tck(np.linspace(0, 1, 150))

    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 20)
    ax.set_xlim(0, 480)
    ax.set_ylim(0, 640)
    ax.scatter(depths[:, 0], depths[:, 1], depths[:, 2])
    ax.plot(
        gt_spline[:, 1],
        gt_spline[:, 0],
        gt_spline[:, 2],
        c="g")
    plt.show()


if __name__ == "__main__":
#     file1 = "../Sarah_imgs/thread_3_left_final.jpg"#sys.argv[1]
#     file2 = "../Sarah_imgs/thread_3_right_final.jpg"#sys.argv[2]
#     img1 = cv2.imread(file1)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.imread(file2)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     calib = "/Users/neelay/ARClabXtra/Sarah_imgs/camera_calibration_fei.yaml"
    # plt.imshow(img1)
    # plt.show()
    # plt.clf()
    # plt.imshow(img2)
    # plt.show()
    # assert False
    folder_num = 1
    file_num = 2
    fileb = "../Blender_imgs/blend%d/blend%d_%d.jpg" % (folder_num, folder_num, file_num)
    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
    imgb = cv2.imread(fileb)
    imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    img1 = imgb[:, :640]
    img2 = imgb[:, 640:]
    img1 = np.where(img1>=200, 255, img1)
    img2 = np.where(img2>=200, 255, img2)
    left_starts = np.load("../Blender_imgs/blend%d/left%d.npy" % (folder_num, folder_num))
    right_starts = np.load("../Blender_imgs/blend%d/right%d.npy" % (folder_num, folder_num))
    j_reconstruction(img1, img2, calib, left_starts[file_num-1][0], right_starts[file_num-1][0], folder_num, file_num)