import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.interpolate as interp
import cv2
import os

from segmentation import segmentation
from keypt_selection import keypt_selection
from keypt_ordering import keypt_ordering
from optim import optim
from utils import *

SIMULATION = False
STORE = False

def fit_eval(img1, img2, calib, gt_tck=None):
    # Read in camera matrix
    cv_file = cv2.FileStorage(calib, cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    ImageSize = cv_file.getNode("ImageSize").mat()
    img_size = (int(ImageSize[0][1]), int(ImageSize[0][0]))
    new_size = (640, 480)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new_size)
    
    cam2img = P1[:,:-1]
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, new_size, cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, new_size, cv2.CV_16SC2)

    img1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    
    mask1 = segmentation(img1)
    mask2 = segmentation(img2)
    

    # plt.figure(1)
    # plt.imshow(mask1.astype("float"))
    # plt.figure(2)
    # plt.imshow(mask2.astype("float"))
    # plt.show()
    # plt.clf()
    # plt.figure(1)
    # plt.clf()

    stack_mask1 = np.stack((mask1, mask1, mask1), axis=-1)
    img1 = np.where(stack_mask1>0, img1, 0)
    stack_mask2 = np.stack((mask2, mask2, mask2), axis=-1)
    img2 = np.where(stack_mask2>0, img2, 0)
    
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    
    # Perform reconstruction
    img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents = keypt_selection(img1, img2, mask1, mask2, Q)
    img_3D, keypoints, grow_paths, order = keypt_ordering(img1, img_3D, clusters, cluster_map, keypoints, grow_paths, adjacents)
    final_tck = optim(img1, mask1, mask2, img_3D, keypoints, grow_paths, order, cam2img, P1, P2)
    final_tck.c = change_coords(final_tck.c, P1[:, :3])
    final_spline = final_tck(np.linspace(final_tck.t[0], final_tck.t[-1], 150))
    return

    # Evaluate reconstruction accuracy
    if SIMULATION:
        gt_spline = gt_tck(np.linspace(0, 1, 150))

        ours_len, gt_len, diff = length_error(final_tck, gt_tck)
        print("Lengths: ours %f, gt %f, diff %f" % (ours_len, gt_len, diff))

        num_eval_pts = int(gt_len*10)
        errors1, spots1, e_mean1, e_max1 = curve_error(final_tck, gt_tck, num_eval_pts)
        errors2, spots2, e_mean2, e_max2 = curve_error(gt_tck, final_tck, num_eval_pts)
        e_mean = (e_mean1 + e_mean2)/2
        e_max = max(e_max1, e_max2)
        print("Curve error: mean %f, max %f" % (e_mean, e_max))
    else:
        num_eval_pts = 200
        left, left_max = reprojection_error(final_tck, mask1, P1, num_eval_pts)
        right, right_max = reprojection_error(final_tck, mask2, P2, num_eval_pts)
        print("Reprojection error: mean left %f, max left %f, mean right %f, max right %f" \
            % (left, left_max, right, right_max))

    # Visualize the result
    plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.tick_params(labelsize=8)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_zlabel("$z$")
    ax1.plot(
        final_spline[:, 0],
        final_spline[:, 1],
        final_spline[:, 2],
        c="r",
        label="Our Result"
    )
    if gt_tck is not None:
        ax1.plot(
            gt_spline[:, 0],
            gt_spline[:, 1],
            gt_spline[:, 2],
            label="Ground Truth",
            c="g")
        ax1.legend()
    set_axes_equal(ax1)
    if not STORE:
        plt.show()
    if SIMULATION:
        return ours_len, gt_len, diff, e_mean, e_max
    else:
        return left, left_max, right, right_max





if __name__ == "__main__":
    inp_folder = os.path.dirname(__file__) + "/../../thread_segmentation/thread_2/"
    prefixes = ["left_recif_", "right_recif_"]
    start = 0
    ext = ".jpg"
    calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
    for i in range(78, 279, 10):
        print(start+i)
        imfile1 = inp_folder+prefixes[0]+str(start+i)+ext
        img1 = cv2.imread(imfile1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        imfile2 = inp_folder+prefixes[1]+str(start+i)+ext
        img2 = cv2.imread(imfile2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        try:
            fit_eval(img1, img2, calib)
        except Exception as e:
            print("FAILED: " + str(e))
    """
    # Run reconstruction on datasets
    # Simulated Dataset
    if SIMULATION:
        data = []
        header = ["file", "ours_len", "gt_len", "diff", "e_mean", "e_max"]
        footer = ["Failed"]
        for folder_num in range(1,11):
            for file_num in range(1,5):
                # Choose correct calibration matrix
                if folder_num < 5:
                    fileb = "../Blender_imgs/blend%d/blend%d_%d.jpg" % (folder_num, folder_num, file_num)
                    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
                else:
                    fileb = "../Blender_imgs/blend%d/blend%d_%d.png" % (folder_num, folder_num, file_num)
                    calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration_new.yaml"
                # Extract and color segment left and right images
                imgb = cv2.imread(fileb)
                imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
                img1 = imgb[:, :640]
                img2 = imgb[:, 640:]
                img1 = np.where(img1>=200, 255, img1)
                img2 = np.where(img2>=200, 255, img2)

                gt_b = np.load("/Users/neelay/ARClabXtra/Blender_imgs/blend%d/blend%d_%d.npy" % (folder_num, folder_num, file_num))
                cv_file = cv2.FileStorage("/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml", cv2.FILE_STORAGE_READ)
                gk = 3
                gt_knots = np.concatenate(
                    (np.repeat(0, gk),
                    np.linspace(0, 1, gt_b.shape[0]-gk+1),
                    np.repeat(1, gk))
                )
                gt_tck = interp.BSpline(gt_knots, gt_b, gk)
                gt_spline = gt_tck(np.linspace(0, 1, 150))

                try:
                    out = list(fit_eval(img1, img2, calib, gt_tck))
                    out = ["%d_%d" % (folder_num, file_num)] + out
                    data.append(out)
                except:
                    footer.append("%d_%d" % (folder_num, file_num))
        # Store results conveniently in csv file
        if STORE:
            with open("results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)
    # Real dataset        
    else:
        files = [(1, 99), (1, 119), (2, 59), (2, 72), (2, 116), \
            (2, 149), (2, 159), (2,174), (2, 187), (2, 209)]
        data = []
        header = ["file", "left mean err", "left max error", "right mean err", "right max err"]
        footer = ["Failed"]
        for folder_num, file_num in files:
            file1 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_left_%d_final.png" % (folder_num, folder_num, file_num)
            file2 = "../Suture_Thread_06_16/thread_%d_seg/thread%d_right_%d_final.png" % (folder_num, folder_num, file_num)
            calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
            img1 = cv2.imread(file1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.imread(file2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # try:
            out = list(fit_eval(img1, img2, calib))
            out = ["%d_%d" % (folder_num, file_num)] + out
            data.append(out)
            # except:
            footer.append("%d_%d" % (folder_num, file_num))
        if STORE:
            with open("results_real.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
                writer.writerow(footer)
    """