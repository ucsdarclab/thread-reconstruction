import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import random
np.seterr(all="raise")
"""
Tip locations:
    Left- x=337, y=259 (bottom rightmost pixel)
    Right- x=313, y=258 (bottom rightmost pixel)
"""
if __name__ == "__main__":
    #""" #For finding image pixels
    img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_left_rembg.png")
    # TODO Uncomment
    plt.imshow(img)
    # plt.show()
    # """
    # Set up image and useful constants
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    thresh = 210
    upsilon = 5
    roi = []
    for i in range(-upsilon, upsilon+1):
        for j in range(-upsilon, upsilon+1):
            if i**2 + j**2 <= upsilon**2:
                roi.append((i,j))
    
    dilate_kernel = np.ones((5, 5), np.uint8)
    erode_kernel = np.ones((2, 2), np.uint8)
    #img_l_dilated = cv2.dilate(np.where(img_l <= thresh, 1, 0).astype("uint8"), dilate_kernel, iterations=2)
    img_l *= cv2.erode(np.where(img_l <= thresh, 1, 0).astype("uint8"), erode_kernel, iterations=1)
    img_l =np.where(img_l == 0, 255, img_l)
    # cv2.imshow("img", img_l)
    # cv2.waitKey(5000)
    # plt.imshow(img_l)
    # plt.show()
    # exit(0)
    
    curr_V_l = (259, 336)
    par_V_l = (259, 337)
    curve_set_l = {curr_V_l, par_V_l}
    curve_l = np.array([
        [par_V_l[1], curr_V_l[1]],
        [par_V_l[0], curr_V_l[0]]
    ])

    # Pixel ordering setup
    active_l = []
    for i, j in roi:
        node_l = (i+curr_V_l[0], j+curr_V_l[1])
        if ((node_l[0] < img_l.shape[0]) and (node_l[1] < img_l.shape[1]) and
            (node_l not in curve_set_l) and (img_l[node_l] <= thresh)
            ):
            active_l.append(node_l)
    
    # Order pixels and stereo match
    e_1 = 1
    e_2 = 0.1
    e_3 = 1.2
    mu = 1
    tau_O = 2.6 * upsilon
    tau_V = 2*math.pi/3
    while len(active_l):
        # calculate min cost active node
        min_cost_l = np.Inf
        min_nodes_l = []
        for prow_l, pcol_l in active_l:
            # Calculate triangle area terms
            to_active_l = np.array([prow_l, pcol_l]) - np.array([curr_V_l[0], curr_V_l[1]])
            to_prev_l = np.array([curr_V_l[0], curr_V_l[1]]) - np.array([par_V_l[0], par_V_l[1]])
            angle_l = np.arccos(np.clip(
                np.dot(to_active_l, to_prev_l) /
                (np.linalg.norm(to_active_l) * np.linalg.norm(to_prev_l)),
                -1,
                1
            ))
            # Compare with terminating threshold
            if (angle_l >= tau_V):
                continue
            
            # calculate out of range number
            o_pixels_l = set()
            delta_row_l = prow_l - curr_V_l[0]
            delta_col_l = pcol_l - curr_V_l[1]
            for i in range(abs(delta_row_l)):
                sign = 1 if delta_row_l > 0 else -1
                drow_l = (0.5 + i) * sign
                dcol_l = delta_col_l/delta_row_l * drow_l
                pixel1_l = (curr_V_l[0] + i*sign, round(curr_V_l[1] + dcol_l))
                pixel2_l = (pixel1_l[0] + 1*sign, pixel1_l[1])
                if img_l[pixel1_l] > thresh:
                    o_pixels_l.add(pixel1_l)
                if img_l[pixel2_l] > thresh:
                    o_pixels_l.add(pixel2_l)

            for i in range(abs(delta_col_l)):
                sign = 1 if delta_col_l > 0 else -1
                dcol_l = (0.5 + i) * sign
                drow_l = delta_row_l/delta_col_l * dcol_l
                pixel1_l = (round(curr_V_l[0] + drow_l), curr_V_l[1] + i*sign)
                pixel2_l = (pixel1_l[0], pixel1_l[1] + 1*sign)
                if img_l[pixel1_l] > thresh:
                    o_pixels_l.add(pixel1_l)
                if img_l[pixel2_l] > thresh:
                    o_pixels_l.add(pixel2_l)
            
            O_num_l = len(o_pixels_l)
            # Compare with terminating threshold
            if (O_num_l >= tau_O):
                continue

            # Calculate node cost and compare to min cost
            cost_l = (
                math.log(e_1*O_num_l + 1) + 
                e_2 * np.linalg.norm(to_active_l) +
                np.exp(e_3 * np.sin(angle_l/2))
            )
            if (cost_l < min_cost_l):
                min_cost_l = cost_l
                min_nodes_l = [(prow_l, pcol_l)]
            elif (cost_l == min_cost_l):
                min_nodes_l.append((prow_l, pcol_l))
        
        # Terminate if conditions all tripped
        if (len(min_nodes_l) == 0):
            break
        # Add selected node to curve
        min_node_l = min_nodes_l[random.randrange(0, len(min_nodes_l))]
        #for min_node_l in min_nodes_l:
        curve_set_l.add(min_node_l)
        curve_l = np.concatenate(
            (
                curve_l,
                np.array([[min_node_l[1]], [min_node_l[0]]])
            ),
            axis=1
        )

        # Update active nodes and curve nodes
        par_V_l = curr_V_l
        curr_V_l = min_node_l #min_nodes_l[random.randrange(0, len(min_nodes_l))]
        active_l = []
        for i, j in roi:
            node_l = (i+curr_V_l[0], j+curr_V_l[1])
            if ((node_l[0] < img_l.shape[0]) and (node_l[1] < img_l.shape[1]) and
                (node_l not in curve_set_l) and (img_l[node_l] <= thresh)
                ):
                active_l.append(node_l)
    
    plt.scatter(curve_l[0], curve_l[1], c=np.linspace(0, curve_l.shape[1]-1, curve_l.shape[1]), cmap="hot")
    plt.show()