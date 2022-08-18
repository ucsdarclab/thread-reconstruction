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
    Right- x=314, y=259 (bottom rightmost pixel)
"""
def order_pixels(img_l, img_r):
    #""" #For finding image pixels
    # img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_right_rembg.png")
    # TODO Uncomment
    # plt.imshow(img)
    # plt.show()
    # exit(0)
    # """
    # Set up image and useful constants
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    # img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    # img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_l_init = img_l.copy()
    # img_r = cv2.imread(img_dir + "thread_1_right_rembg.png")
    # img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_r_init = img_r.copy()
    thresh = 226
    upsilon = 10
    roi = []
    for i in range(-upsilon, upsilon+1):
        for j in range(-upsilon, upsilon+1):
            if i**2 + j**2 <= upsilon**2:
                roi.append((i,j))
    
    
    curr_V = (1, 117)#(288, 375)
    par_V = (0, 117)#(288, 374)

    curve_set = np.zeros_like(img_l)
    curve_set[curr_V] = 1
    curve_set[par_V] = 1

    curve_l = np.array([
        [par_V[0], curr_V[0]],
        [par_V[1], curr_V[1]]
    ])
    curve_steps_l = []
    curve_steps_r = []

    # Pixel ordering setup
    active = []
    for i, j in roi:
        node = (i+curr_V[0], j+curr_V[1])
        if ((node[0] < img_l.shape[0]) and (node[1] < img_l.shape[1]) and
            (not curve_set[node]) and (img_l[node] <= thresh)
            ):
            active.append(node)
    
    # Order pixels and stereo match
    e_1 = 2
    e_2 = 0.3
    e_3 = 0.07
    e_4 = 1
    mu = 1
    tau_O = 2.6 * upsilon
    tau_V = 2*math.pi/3
    max_chunksize = 5
    sum_thresh = 7
    while len(active):
        # calculate min cost active node
        min_cost = np.Inf
        min_node = None
        glob_step = 0
        for prow, pcol in active:
            # Aggregate into chunks, growing in direction of parent
            row_dir = np.sign(curr_V[0] - prow)
            col_dir = np.sign(curr_V[1] - pcol)
            for step in range(1, max_chunksize):
                row_slice = slice(prow, prow+step+1) if row_dir > 0 else slice(prow-step, prow+1)
                col_slice = slice(pcol, pcol+step+1) if col_dir > 0 else slice(pcol-step, pcol+1)
                if (img_l[row_slice, col_slice] > thresh).any() \
                    or (curve_set[row_slice, col_slice]).any():
                    break
            
            # Prevent doubling back
            if step == 1 and np.sum(curve_set[prow-2:prow+3, pcol-2:pcol+3]) > sum_thresh:
                continue
            
            # Calculate triangle area terms
            to_active = np.array([prow, pcol]) - np.array([curr_V[0], curr_V[1]])
            to_prev = np.array([curr_V[0], curr_V[1]]) - np.array([par_V[0], par_V[1]])
            angle = np.arccos(np.clip(
                np.dot(to_active, to_prev) /
                (np.linalg.norm(to_active) * np.linalg.norm(to_prev)),
                -1,
                1
            ))
            # Compare with terminating threshold
            if (angle >= tau_V):
                continue
            
            # calculate out of range number
            o_pixels = set()
            delta_row = prow - curr_V[0]
            delta_col = pcol - curr_V[1]
            for i in range(abs(delta_row)):
                sign = 1 if delta_row > 0 else -1
                drow = (0.5 + i) * sign
                dcol = delta_col/delta_row * drow
                pixel1 = (curr_V[0] + i*sign, round(curr_V[1] + dcol))
                pixel2 = (pixel1[0] + 1*sign, pixel1[1])
                if img_l[pixel1] > thresh:
                    o_pixels.add(pixel1)
                if img_l[pixel2] > thresh:
                    o_pixels.add(pixel2)

            for i in range(abs(delta_col)):
                sign = 1 if delta_col > 0 else -1
                dcol = (0.5 + i) * sign
                drow = delta_row/delta_col * dcol
                pixel1 = (round(curr_V[0] + drow), curr_V[1] + i*sign)
                pixel2 = (pixel1[0], pixel1[1] + 1*sign)
                if img_l[pixel1] > thresh:
                    o_pixels.add(pixel1)
                if img_l[pixel2] > thresh:
                    o_pixels.add(pixel2)
            
            O_num = len(o_pixels)
            # Compare with terminating threshold
            if (O_num >= tau_O):
                continue

            # Calculate node cost and compare to min cost
            cost = (
                math.log(e_1*O_num + 1) + 
                e_2 * np.linalg.norm(to_active) *
                np.exp(e_3 * np.sin(angle/2))
            )/(e_4 * step)**2
            if (cost < min_cost):
                min_cost = cost
                min_node = (prow, pcol)
                glob_step = step
            # elif (cost == min_cost):
            #     min_nodes.append((prow, pcol))
        
        # Terminate if conditions all tripped
        if (min_node is None):
            break
        prow, pcol = min_node
        row_dir = np.sign(curr_V[0] - prow)
        col_dir = np.sign(curr_V[1] - pcol)
        row_range = range(prow, prow+glob_step) if row_dir > 0 else range(prow-glob_step+1, prow+1)
        col_range = range(pcol, pcol+glob_step) if col_dir > 0 else range(pcol-glob_step+1, pcol+1)
        for r in reversed(row_range):
            for c in reversed(col_range):
                curve_set[r, c] = 1
                curve_l = np.concatenate(
                    (
                        curve_l,
                        np.array([[r], [c]])
                    ),
                    axis=1
                )
        curve_steps_l.append(glob_step)

        # Update active nodes and curve nodes
        par_V = curr_V
        curr_V = min_node #min_nodes[random.randrange(0, len(min_nodes))]
        active = []
        for i, j in roi:
            node = (i+curr_V[0], j+curr_V[1])
            if ((node[0] < img_l.shape[0]) and (node[1] < img_l.shape[1]) and
                (not curve_set[node]) and (img_l[node] <= thresh)
                ):
                active.append(node)

    thresh = 238
    
    curr_V = (1, 93)#(288, 349)
    par_V = (0, 93)#(288, 348)

    curve_set = np.zeros_like(img_r)
    curve_set[curr_V] = 1
    curve_set[par_V] = 1

    curve_r = np.array([
        [par_V[0], curr_V[0]],
        [par_V[1], curr_V[1]]
    ])

    # Pixel ordering setup
    active = []
    for i, j in roi:
        node = (i+curr_V[0], j+curr_V[1])
        if ((node[0] < img_r.shape[0]) and (node[1] < img_r.shape[1]) and
            (not curve_set[node]) and (img_r[node] <= thresh)
            ):
            active.append(node)
    
    # Order pixels and stereo match
    e_1 = 2
    e_2 = 0.3
    e_3 = 0.07
    e_4 = 1
    mu = 1
    tau_O = 2.6 * upsilon
    tau_V = 2*math.pi/3
    max_chunksize = 5
    sum_thresh = 7
    while len(active):
        # calculate min cost active node
        min_cost = np.Inf
        min_node = None
        glob_step = 0
        for prow, pcol in active:
            # Aggregate into chunks, growing in direction of parent
            row_dir = np.sign(curr_V[0] - prow)
            col_dir = np.sign(curr_V[1] - pcol)
            for step in range(1, max_chunksize):
                row_slice = slice(prow, prow+step+1) if row_dir > 0 else slice(prow-step, prow+1)
                col_slice = slice(pcol, pcol+step+1) if col_dir > 0 else slice(pcol-step, pcol+1)
                if (img_r[row_slice, col_slice] > thresh).any() \
                    or (curve_set[row_slice, col_slice]).any():
                    break
            
            # Prevent doubling back
            if step == 1 and np.sum(curve_set[prow-2:prow+3, pcol-2:pcol+3]) > sum_thresh:
                continue
            
            # Calculate triangle area terms
            to_active = np.array([prow, pcol]) - np.array([curr_V[0], curr_V[1]])
            to_prev = np.array([curr_V[0], curr_V[1]]) - np.array([par_V[0], par_V[1]])
            angle = np.arccos(np.clip(
                np.dot(to_active, to_prev) /
                (np.linalg.norm(to_active) * np.linalg.norm(to_prev)),
                -1,
                1
            ))
            # Compare with terminating threshold
            if (angle >= tau_V):
                continue
            
            # calculate out of range number
            o_pixels = set()
            delta_row = prow - curr_V[0]
            delta_col = pcol - curr_V[1]
            for i in range(abs(delta_row)):
                sign = 1 if delta_row > 0 else -1
                drow = (0.5 + i) * sign
                dcol = delta_col/delta_row * drow
                pixel1 = (curr_V[0] + i*sign, round(curr_V[1] + dcol))
                pixel2 = (pixel1[0] + 1*sign, pixel1[1])
                if img_r[pixel1] > thresh:
                    o_pixels.add(pixel1)
                if img_r[pixel2] > thresh:
                    o_pixels.add(pixel2)

            for i in range(abs(delta_col)):
                sign = 1 if delta_col > 0 else -1
                dcol = (0.5 + i) * sign
                drow = delta_row/delta_col * dcol
                pixel1 = (round(curr_V[0] + drow), curr_V[1] + i*sign)
                pixel2 = (pixel1[0], pixel1[1] + 1*sign)
                if img_r[pixel1] > thresh:
                    o_pixels.add(pixel1)
                if img_r[pixel2] > thresh:
                    o_pixels.add(pixel2)
            
            O_num = len(o_pixels)
            # Compare with terminating threshold
            if (O_num >= tau_O):
                continue

            # Calculate node cost and compare to min cost
            cost = (
                math.log(e_1*O_num + 1) + 
                e_2 * np.linalg.norm(to_active) *
                np.exp(e_3 * np.sin(angle/2))
            )/(e_4 * step)**2
            if (cost < min_cost):
                min_cost = cost
                min_node = (prow, pcol)
                glob_step = step
            # elif (cost == min_cost):
            #     min_nodes.append((prow, pcol))
        
        # Terminate if conditions all tripped
        if (min_node is None):
            break
        prow, pcol = min_node
        row_dir = np.sign(curr_V[0] - prow)
        col_dir = np.sign(curr_V[1] - pcol)
        row_range = range(prow, prow+glob_step) if row_dir > 0 else range(prow-glob_step+1, prow+1)
        col_range = range(pcol, pcol+glob_step) if col_dir > 0 else range(pcol-glob_step+1, pcol+1)
        for r in reversed(row_range):
            for c in reversed(col_range):
                curve_set[r, c] = 1
                curve_r = np.concatenate(
                    (
                        curve_r,
                        np.array([[r], [c]])
                    ),
                    axis=1
                )
        curve_steps_r.append(glob_step)

        # Update active nodes and curve nodes
        par_V = curr_V
        curr_V = min_node #min_nodes[random.randrange(0, len(min_nodes))]
        active = []
        for i, j in roi:
            node = (i+curr_V[0], j+curr_V[1])
            if ((node[0] < img_r.shape[0]) and (node[1] < img_r.shape[1]) and
                (not curve_set[node]) and (img_r[node] <= thresh)
                ):
                active.append(node)

    # plt.imshow(img_l_init, cmap="gray")
    # plt.scatter(curve_l[1], curve_l[0], c=np.linspace(0, curve_l.shape[1]-1, curve_l.shape[1]), cmap="hot")
    # # plt.show()
    # plt.figure(2)
    # plt.imshow(img_r_init, cmap="gray")
    # plt.scatter(curve_r[1], curve_r[0], c=np.linspace(0, curve_r.shape[1]-1, curve_r.shape[1]), cmap="hot")
    # plt.show()

    return curve_l, curve_r, curve_steps_l, curve_steps_r

if __name__ == "__main__":
    file1 = "../Sarah_imgs/thread_1_left_rembg.png"#sys.argv[1]
    file2 = "../Sarah_imgs/thread_1_right_rembg.png"#sys.argv[2]
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(file2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    order_pixels(img1, img2)