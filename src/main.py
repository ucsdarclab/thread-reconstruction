import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

"""
Tip locations:
    Left- x=336, y=258 (bottom rightmost pixel)
    Right- x=313, y=258 (bottom rightmost pixel)
"""
if __name__ == "__main__":
    """ #For finding image pixels
    img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_left_rembg.png")
    plt.imshow(img)
    plt.show()
    """
    # Set up image and useful constants
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img_l = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(img_dir + "thread_1_right_rembg.png")
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    thresh = 250
    upsilon = 5
    roi = []
    for i in range(-upsilon, upsilon+1):
        for j in range(-upsilon, upsilon+1):
            if i**2 + j**2 <= upsilon**2:
                roi.append((i,j))

    # Pixel ordering setup
    curr_V_l = (258, 336)
    par_V_l = None
    segmented_l = {curr_V_l}
    active_l = []
    for i, j in roi:
        node_l = (i+curr_V_l[0], j+curr_V_l[1])
        if ((node_l[0] < img_l.shape[0]) and (node_l[1] < img_l.shape[1]) and
            (node_l not in segmented_l) and (img_l[node_l] <= thresh)
            ):
            active_l.append(node_l)
    
    # Order pixels and stereo match
    while len(active_l):
        # calculate min cost active node
        min_cost_l = np.Inf
        for prow_l, pcol_l in active_l:
            # calculate out of range number
            o_pixels = set()
            delta_row_l = prow_l - curr_V_l[0]
            delta_col_l = pcol_l - curr_V_l[1]
            for i in range(abs(delta_row_l)):
                sign = 1 if delta_row_l > 0 else -1
                drow_l = (0.5 + i) * sign
                dcol_l = delta_col_l/delta_row_l * drow_l
                pixel1_l = (curr_V_l[0] + i*sign, round(curr_V_l[1] + dcol_l))
                pixel2_l = (pixel1_l[0] + 1*sign, pixel1_l[1])
                if img_l[pixel1_l] > thresh:
                    o_pixels.add(pixel1_l)
                if img_l[pixel2_l] > thresh:
                    o_pixels.add(pixel2_l)

            for i in range(abs(delta_col_l)):
                sign = 1 if delta_col_l > 0 else -1
                dcol_l = (0.5 + i) * sign
                drow_l = delta_row_l/delta_col_l * dcol_l
                pixel1_l = (round(curr_V_l[0] + drow_l), curr_V_l[1] + i*sign)
                pixel2_l = (pixel1_l[0], pixel1_l[1] + 1*sign)
                if img_l[pixel1_l] > thresh:
                    o_pixels.add(pixel1_l)
                if img_l[pixel2_l] > thresh:
                    o_pixels.add(pixel2_l)
            
            O_num = len(o_pixels)
        break