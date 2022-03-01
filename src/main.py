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
    img = mpimg.imread("/Users/neelay/ARClabXtra/Sarah_imgs/thread_1_right_rembg.png")
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
    upsilon = 3
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
        if (node_l not in segmented_l) and img_l[node_l] < thresh:
            active_l.append(node_l)
    
    # Order pixels and stereo match
    while len(active_l):
        # calculate min cost active node
        min_cost = np.Inf
        for pr_l, pc_l in active_l:
            # out of range numbers
            slope = (pr_l - curr_V_l[0]) / (pc_l - curr_V_l[1])
