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

    # Active node setup
    curr_V_l = (258, 336)
    segmented_l = {curr_V_l}
    active_l = set()
                
    # Get inital active nodes
    for i, j in roi:
        node = (i+curr_V_l[0], j+curr_V_l[1])
        if (node not in segmented_l) and img_l[node] < thresh:
            active_l.add(node)
    print(active_l)