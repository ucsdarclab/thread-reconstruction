import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
"""
blend1:
np.array([
    [[273, 522],[273, 523]],
    [[302, 570],[302, 571]],
    [[171, 607],[172, 607]],
    [[194, 238],[194, 237]]
])
np.array([
    [[273, 510],[273, 511]],
    [[302, 559],[302, 560]],
    [[171, 587],[172, 587]],
    [[194, 229],[194, 228]]
])
"""
if __name__ == "__main__":
    folder_num = 1
    folder = "../Blender_imgs/blend%d/" % (folder_num,)
    # calib = "/Users/neelay/ARClabXtra/Blender_imgs/blend_calibration.yaml"
    # files = [folder + "blend%d_%d.jpg" % (folder_num,i) for i in range(1,5)]
    # for fileb in files:
    #     imgb = cv2.imread(fileb)
    #     imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    #     img1 = imgb[:, :640]
    #     img2 = imgb[:, 640:]
    #     img1 = np.where(img1>=200, 255, img1)
    #     img2 = np.where(img2>=200, 255, img2)

    #     plt.imshow(img1)
    #     plt.show()
    #     plt.clf()
    #     plt.imshow(img2)
    #     plt.show()
    left_starts = np.array([
    [[273, 522],[273, 523]],
    [[302, 570],[302, 571]],
    [[171, 607],[172, 607]],
    [[194, 238],[194, 237]]
    ])
    right_starts = np.array([
        [[273, 510],[273, 511]],
        [[302, 559],[302, 560]],
        [[171, 587],[172, 587]],
        [[194, 229],[194, 228]]
    ])

    np.save(folder + "left%d.npy" % (folder_num,), left_starts)
    np.save(folder + "right%d.npy" % (folder_num,), right_starts)

    print(np.load(folder + "left%d.npy" % (folder_num,)))
    print(np.load(folder + "right%d.npy" % (folder_num,)))