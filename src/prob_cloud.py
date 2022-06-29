import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg
import numpy as np
import cv2
from stereo_matching import stereo_match

def prob_cloud():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    img = cv2.imread(img_dir + "thread_1_left_rembg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_3D = stereo_match()
    thresh = 226
    seg_pix = np.argwhere(img <= thresh)

    rng = np.random.default_rng()
    sample = seg_pix[rng.choice(seg_pix.shape[0], seg_pix.shape[0]//10)]
    # test = np.ones_like(img) * 255
    # test[sample[:, 0], sample[:, 1]] = 0
    # plt.imshow(test)
    # plt.show()
    rad = 6
    # TODO try sampling
    # means = np.zeros_like(img)
    # varians = np.zeros_like(means)
    cloud = np.zeros((5*seg_pix.shape[0], 3))
    for i, pix in enumerate(seg_pix):
        roi = img[pix[0]-rad:pix[0]+rad+1, pix[1]-rad:pix[1]+rad+1]
        roi_seg = np.argwhere(roi <= thresh) + np.expand_dims(pix, 0) - rad
        roi_depths = img_3D[roi_seg[:, 0], roi_seg[:, 1], 2]
        roi_depths = roi_depths[roi_depths != np.inf]
        roi_depths = roi_depths[roi_depths >0]
        mean = np.mean(roi_depths)
        std = np.std(roi_depths)
        cloud[5*i] = np.array([pix[0], pix[1], mean-2*std])
        cloud[5*i+1] = np.array([pix[0], pix[1], mean-std])
        cloud[5*i+2] = np.array([pix[0], pix[1], mean])
        cloud[5*i+3] = np.array([pix[0], pix[1], mean+std])
        cloud[5*i+4] = np.array([pix[0], pix[1], mean+2*std])
    ax = plt.axes(projection="3d")
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2)
    ax.scatter(
        seg_pix[:, 0],
        seg_pix[:, 1],
        img_3D[seg_pix[:, 0], seg_pix[:, 1], 2],
        s=2, c="r")
    ax.set_zlim(0, 1000)
    plt.show()
        
    
    
    # means = np.expand_dims(means, -1)
    # varians = np.expand_dims(varians, -1)
    # X, Y, Z = np.mgrid[240:280, 320:360, 200:600:30j]
    # values = np.zeros_like(X)#1/np.sqrt(2*np.pi*varians + 1e-7) * np.exp(-0.5 * (Z - means)**2 / (varians + 1e-7))
    # valid = np.argwhere(np.squeeze(varians) > 1e-7)
    # values[valid[:, 0], valid[:, 1], :] = (
    #     1/np.sqrt(2*np.pi*varians[valid[:, 0], valid[:, 1]]) * \
    #     np.exp(-0.5 * (Z[0, 0] - \
    #     means[valid[:, 0], valid[:, 1]])**2 / varians[valid[:, 0], valid[:, 1]])
    # )

    # fig = go.Figure(data=go.Volume(
    #     x=X.flatten(),
    #     y=Y.flatten(),
    #     z=Z.flatten(),
    #     value=values.flatten(),
    #     isomin=0,
    #     isomax=1,
    #     opacity=0.1,
    #     surface_count=17,
    #     ))
    # fig.show()

if __name__ == "__main__":
    prob_cloud()