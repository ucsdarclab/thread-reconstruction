import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as manimation
import os

def segmentation(img):
    unet = torch.load(os.path.dirname(__file__) + "/segmenter.pt")
    inp = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    inp = torch.from_numpy(inp)
    inp = inp.permute(2,0,1)
    out = unet(inp.unsqueeze(0))
    mask = out.squeeze().detach().numpy()
    # mask = np.clip(mask, 0, 1)
    mask = np.where(mask>=0.5, 1, 0)
    return mask

if __name__ == "__main__":
    pass