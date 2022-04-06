import cv2
import torch
import numpy as np

def stereo_match():
    img_dir = "/Users/neelay/ARClabXtra/Sarah_imgs/"
    cv_file = cv2.FileStorage(img_dir + "camera_calibration_fei.yaml", cv2.FILE_STORAGE_READ)
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    ImageSize = cv_file.getNode("ImageSize").mat()
    print(K1, K2, D1, D2, R, T, ImageSize)

if __name__ == "__main__":
    stereo_match()