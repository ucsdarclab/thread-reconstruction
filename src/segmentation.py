import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as manimation

def segmentation(img):
    # Convolve with gaussian second derivative
    pass

if __name__ == "__main__":
    folder_num = 1
    file_num = 0
    file1 = "../Suture_Thread_06_16/thread_%d/left_recif_%d.jpg" % (folder_num, file_num)
    file2 = "../Suture_Thread_06_16/thread_%d/right_recif_%d.jpg" % (folder_num, file_num)
    calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
    img1_z = cv2.imread(file1)
    img1_z = cv2.cvtColor(img1_z, cv2.COLOR_BGR2RGB)
    img2_z = cv2.imread(file2)
    img2_z = cv2.cvtColor(img2_z, cv2.COLOR_BGR2RGB)

    print("creating video -- this can take a few minutes")
    FFMpegWriter = manimation.writers["ffmpeg"]
    fps = 10
    metadata = dict(title="Frame Change", artist="NeelayJ", comment="Result of subtracting consecutive frames")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.figure(fig)
    with writer.saving(fig, "change1.mp4", dpi=200):
        for file_num in range(1259):
            file1 = "../Suture_Thread_06_16/thread_%d/left_recif_%d.jpg" % (folder_num, file_num)
            file2 = "../Suture_Thread_06_16/thread_%d/right_recif_%d.jpg" % (folder_num, file_num)
            calib = "/Users/neelay/ARClabXtra/Suture_Thread_06_16/camera_calibration_sarah.yaml"
            img1 = cv2.imread(file1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread(file2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            img1 = np.float32(img1)
            img2 = np.float32(img2)

            img1_d = img1 - img1_z
            img2_d = img2 - img2_z

            img1_d = np.abs(np.int32(img1_d))
            img2_d = np.abs(np.int32(img2_d))

            plt.clf()
            plt.imshow(img1_d)
            plt.title("Frame %d" %(file_num,))
            # plt.show()
            writer.grab_frame()

            img1_z = img1
            img2_z = img2
    plt.close(fig)