# Datasets

## Real Dataset

The 10 thread data points are broken up in to 2 files (2 data points in [thread_1_seg](./real/thread_1_seg/), 8 in [thread_2_seg](./real/thread_2_seg/)), as these images were taken from 2 experiments. Images are stereo-rectified, and the camera calibration file is [camera_calibration.yaml](./real/camera_calibration.yaml).

NOTE: in our paper, these thread data points are ordered as follows

1. thread 99 in thread_1_seg
2. thread 119 in thread_1_seg
3. thread 59 in thread_2_seg
4. thread 72 in thread_2_seg
5. thread 116 in thread_2_seg
6. thread 149 in thread_2_seg
7. thread 159 in thread_2_seg
8. thread 174 in thread_2_seg
9. thread 187 in thread_2_seg
10. thread 209 in thread_2_seg

## Simulated Dataset

The 40 data points are broken up into 10 folders, as each group of 4 images consist of different orientations of the same model. Groups 1-4 have a separate calibration matrix and image storage type as groups 5-10 due to the way the data was generated. All models were generated using NURBS curves in blender, and the ground-truth control points of these curves were recorded using [get_gt.py](./simulated/get_gt.py). These control points are stored as .npy files paired with each data point.