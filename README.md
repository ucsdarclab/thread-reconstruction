# Thread Reconstruction and Grasping

This repo includes the implementation of **"Autonomous Image-to-Grasp Robotic Suturing Using Reliability-Driven Suture Thread Reconstruction"** ([paper](https://ieeexplore.ieee.org/abstract/document/10900411), [video](https://youtu.be/dDUOSXU4Q8g?si=VxuFiPY6sbPBtZcd))

> Please see the "legacy" branch for the implementation of **"Suture Thread Spline Reconstruction from Endoscopic Images for Robotic Surgery with Reliability-driven Keypoint Detection"** ([paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10161539))

## Usage

```
segmenter = SAMSegmenter(device, model_type) # See segmenter.py
reconstr, reliability = fit_eval(left_img, right_img, calibration_filename, segmenter) # See fit_eval.py

# reconstr is a scipy bspline object: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
# reliability is a scipy 1d interpolation object: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
some_point, some_point_rel = reconstr(0.5), reliability(0.5) # can choose any float value from 0 (start of spline) to 1 (end of spline)
sampled_points, sampled_points_rel = reconstr(np.linspace(0, 1, 500)), reliability(np.linspace(0, 1, 500)) # Can sample multiple points at a time
```

To see grasping implementation, reference `robust_grasp()` in `thread_reconstr_node.py`

## Important files
- `fit_eval.py`: main function for reconstruction
- `segmenter.py`: implements segmentation logic
- `keypoint_selection.py, keypoint_ordering.py, optim.py, reparam.py`: implements reconstruction method
- `thread_reconstr_node.py`: Our ROS implementation, including our robust grasp policy