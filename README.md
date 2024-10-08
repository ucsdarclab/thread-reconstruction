# Thread reconstruction

## Usage in other code

```
segmenter = SAMSegmenter(device, model_type) # See segmenter.py
reconstr = fit_eval(left_img, right_img, calibration_filename, segmenter) # See fit_eval.py

# reconstr is a scipy bspline object: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
some_point = reconstr(0.5) # can choose any float value from 0 (start of spline) to 1 (end of spline)
sampled_points = reconstr(np.linspace(0, 1, 500)) # Can sample multiple points at a time
```

## Important files
- `fit_eval.py`: main function for reconstruction
- `segmenter.py`: implements segmentation logic
- `keypoint_selection.py, keypoint_ordering.py, optim.py, reparam.py`: implements reconstruction method

Most other files are related to running this in ROS, which includes thread grasping functionality