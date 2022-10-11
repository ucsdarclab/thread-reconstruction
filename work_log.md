# Thread Reconstruction Work Log
## Useful resources
- "Real-Time Visual Tracking of Dynamic Surgical Suture Threads": https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8008771
- "Toward Image-Guided Automated Suture Grasping Under Complex Environments: A Learning-Enabled and Optimization-Based Holistic Framework": https://ieeexplore.ieee.org/document/9664632

# Week of 10/10
Last week, I developed a possible framework for segmentation from multiframe data. This can be broken into 2 main parts: segmentation prediction and bayesian filtering. This week, I will work on the prediction part.

- I looked through the data that Fei sent to me. The most useful data for prediction seems to be surgical tool location, as, when grasped, the thread moves predictably with the tool. However, the tools aren't always grasping the thread, so their movements may sometimes have no impact on the thread movement. I'm thinking that tool movement can be used to help define a region of interest, within which some (possible deep learning) segmentation algorithm can run.
