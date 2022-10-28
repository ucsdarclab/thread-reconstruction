# Thread Reconstruction Work Log
## Useful resources
- "Real-Time Visual Tracking of Dynamic Surgical Suture Threads": https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8008771
- "Toward Image-Guided Automated Suture Grasping Under Complex Environments: A Learning-Enabled and Optimization-Based Holistic Framework": https://ieeexplore.ieee.org/document/9664632
- "Event-based Vision: A Survey": https://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf

# Week of 10/24
The focus of this week is to build a manually labeled thread image dataset and further develop my proposed segmentation method. The 2 priors I plan on leveraging are surgical tool motion and the static nature of the surgical background

- By finding the difference between adjacent images in a surgical video, the following video is obtained. Notice how the thread stands out while the mostly static background is almost completely black

  https://user-images.githubusercontent.com/73408212/198708864-3a8cf477-1f2c-4392-8d2d-a33e6c117f13.mp4
  
  This operation is somewhat similar to event-based vision. I will be looking through [this paper](https://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf) over the next few weeks to see if any useful mathematical formulations from event-based learning can be applied to this method.
- Much of this week was spent doing manual segmentation. I have likely segmented 40-50 surgical image pairs so far.


# Week of 10/10
Last week, I developed a possible framework for segmentation from multiframe data. This can be broken into 2 main parts: segmentation prediction and bayesian filtering. This week, I will work on the prediction part.

- I looked through the data that Fei sent to me. The most useful data for prediction seems to be surgical tool location, as, when grasped, the thread moves predictably with the tool. However, the tools aren't always grasping the thread, so their movements may sometimes have no impact on the thread movement. I'm thinking that tool movement can be used to help define a region of interest, within which some (possible deep learning) segmentation algorithm can run.
