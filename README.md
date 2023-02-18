## DepthPerception

### The Framework:

DepthPerception is a deep learning framework to assess squat depth in videos. It aims to automate the judging process in competitive powerlifting settings, and encourage proper depth and posture for non-competitive lifters.

The framework takes an input squat video. It applies Yolov3 to identify the lifter in the video, and employs MediaPipe Blazepose to apply a pose estimation across frames. It then extracts the estimated 3D coordinates to determine the knee and hip planes, computing a depth classification. A more detailed description of the 

#### Install Requirements

Install models and other dependencies with pip

    pip install -r requirements.txt

#### Run the Demo

A demo video can be found in the global directory and is named squatExample.mov. 

sideAngle.py and main.py can be run on this file like so:

    python3 sideAngle.py squatExample.mov

or

    python3 main.py squatExample.mov

The user will be prompted to optionally input a height for the subject in centimeters.  
Input **180** (the height of the subject in the example video) for the model to compute depth discrepancies in real-world metrics, or type **none**.  

#### Output

The frameworks will print a depth classification, as well as the displacement from depth, to console.  

The frameworks will also produce two plots and an output video.  
The first plot will display the results of the Yolov3 detection, like so:  
<br />
<img src="https://user-images.githubusercontent.com/77904151/207735737-3111af0e-eb74-47e6-8fc8-1a0b0e71f9ee.png" width="300">  
<br />
The second plot will display the frame containing the deepest instance of the squat, like so:  
<br />
<img src="https://user-images.githubusercontent.com/77904151/207735719-7e3597f8-161e-42a9-99f9-5d23fd51eefc.png" width="300">  
<br />
If sideAngle.py is being used, the plot will also contain a line representing the depth threshold:  
<br />
<img src="https://user-images.githubusercontent.com/77904151/207735695-881ac193-2f4c-46c4-a315-19e9518c9eeb.png" width="300">  
<br />

Both frameworks will also produce a file **output.mp4** containing the parsed frames, with pose annotations, combined into an output video:  
<br />
<img src="https://user-images.githubusercontent.com/77904151/207737334-5ba1014b-1f95-485e-a6bc-8f794022f58b.gif" width="300">  

### Files

helperFunctions.py contains helper functions used by both primary frameworks.  
blazePoseDemo.py contains a demo implementation for the BlazePose estimation model.  
yolov3Demo.py contains a demo implementation for the Yolov3 object detection model.  

More video samples can be found in the **data** directory. 
Image samples used to test the BlazePose and Yolov3 demos are stored in the **images** directory.  

### References

Pyorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the Pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch).

Google's MediaPipe [pose estimation](https://google.github.io/mediapipe/solutions/pose.html) was imported and used according to their [documentation](https://google.github.io/mediapipe/solutions/pose.html).

Yolov3 object detection was imported from [gluoncv](https://cv.gluon.ai/contents.html). The source code can be found [here](https://cv.gluon.ai/_modules/gluoncv/model_zoo/yolo/yolo3.html).
