## DepthPerception

#### Install Requirements

Install models and other dependencies with pip

    pip install -r requirements.txt

#### Run the Demo

A demo video can be found in the global directory and is named squatExample.mov. 

sideAngle.py and main.py can be run on this file like so:

    python sideAngle.py squatExample.mov

or

    python main.py squatExample.mov

The user will be prompted to optionally input a height for the subject in centimeters.  
Input **180** for the model to compute depth discrepancies in real-world metrics, or type **none**.  

helperFunctions.py contains helper functions used by both primary frameworks.  
blazePoseDemo.py contains a demo implementation for the BlazePose estimation model.  
yolov3Demo.py contains a demo implementation for the Yolov3 object detection model.  

More video samples can be found in the data directory.  

pytorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). 
