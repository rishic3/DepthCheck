## DepthPerception

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
Input **180** for the model to compute depth discrepancies in real-world metrics, or type **none**.  

### Output

The model will produce two plots and an output video.  
The first plot will display the results of the Yolov3 detection, like so:  
![alt text](https://github.com/rishic3/DepthCheck/images/detectionExample.png)

### Files

helperFunctions.py contains helper functions used by both primary frameworks.  
blazePoseDemo.py contains a demo implementation for the BlazePose estimation model.  
yolov3Demo.py contains a demo implementation for the Yolov3 object detection model.  

More video samples can be found in the **data** directory. 
Image samples used to test the BlazePose and Yolov3 demos are stored in the **images** directory.  

### References

pytorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). 
