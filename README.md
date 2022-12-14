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
<img src="https://user-images.githubusercontent.com/77904151/207735737-3111af0e-eb74-47e6-8fc8-1a0b0e71f9ee.png" width="300">  
<br />
The second plot will display the frame containing the deepest instance of the squat, like so:  
<img src="https://user-images.githubusercontent.com/77904151/207735719-7e3597f8-161e-42a9-99f9-5d23fd51eefc.png" width="300">  
<br />
If sideAngle.py is being used, the plot will also contain a line representing the depth threshold:  
<img src="https://user-images.githubusercontent.com/77904151/207735695-881ac193-2f4c-46c4-a315-19e9518c9eeb.png" width="300">  
<br />

Both frameworks will also produce a file **output.mp4** containing the parsed frames, with pose annotations, combined into an output video:  


### Files

helperFunctions.py contains helper functions used by both primary frameworks.  
blazePoseDemo.py contains a demo implementation for the BlazePose estimation model.  
yolov3Demo.py contains a demo implementation for the Yolov3 object detection model.  

More video samples can be found in the **data** directory. 
Image samples used to test the BlazePose and Yolov3 demos are stored in the **images** directory.  

### References

pytorch implementation of [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) including **Body and Hand Pose Estimation**, and the pytorch model is directly converted from [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) caffemodel by [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). 
