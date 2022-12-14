import math
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import util
from src.body import Body
from gluoncv import model_zoo, utils
import mxnet as mx
import time
from helperFunctions import transform_image, detect, count_object

# Launch framework.
start_time = time.time()
print("DepthPerception has been launched.")

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/rcmedlowangle2.mov')

# Optionally receive an input height for distance and velocity measurements.
checkHeight = 0
height = input("Enter a height in cm, or type '"'none'"': ")
if height != "none":
    checkHeight = 1
    height = float(height)

# Returns the list of durations where to save the frames.
def get_saving_frames_durations(cap, saving_fps):
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

# Set desired FPS at which to parse video.
fps = cap.get(cv2.CAP_PROP_FPS)
savingFPS = 5
saving_frames_durations = get_saving_frames_durations(cap, savingFPS)

# Output path for video with pose estimations.
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), savingFPS, (int(cap.get(3)), int(cap.get(4))))

# Variables to store information across frames.
count = 0
pixelHeight = -1
firstFrame = 1
detectSizeY = -1
frameSizeY = -1
rHipCoords = []
lHipCoords = []
rKneeCoords = []
lKneeCoords = []
arrFrames = []

while True:
    is_read, frame = cap.read()
    if not is_read:
        break
    frameSize = frame.shape
    frameSizeY = frameSize[1]
    frame_duration = count / fps

    # UNCOMMENT FOR .mov videos: .mov video types must be flipped prior to processing.
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    try:
        closest_duration = saving_frames_durations[0]
    except IndexError:
        break
    if frame_duration >= closest_duration:

        # Compute pose estimation.
        canvas = copy.deepcopy(frame)
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Store keypoints of interest.
        try:
            index = int(subset[0][8])
        except IndexError:
            count += 1
            continue
        if index != -1:
            rHipCoords.append(candidate[index].tolist())
        index = int(subset[0][11])
        if index != -1:
            lHipCoords.append(candidate[index].tolist())
        index = int(subset[0][9])
        if index != -1:
            rKneeCoords.append(candidate[index].tolist())
        index = int(subset[0][12])
        if index != -1:
            lKneeCoords.append(candidate[index].tolist())

        # Estimate height of subject with bounding box detection.
        if firstFrame and checkHeight:
            network = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
            norm_image, img = transform_image(mx.nd.array(frame))
            xdim, ydim, _ = img.shape
            detectSizeY = ydim
            cx = xdim / 2
            cy = ydim / 2
            class_ids, scores, bounding_boxes = detect(network, norm_image)

            # Plot detected bounding boxes:

            ax = utils.viz.plot_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)
            fig = plt.gcf()
            fig.set_size_inches(14, 14)
            plt.title('Detected subject(s):')
            plt.show()


            thresh = 0.7
            num_people, person_boxes = count_object(network, class_ids, scores, bounding_boxes, "person", threshold=thresh)
            if num_people == 0:
                print("0 subjects detected.")
            if num_people == 1:
                print("1 subject detected with confidence threshold ", thresh, ". Analyzing subject's squat.")
                for b in person_boxes:
                    xmin, ymin, xmax, ymax = [int(x) for x in b]
                    pixelHeight = (ymax - ymin)

            # If multiple subjects are detected, analyze the center most subject.
            if num_people > 1:
                print(num_people, " subjects detected with confidence threshold ", thresh, ". Analyzing the most central subject.")
                mindist = math.inf
                for b in person_boxes:
                    xmin, ymin, xmax, ymax = [int(x) for x in b]
                    curdist = math.dist([xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2], [cx, cy])
                    if curdist < mindist:
                        mindist = curdist
                        pixelHeight = (ymax-ymin)

            firstFrame = 0

        arrFrames.append(canvas)
        out.write(canvas)
        try:
            saving_frames_durations.pop(0)
        except IndexError:
            pass
    count += 1

out.release()

# Identification and analysis of deepest frame.
leftOnly = True

if not (len(rHipCoords) == len(lHipCoords) == len(rKneeCoords) == len(lKneeCoords)):
    if len(rHipCoords) == len(rKneeCoords):
        leftOnly = False
    elif len(lHipCoords) == len(lKneeCoords):
        leftOnly = True
    else:
        print("Inconsistent number of joints detected across frames")
        exit(1)

rHipYs = [coords[1] for coords in rHipCoords]
lHipYs = [coords[1] for coords in lHipCoords]

# Identify frame with lowest detected hip joint.
minRHip = max(rHipYs)
minRIndex = rHipYs.index(minRHip)
minLHip = max(lHipYs)
minLIndex = lHipYs.index(minLHip)

keyframe = minRIndex
rKneeYs = [coords[1] for coords in rKneeCoords]
lKneeYs = [coords[1] for coords in lKneeCoords]
minRKnee = rKneeYs[keyframe]
minLKnee = lKneeYs[keyframe]

# Compute cm per pixel for distance measurements.
cmpp = -1
if height != "none":
    pixelHeight *= (frameSizeY / detectSizeY)
    cmpp = height / pixelHeight

# Print classification output.
if leftOnly:
    if minLKnee < minLHip:
        if pixelHeight != -1:
            print("Classification: DEPTH! Passed depth by", round((minLHip - minLKnee) * cmpp, 3), "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            print("Classification: NOT DEPTH! Missed depth by", round((minLKnee - minLHip) * cmpp, 3), "cm")
        else:
            print("Classification: NOT DEPTH!")
else:
    if minRKnee < minRHip:
        if pixelHeight != -1:
            print("Classification: DEPTH! Passed depth by", round((minRHip - minRKnee) * cmpp, 3), "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            print("Classification: NOT DEPTH! Missed depth by", round((minRKnee - minRHip) * cmpp, 3), "cm")
        else:
            print("Classification: NOT DEPTH!")

print("\n")
print("--- %s seconds ---" % (time.time() - start_time))

# Plot deepest frame.
keyCanvas = arrFrames[keyframe]
plt.title('Deepest Frame')
plt.axline((0, minRKnee), (keyCanvas.shape[1], minRKnee))
plt.imshow(keyCanvas[:, :, [2, 1, 0]])
plt.show()



