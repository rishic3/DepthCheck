import cv2
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import ssl
import numpy as np
import time
import mxnet as mx
from gluoncv import model_zoo, utils
from helperFunctions import calculatePlane, computeDistance, transform_image, detect, count_object

# Launch framework.
start_time = time.time()
print("DepthPerception has been launched.")

# Initialize mediapipe dependencies.
ssl._create_default_https_context = ssl._create_unverified_context
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
BG_COLOR = (192, 192, 192)

cap = cv2.VideoCapture('data/rclowangle2.mov')

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
resultWorldCoords = []
resultPoseCoords = []

rHipCoords = []
lHipCoords = []
rKneeCoords = []
lKneeCoords = []
arrFrames = []

while True:
    is_read, frame = cap.read()
    if not is_read:
        break
    _, frameSizeY, _ = frame.shape
    frame_duration = count / fps

    # UNCOMMENT FOR .mov files: .mov video types must be flipped prior to processing.
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    try:
        closest_duration = saving_frames_durations[0]
    except IndexError:
        break
    if frame_duration >= closest_duration:

        # Compute pose estimation.
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:
            image_height, image_width, _ = frame.shape
            #results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = pose.process(frame)
            if not results.pose_landmarks:
                continue
            annotated_image = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            resultPoseCoords.append(results.pose_landmarks)
            resultWorldCoords.append(results.pose_world_landmarks)

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
            '''
            ax = utils.viz.plot_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)
            fig = plt.gcf()
            fig.set_size_inches(14, 14)
            plt.title('Detected subject(s):')
            plt.show()
            '''

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

        arrFrames.append(annotated_image)
        out.write(annotated_image)
        try:
            saving_frames_durations.pop(0)
        except IndexError:
            pass
    count += 1

out.release()

leftOnly = False
rightOnly = False
useImg = False

# Identification and analysis of deepest frame.
minRHip = resultPoseCoords[0].landmark[mp_pose.PoseLandmark.RIGHT_HIP]
minRKnee = resultPoseCoords[0].landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
minRHipIndex = -1
minLHip = resultPoseCoords[0].landmark[mp_pose.PoseLandmark.LEFT_HIP]
minLKnee = resultPoseCoords[0].landmark[mp_pose.PoseLandmark.LEFT_KNEE]
minLHipIndex = -1

# Identify frame with lowest detected hip joint.
for i, res in enumerate(resultPoseCoords):
    if res.landmark[mp_pose.PoseLandmark.LEFT_HIP].y > minLHip.y:
        minLHip = res.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        minLKnee = res.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        minLHipIndex = i
    if res.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y > minRHip.y:
        minRHip = res.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        minRKnee = res.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        minRHipIndex = i

# If deepest hip indices do not align, only analyze the hip joint with better visibility.
if minRHipIndex != minLHipIndex:
    if minRHip.visibility > minLHip.visibility:
        rightOnly = True
    else:
        leftOnly = True
print("Deepest frame detected at ", minLHipIndex, ". Analyzing.")

assert len(resultPoseCoords) == len(resultWorldCoords), f"Pose coords do not align with world coords."

# Extract key landmarks from world coordinates of deepest frame.
worldResults = resultWorldCoords[minLHipIndex]
rightfoot = worldResults.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
rightheel = worldResults.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
leftfoot = worldResults.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
leftheel = worldResults.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
rightknee = worldResults.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
leftknee = worldResults.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
righthip = worldResults.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
lefthip = worldResults.landmark[mp_pose.PoseLandmark.LEFT_HIP]

# If feet are occluded, classify depth strictly based on image coordinates.
if rightfoot.visibility < 0.75 or leftfoot.visibility < 0.75:
    useImg = True

# Compute base plane using feet landmarks.
rfootcoords = [rightfoot.x, rightfoot.y, rightfoot.z]
rheelcoords = [rightheel.x, rightheel.y, rightheel.z]
lfootcoords = [leftfoot.x, leftfoot.y, leftfoot.z]
lheelcoords = [leftheel.x, leftheel.y, leftheel.z]
feetpts = [rfootcoords, rheelcoords, lfootcoords]

basePlane = calculatePlane([rfootcoords, rheelcoords, lfootcoords]) if leftheel.visibility > rightheel.visibility else calculatePlane([rfootcoords, lfootcoords, lheelcoords])
rHiptoBase = computeDistance([righthip.x, righthip.y, righthip.z], basePlane)
lHiptoBase = computeDistance([lefthip.x, lefthip.y, lefthip.z], basePlane)
rKneetoBase = computeDistance([rightknee.x, rightknee.y, rightknee.z], basePlane)
lKneetoBase = computeDistance([leftknee.x, leftknee.y, leftknee.z], basePlane)

print("hip distance to base: ", rHiptoBase, lHiptoBase, ", knee distance to base: ", rKneetoBase, lKneetoBase)

# Compute cm per pixel for distance measurements.
cmpp = -1
if height != "none":
    pixelHeight *= (frameSizeY / detectSizeY)
    cmpp = height / pixelHeight

# Mediapipe outputs normalized image coordinates; recover the original coordinates.
unnormedLHip = minLHip.y * frameSizeY
unnormedRHip = minRHip.y * frameSizeY
unnormedLKnee = minLKnee.y * frameSizeY
unnormedRKnee = minRKnee.y * frameSizeY


# Determine depth classification based on distances of hip and knee to base plane.
if leftOnly:
    if lHiptoBase < lKneetoBase:
        if pixelHeight != -1:
            print("Classification: DEPTH! Passed depth by", (unnormedLHip - unnormedLKnee) * cmpp, "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            print("Classification: NOT DEPTH! Missed depth by", (unnormedLKnee - unnormedLHip) * cmpp, "cm")
        else:
            print("Classification: NOT DEPTH!")
elif rightOnly:
    if rHiptoBase < rKneetoBase:
        if pixelHeight != -1:
            print("Classification: DEPTH! Passed depth by", (unnormedRHip - unnormedRKnee) * cmpp, "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            print("Classification: NOT DEPTH! Missed depth by", (unnormedRKnee - unnormedRHip) * cmpp, "cm")
        else:
            print("Classification: NOT DEPTH!")
else:
    if lHiptoBase < lKneetoBase and rHiptoBase < rKneetoBase:
        if pixelHeight != -1:
            avgDist = (unnormedLHip - unnormedLKnee + unnormedRHip - unnormedRKnee) / 2
            print("Classification: DEPTH! Passed depth by", avgDist * cmpp, "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            avgDist = (unnormedLHip - unnormedLKnee + unnormedRHip - unnormedRKnee) / 2
            print("Classification: NOT DEPTH! Missed depth by", avgDist * cmpp, "cm")
        else:
            print("Classification: NOT DEPTH!")

if useImg:
    if minLHip.y > minLKnee.y:
        if pixelHeight != -1:
            print("Classification: DEPTH! Passed depth by", (unnormedLHip - unnormedLKnee) * cmpp, "cm.")
        else:
            print("Classification: DEPTH!")
    else:
        if pixelHeight != -1:
            print("Classification: NOT DEPTH! Missed depth by", (unnormedLKnee - unnormedLHip) * cmpp, "cm")
        else:
            print("Classification: NOT DEPTH!")

print("\n")
print("--- %s seconds ---" % (time.time() - start_time))

keyCanvas = arrFrames[minLHipIndex]
plt.title('Deepest Frame')
plt.imshow(keyCanvas[:, :, [2, 1, 0]])
plt.show()


