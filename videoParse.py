import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand
import time
start_time = time.time()

print("DepthPerception has been launched.")

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/rcmedlowangle2.mov')

checkHeight = 0
height = input("Enter a height in cm, or type '"'none'"': ")
if height != "none":
    checkHeight = 1
    height = float(height)

def get_saving_frames_durations(cap, saving_fps):
    # returns the list of durations where to save the frames
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

fps = cap.get(cv2.CAP_PROP_FPS)

#SET DESIRED FPS
savingFPS = 5
saving_frames_durations = get_saving_frames_durations(cap, savingFPS)

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), savingFPS, (int(cap.get(3)), int(cap.get(4))))
count = 0
pixelHeight = 0

rHipCoords = []  #8
lHipCoords = []  #11
rKneeCoords = []  #9
lKneeCoords = []  #12

arrFrames = []

while True:
    is_read, frame = cap.read()
    if not is_read:
        break
    frameSize = frame.shape
    frame_duration = count / fps

    #frame = cv2.rotate(frame, cv2.ROTATE_180)

    try:
        closest_duration = saving_frames_durations[0]
    except IndexError:
        # list is empty, all duration frames were saved
        break
    if frame_duration >= closest_duration:
        canvas = copy.deepcopy(frame)
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

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

        if checkHeight:
            miny = min(x[1] for x in candidate)
            maxy = max(x[1] for x in candidate)
            pixelHeight = abs(miny - maxy)
            checkHeight = 0

        arrFrames.append(canvas)
        out.write(canvas)
        try:
            saving_frames_durations.pop(0)
        except IndexError:
            pass
    count += 1

out.release()

'''
print("Right Hip: Numfound ", len(rHipCoords))
print("Left Hip: Numfound: ", len(lHipCoords))
print("Right Knee: Numfound: ", len(rKneeCoords))
print("Left Knee: Numfound: ", len(lKneeCoords))
'''

rightOnly = False
leftOnly = False

if not (len(rHipCoords) == len(lHipCoords) == len(rKneeCoords) == len(lKneeCoords)):
    if len(rHipCoords) == len(rKneeCoords):
        rightOnly = True
    elif len(lHipCoords) == len(lKneeCoords):
        leftOnly = True
    else:
        print("barfed: inconsistent number of joints detected across frames")
        exit(1)

#TODO: code leftonly and rightonly cases

rHipYs = [coords[1] for coords in rHipCoords]
lHipYs = [coords[1] for coords in lHipCoords]

print(rHipYs)
print(lHipYs)

# origin of image is top left; lowest depth is max
minRHip = max(rHipYs)
minRIndex = rHipYs.index(minRHip)
minLHip = max(lHipYs)
minLIndex = lHipYs.index(minLHip)

'''
print("Right Hip Y coords: ", rHipYs)
print("Left Hip Y coords: ", lHipYs)
print("Lowest right hip coord: ", minRHip, " at frame ", minRIndex)
print("Lowest left hip coord: ", minLHip, " at frame ", minLIndex)
'''

if not (minRIndex == minLIndex):
    rightOnly = True

keyframe = minRIndex
rKneeYs = [coords[1] for coords in rKneeCoords]
lKneeYs = [coords[1] for coords in lKneeCoords]

minRKnee = rKneeYs[keyframe]
minLKnee = lKneeYs[keyframe]

print("Lowest Right Hip: ", minRHip, ", lowest right knee: ", minRKnee)
print("Lowest Left Hip: ", minLHip, ", lowest left knee: ", minLKnee)

#cm per pixel:
cmpp = 1
if height != "none":
    cmpp = height / pixelHeight

# Y coordinate 0 is TOP left of page --> higher number == physically lower
if leftOnly:
    if minLKnee < minLHip:
        print("DEPTH! passed depth by", (minLHip - minLKnee) * cmpp, "cm")
    else:
        print("NO DEPTH! missed depth by", (minLKnee - minLHip) * cmpp, "cm")
else:
    if minRKnee < minRHip:
        print("DEPTH! passed depth by", (minRHip-minRKnee) * cmpp, "cm")
    else:
        print("NO DEPTH! missed depth by", (minRKnee - minRHip) * cmpp, "cm")

print("\n")
print("--- %s seconds ---" % (time.time() - start_time))

keyCanvas = arrFrames[keyframe]
plt.title('Deepest Frame')
plt.axline((0, minRKnee), (keyCanvas.shape[1], minRKnee))
plt.imshow(keyCanvas[:, :, [2, 1, 0]])
plt.show()



