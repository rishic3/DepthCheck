import cv2
import matplotlib.pyplot as plt
from src.body import Body
import mediapipe as mp
import ssl
import copy
import numpy as np
from src import util
import time



# Launch framework.
start_time = time.time()
print("DepthPerception has been launched.")

# Initialize mediapipe dependencies.
ssl._create_default_https_context = ssl._create_unverified_context
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Depth plane calculation helper functions.
def calculatePlane(pts):
    p1, p2, p3 = pts
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return [a, b, c, d]

def plotPlane(pts, ax):
    p0, p1, p2 = pts
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
    vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]
    u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
    point = np.array(p0)
    normal = np.array(u_cross_v)
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(-1), range(1))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, z)
    plt.show()


cap = cv2.VideoCapture('data/rcmedlowangle2.mov')


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
        BG_COLOR = (192, 192, 192)  # gray
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5) as pose:

            image = frame
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue
            annotated_image = image.copy()
            # Draw pose landmarks on the image.
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            '''
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            '''
            rKneeCoords.append(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE])
            lKneeCoords.append(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE])
            rHipCoords.append(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP])
            lHipCoords.append(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
            arrFrames.append(annotated_image)
            # Plot pose world landmarks.
            '''
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            '''
            out.write(annotated_image)
        try:
            saving_frames_durations.pop(0)
        except IndexError:
            pass
    count += 1

out.release()



    rightfoot = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    rightheel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    leftfoot = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    leftheel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]





    critpts = [rightfoot, rightheel, leftfoot, leftheel, rightknee, leftknee, righthip, lefthip]
    xs = []
    ys = []
    zs = []
    for critpt in critpts:
        xs.append(critpt.x)
        ys.append(critpt.y)
        zs.append(critpt.z)

    rfootcoords = [rightfoot.x, rightfoot.y, rightfoot.z]
    rheelcoords = [rightheel.x, rightheel.y, rightheel.z]
    lfootcoords = [leftfoot.x, leftfoot.y, leftfoot.z]

    feetpts = [rfootcoords, rheelcoords, lfootcoords]

    print(feetpts)



    plt3d = plt.figure()
    ax = plt3d.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    ax.plot([rightfoot.x, rightheel.x], [rightfoot.y, rightheel.y], [rightfoot.z, rightheel.z], color='black')
    ax.plot([leftfoot.x, leftheel.x], [leftfoot.y, leftheel.y], [leftfoot.z, leftheel.z], color='black')
    ax.plot([righthip.x, rightknee.x], [righthip.y, rightknee.y], [righthip.z, rightknee.z], color='blue')
    ax.plot([lefthip.x, leftknee.x], [lefthip.y, leftknee.y], [lefthip.z, leftknee.z], color='blue')
    plt.show()

#TODO: end blazepose


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



