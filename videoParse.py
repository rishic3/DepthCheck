import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/skateClip.mp4')

def get_saving_frames_durations(cap, saving_fps):
    # returns the list of durations where to save the frames
    s = []
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

fps = cap.get(cv2.CAP_PROP_FPS)
savingFPS = 10
saving_frames_durations = get_saving_frames_durations(cap, savingFPS)

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), savingFPS, (int(cap.get(3)), int(cap.get(4))))

count = 0
while True:
    is_read, frame = cap.read()
    if not is_read:
        break
    frameSize = frame.shape
    frame_duration = count / fps

    try:
        closest_duration = saving_frames_durations[0]
    except IndexError:
        # list is empty, all duration frames were saved
        break
    if frame_duration >= closest_duration:
        canvas = copy.deepcopy(frame)
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        img = canvas[:, :, [2, 1, 0]]
        out.write(img)
        try:
            saving_frames_durations.pop(0)
        except IndexError:
            pass
    count += 1

out.release()
