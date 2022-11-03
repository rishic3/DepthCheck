import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/mikeSquat.MOV')


img_1 = cv2.imread('images/mTest.jpeg')
candidate, subset = body_estimation(img_1)
print(candidate)
print(subset)
index = int(subset[0][8])
rHipCoords = []
print(index)
print(candidate[index].tolist())
print(rHipCoords)

canvas = util.draw_bodypose(img_1, candidate, subset)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.show()






