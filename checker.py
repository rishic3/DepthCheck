import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from src import util
from src.body import Body

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/mikeSquat.MOV')


img_1 = cv2.imread('images/rishiStanding.png')

print("shape: ", img_1.shape)

candidate, subset = body_estimation(img_1)
'''
print("candidates: ", candidate)
print("subset: ", subset)
index = int(subset[0][8])
print("index: ", index)
print("coords of index: ", candidate[index].tolist())
'''
print("candidates: ", candidate)
miny = min(x[1] for x in candidate)
print("min y: ", miny)
maxy = max(x[1] for x in candidate)
print("max y: ", maxy)

height = abs(miny-maxy)
print(height)

canvas = util.draw_bodypose(img_1, candidate, subset)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.show()






