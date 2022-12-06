import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import cv2 as cv
import numpy as np
import requests as req
import os as os
from mxnet import image
from src import util
from src.body import Body
import os
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/mikeSquat.MOV')

def transform_image(array):
    norm_image,image=data.transforms.presets.yolo.transform_test(array)
    return norm_image,image

def detect(network, data):
    pred=network(data)
    class_ids,scores,bounding_boxes=pred
    return class_ids, scores, bounding_boxes



network = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
img = image.imread('images/rishiStanding.png')
norm_image, img = transform_image(img)

class_ids, scores, bounding_boxes = detect(network, norm_image)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()

'''

img_1 = cv2.imread('images/mansquat3.png')
print("shape: ", img_1.shape)
candidate, subset = body_estimation(img_1)

print("candidates: ", candidate)
print("subset: ", subset)
index = int(subset[0][8])
print("index: ", index)
print("coords of index: ", candidate[index].tolist())

print("candidates: ", candidate)
print("subset: ", subset)
miny = min(x[1] for x in candidate)
print("min y: ", miny)
maxy = max(x[1] for x in candidate)
print("max y: ", maxy)

height = abs(miny-maxy)
print(height)

canvas = util.draw_bodypose(img_1, candidate, subset)
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.show()

'''




