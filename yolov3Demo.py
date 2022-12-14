import math
import cv2
from mxnet import image
from src import util
from src.body import Body
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture('data/mikeSquat.MOV')

def transform_image(array):
    norm_image, image = data.transforms.presets.yolo.transform_test(array)
    return norm_image, image

def detect(network, data):
    pred = network(data)
    class_ids, scores, bounding_boxes=pred
    return class_ids, scores, bounding_boxes


def count_object(network, class_ids, scores, bounding_boxes, object_label, threshold=0.75):
    idx = 0
    for i in range(len(network.classes)):
        if network.classes[i] == object_label:
            idx = i
    scores = scores[0]
    class_ids = class_ids[0]
    num_people = 0
    bboxes = []
    for i in range(len(scores)):
        proba = scores[i].astype('float32').asscalar()
        if proba > threshold and class_ids[i].asscalar() == idx:
            num_people += 1
            bboxes.append(bounding_boxes[0].asnumpy()[i])
    return num_people, bboxes

network = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
img = image.imread('images/mansquat3.png')
norm_image, img = transform_image(img)

class_ids, scores, bounding_boxes = detect(network, norm_image)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)
num_people, person_boxes = count_object(network, class_ids, scores, bounding_boxes, "person", threshold=0.75)
print("people found: ",num_people)
print("person boxes: ", person_boxes)
# Bounding box format: xmin, ymin, xmax, ymax
cy, cx, _ = img.shape
print(cx, cy)
cx /= 2
cy /= 2
print(cx, cy)
mindist = math.inf
for b in person_boxes:
    xmin, ymin, xmax, ymax = [int(x) for x in b]
    print([xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2], [cx, cy])
    curdist = math.dist([xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2], [cx, cy])
    print(curdist)
    if curdist < mindist:
        mindist = curdist
        print("height: ", abs(ymax - ymin))


fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()

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




