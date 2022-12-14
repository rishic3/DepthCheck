import math
import numpy as np
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

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

def computeDistance(pt, plane):
    a, b, c, d = plane
    x, y, z = pt
    d = abs((a * x + b * y + c * z + d))
    e = (math.sqrt(a * a + b * b + c * c))
    return d / e