import cv2
import matplotlib.pyplot as plt
import copy

from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

test_image = 'images/rishiSquat2.png'
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)
canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)


'''
# detect hand
hands_list = util.handDetect(candidate, subset, oriImg)

all_hand_peaks = []
for x, y, w, is_left in hands_list:
    # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # if is_left:
        # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
        # plt.show()
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    # else:
    #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
    #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
    #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    #     print(peaks)
    all_hand_peaks.append(peaks)

canvas = util.draw_handpose(canvas, all_hand_peaks)
'''

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

'''
TODO:

Parsing input:
Given a squat video, identify the frame at which the squat is at its deepest, then input this frame as an image into
the openpose classifier. Identifying the deepest frame can be done directly using motion fields/feature tracking,
or another ML model like a CNN. 

Classifying output:
Try to find the common hyperplane in which one set (right or left) of the detected key points lie; this plane is the true
'side angle' perspective of the image. We can draw a horizontal line in this hyperplane from the knee key point to 
classify whether the squat is depth or not. We can also use this to assess how far off the lifter is from depth
(but this would require some information about the focal length to perform a projection using to map the pixel distance 
to real-world measurement).
'''