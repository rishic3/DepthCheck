import cv2
import mediapipe as mp
import numpy as np
import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ["images/rishimedlow2.png"]
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
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

    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('images/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    rightfoot = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    rightheel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    leftfoot = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    leftheel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]

    rightknee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    leftknee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    righthip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    lefthip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

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

    plt3d = plt.figure()
    ax = plt3d.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    ax.plot([rightfoot.x, rightheel.x], [rightfoot.y, rightheel.y], [rightfoot.z, rightheel.z], color='black')
    ax.plot([leftfoot.x, leftheel.x], [leftfoot.y, leftheel.y], [leftfoot.z, leftheel.z], color='black')
    ax.plot([righthip.x, rightknee.x], [righthip.y, rightknee.y], [righthip.z, rightknee.z], color='blue')
    ax.plot([lefthip.x, leftknee.x], [lefthip.y, leftknee.y], [lefthip.z, leftknee.z], color='blue')
    plt.show()

    #calcPlane(feetpts)


    #ax.scatter3D(xdata, ydata, zdata)