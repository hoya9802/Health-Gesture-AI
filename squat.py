import cv2
import numpy as np
import mediapipe as mp
from math import acos, degrees

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_pose = mp.solutions.mediapipe.python.solutions.pose

# cap = cv2.VideoCapture('C:/Users/hoya9/Desktop/HSAI/data/ET_Squat.mp4')
cap = cv2.VideoCapture(0)

up = False
down = False
count = 0

with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = cap.read()
        if ret == False:
            print('frame error!!')
            break
        # frame = cv2.resize(frame, dsize=(0, 0), fx=0.4, fy=0.35, interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[24].x * width)
            y1 = int(results.pose_landmarks.landmark[24].y * height)

            x2 = int(results.pose_landmarks.landmark[26].x * width)
            y2 = int(results.pose_landmarks.landmark[26].y * height)

            x3 = int(results.pose_landmarks.landmark[28].x * width)
            y3 = int(results.pose_landmarks.landmark[28].y * height)

            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.array([x3, y3])

            l1 = np.linalg.norm(p2 - p3)
            l2 = np.linalg.norm(p1 - p3)
            l3 = np.linalg.norm(p1 - p2)

            angle = degrees(acos((pow(l1, 2) + pow(l3, 2) - pow(l2, 2)) / (2 * l1 * l3)))
            if angle >= 160:
                up = True
            if up == True and down == False and angle <= 70:
                down = True
            if up == True and down == True and angle >= 160:
                count += 1
                up = False
                down = False

            point_image = np.zeros(frame.shape, np.uint8)
            cv2.line(point_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
            cv2.line(point_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
            cv2.line(point_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(point_image, pts=[contours], color=(255, 255, 255))

            output = cv2.addWeighted(frame, 1, point_image, 0.8, 0)

            cv2.circle(frame, (x1, y1), 3, (255, 0, 0), 3)
            cv2.circle(frame, (x2, y2), 3, (255, 0, 0), 3)
            cv2.circle(frame, (x3, y3), 3, (255, 0, 0), 3)
            cv2.rectangle(output, (0, 0), (60, 60), (255, 255, 0), -1)
            cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
            cv2.putText(output, str(count), (10, 50), 1, 3.5, (128, 0, 250), 2)
            cv2.imshow('output', output)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
            break

cap.release()
cv2.destroyAllWindows()
        