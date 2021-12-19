import cv2
import numpy as np
import mediapipe as mp
import winsound as sd
from math import acos, degrees
from tensorflow.keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

def beepsound():
    fr = 1700    # range : 37 ~ 32767
    du = 75     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image, results

def draw_styled_landmarks(images, results):
    mp_drawing.draw_landmarks(images, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(images, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def hand_extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    output_frame = cv2.flip(output_frame, 1)
    return output_frame

# Actions that we try to detect
actions = np.array(['nothing', 'add_count', 'reset_count'])

# Thirty videos worth of data
no_sequences = 40

# Video are going to be 30 frames in length
sequence_length = 20

model = load_model('C:/Users/hoya9/Desktop/CV/MediaPipe/best-model_RNN.h5')
# model = load_model('C:/Users/hoya9/Desktop/HSAI/model/best-model.h5')

up = False
down = False
goal_count = 0
count = 0
sequence = []
sentence = []
predictions = []
threshold = 0.7

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(static_image_mode=False) as holistic:
    while True:
        ret, frame = cap.read()
        height, width, _ = frame.shape
        
        if ret == False:
            print('frame error!!')
            break

        # frame = cv2.resize(frame, dsize=(0, 0), fx=0.4, fy=0.35, interpolation=cv2.INTER_LINEAR)
        image, results = mediapipe_detection(frame, holistic)
        
        draw_styled_landmarks(image, results)
        
        keypoints = hand_extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-20:]
        
        if len(sequence) == 20:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            if actions[np.argmax(res)] == actions[0]:
                                sentence.append(actions[0])

                            elif actions[np.argmax(res)] == actions[1]:
                                goal_count += 5
                                sentence.append(actions[1])
                            elif actions[np.argmax(res)] == actions[2]:
                                count = 0
                                goal_count = 0
                                sentence.append(actions[2])
                            # else:
                            #     count, goal_count = 0

                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 1:
                sentence = sentence[-1:]
            
            image = cv2.flip(image, 1)
            
            frame = prob_viz(res, actions, image, colors)
        

        if results.pose_landmarks is not None:
            x1 = int(results.pose_landmarks.landmark[23].x * width)
            y1 = int(results.pose_landmarks.landmark[23].y * height)

            x2 = int(results.pose_landmarks.landmark[25].x * width)
            y2 = int(results.pose_landmarks.landmark[25].y * height)

            x3 = int(results.pose_landmarks.landmark[27].x * width)
            y3 = int(results.pose_landmarks.landmark[27].y * height)

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
                beepsound()
                up = False
                down = False

            point_image = np.zeros(frame.shape, np.uint8)
            cv2.line(point_image, (x1, y1), (x2, y2), (255, 255, 0), 20)
            cv2.line(point_image, (x2, y2), (x3, y3), (255, 255, 0), 20)
            cv2.line(point_image, (x1, y1), (x3, y3), (255, 255, 0), 5)
            contours = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(point_image, pts=[contours], color=(255, 255, 255))

            output = cv2.addWeighted(frame, 1, point_image, 0.8, 0)

            output = cv2.flip(output, 1)
            # frame = cv2.flip(frame, 1)

            # cv2.circle(frame, (x1, y1), 3, (255, 0, 0), 3)
            # cv2.circle(frame, (x2, y2), 3, (255, 0, 0), 3)
            # cv2.circle(frame, (x3, y3), 3, (255, 0, 0), 3)

            cv2.rectangle(output, (530, 0), (640, 60), (0, 0, 0), -1)
            cv2.rectangle(output, (530, 60), (640, 120), (255, 255, 0), -1)
            cv2.putText(output, str(int(angle)), (x2 + 30, y2), 1, 1.5, (128, 0, 250), 2)
            cv2.putText(output, str(goal_count), (527, 50), 1, 3.5, (0, 255, 255), 2)
            cv2.putText(output, str(count), (527, 110), 1, 3.5, (128, 0, 250), 2)
            cv2.putText(output, 'Squat', (260, 40), 1, 2, (0, 0, 255), 2)
            
            if goal_count == 0:
                cv2.putText(output, 'Setting your Goal', (170, 200), 1, 2, (0, 0, 255), 2)
            
            if goal_count == count and goal_count >= 10:
                cv2.putText(output, 'Congratulations!!', (170, 200), 1, 2, (0, 0, 255), 2)
                
            image = output

        cv2.rectangle(image, (1,0), (200, 40), (0,0,0), -1)
        cv2.putText(image, ' '.join(sentence), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Main_Frame', image)
        # cv2.imshow("Squat_Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
            break

cap.release()
cv2.destroyAllWindows()