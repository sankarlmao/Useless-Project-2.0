import cv2
import dlib
import os
import time
from scipy.spatial import distance as dist
import mediapipe as mp

SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
(mStart, mEnd) = (48, 68)
YAWN_THRESH = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

yawn_detected = False
hand_detected = False

# Track last action time
last_yawn_time = 0
last_hand_time = 0
DELAY = 3  # seconds delay between actions

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def pause_youtube():
    os.system("xdotool key space")

def switch_tab():
    os.system("xdotool key ctrl+Tab")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    hand_detected = False
    current_time = time.time()

    # Yawn detection with cooldown
    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if mar > YAWN_THRESH:
            if not yawn_detected and (current_time - last_yawn_time > DELAY):
                yawn_detected = True
                last_yawn_time = current_time
                print("Yawn detected! Pausing YouTube...")
                pause_youtube()
        else:
            yawn_detected = False

    # Hand detection with cooldown
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if hand_detected and (current_time - last_hand_time > DELAY):
        last_hand_time = current_time
        print("Hand detected! Switching tab...")
        switch_tab()

    cv2.imshow("Yawn & Hand Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
