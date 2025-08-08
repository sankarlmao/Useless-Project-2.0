import cv2
import dlib
import os
from scipy.spatial import distance as dist
import mediapipe as mp

# Path to the dlib shape predictor file
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Indices for mouth landmarks
(mStart, mEnd) = (48, 68)

# Yawn detection threshold
YAWN_THRESH = 0.6

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

yawn_detected = False
hand_detected = False

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # vertical
    B = dist.euclidean(mouth[4], mouth[8])   # vertical
    C = dist.euclidean(mouth[0], mouth[6])   # horizontal
    return (A + B) / (2.0 * C)

def pause_youtube():
    # Simulate pressing space (works if video tab is active)
    os.system("xdotool key space")

def switch_tab():
    # Simulate pressing ctrl+tab to switch tab
    os.system("xdotool key ctrl+Tab")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # Reset flags each frame
    hand_detected = False

    # Detect faces and check for yawn
    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if mar > YAWN_THRESH and not yawn_detected:
            yawn_detected = True
            print("Yawn detected! Pausing YouTube...")
            pause_youtube()
        elif mar <= YAWN_THRESH:
            yawn_detected = False

    # Hand detection using MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_detected = True
        # Draw hand landmarks on frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If hand detected, execute tab switch once
    if hand_detected:
        print("Hand detected! Switching tab...")
        switch_tab()

    # Show the output frame
    cv2.imshow("Yawn & Hand Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
