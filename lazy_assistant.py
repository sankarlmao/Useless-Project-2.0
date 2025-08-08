import cv2
import dlib
import pyautogui
import os
from scipy.spatial import distance as dist
import time

# Path to the dlib shape predictor file
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Indices for mouth landmarks
(mStart, mEnd) = (48, 68)

# Yawn detection threshold
YAWN_THRESH = 0.6
yawn_detected = False

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # vertical
    B = dist.euclidean(mouth[4], mouth[8])   # vertical
    C = dist.euclidean(mouth[0], mouth[6])   # horizontal
    return (A + B) / (2.0 * C)

# Function to pause YouTube in Chrome
def pause_youtube():
    # Simulate pressing space (works if video tab is active)
    os.system("xdotool key space")

# Function to switch Chrome tabs
def switch_tab(direction="next"):
    if direction == "next":
        os.system("xdotool key ctrl+Tab")
    elif direction == "prev":
        os.system("xdotool key ctrl+shift+Tab")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if mar > YAWN_THRESH and not yawn_detected:
            yawn_detected = True
            print("Yawn detected! Pausing YouTube...")
            pause_youtube()
            switch_tab("next")  # Change to "prev" if needed

        elif mar <= YAWN_THRESH:
            yawn_detected = False

    cv2.imshow("Yawn Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
s
