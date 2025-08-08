import cv2
import dlib
import os
import time
from scipy.spatial import distance as dist
import mediapipe as mp

# --- CONFIGURATION ---
# Path to dlib's facial landmark predictor
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Facial landmark indices for the mouth
(mStart, mEnd) = (48, 68)

# Thresholds and delays
YAWN_THRESH = 0.6
ACTION_DELAY = 2  # seconds delay between actions

# --- INITIALIZATION ---
# Dlib for yawn detection
print("[INFO] Loading dlib face detector and predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

# MediaPipe for hand gesture detection
print("[INFO] Loading MediaPipe hand detector...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Timestamps for action cooldown
last_action_time = 0

# --- ACTION FUNCTIONS ---
def pause_youtube():
    """Sends a 'spacebar' key press to the system."""
    print("Yawn detected! Pausing YouTube...")
    os.system("xdotool key space")

def close_tab():
    """Sends a 'ctrl+w' key press to the system to close the current tab."""
    print("Victory Sign detected! Closing tab...")
    os.system("xdotool key ctrl+w")

# --- CORE LOGIC FUNCTIONS ---
def mouth_aspect_ratio(mouth):
    """Calculates the mouth aspect ratio (MAR) to detect yawns."""
    # Vertical distances
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    # Horizontal distance
    C = dist.euclidean(mouth[0], mouth[6])
    # Mouth Aspect Ratio
    mar = (A + B) / (2.0 * C)
    return mar

def is_victory_sign(hand_landmarks):
    """
    Checks if the hand landmarks form a victory/peace sign (✌️).
    Returns True if it is, False otherwise.
    """
    try:
        # Get y-coordinates of fingertips and relevant knuckles
        # Note: In image coordinates, a smaller 'y' is higher up.
        tip_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        pip_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

        tip_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        pip_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

        tip_ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        pip_ring = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y

        tip_pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_TIP].y
        pip_pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_FINGER_PIP].y

        # Check conditions:
        # 1. Index and Middle fingers are extended (tip is above the pip joint)
        index_up = tip_index < pip_index
        middle_up = tip_middle < pip_middle
        # 2. Ring and Pinky fingers are curled (tip is below the pip joint)
        ring_down = tip_ring > pip_ring
        pinky_down = tip_pinky > pip_pinky

        if index_up and middle_up and ring_down and pinky_down:
            return True
    except Exception as e:
        print(f"Error in gesture recognition: {e}")
        return False
    return False


# --- MAIN LOOP ---
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0) # Allow camera to warm up

yawn_detected_flag = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    current_time = time.time()
    action_ready = (current_time - last_action_time) > ACTION_DELAY

    # 1. Yawn Detection
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        if mar > YAWN_THRESH:
            if not yawn_detected_flag and action_ready:
                pause_youtube()
                last_action_time = current_time
            yawn_detected_flag = True
        else:
            yawn_detected_flag = False
        
        # Draw the mouth outline
        mouth_hull = cv2.convexHull(np.array(mouth))
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (rect.left(), rect.top() - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 2. Hand Gesture Detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and action_ready:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check for victory sign
            if is_victory_sign(hand_landmarks):
                close_tab()
                last_action_time = current_time
                # Break after first successful gesture to avoid multiple triggers
                break 

    # Display status on screen
    if not action_ready:
        cooldown_remaining = ACTION_DELAY - (current_time - last_action_time)
        cv2.putText(frame, f"COOLDOWN: {cooldown_remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "STATUS: READY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
hands.close()
