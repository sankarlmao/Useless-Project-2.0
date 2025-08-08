import cv2
import dlib
import os
import time
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp

# --- CONFIGURATION ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
(mStart, mEnd) = (48, 68)
YAWN_THRESH = 0.6
ACTION_DELAY = 2

# --- INITIALIZATION ---
print("[INFO] Loading dlib and MediaPipe models...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

mp_hands = mp.solutions.hands
# Lowering confidence to increase detection chances, adjust if needed
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

last_action_time = 0

# --- ACTION FUNCTIONS ---
def perform_action(action_name):
    """Performs an action based on the name and prints it."""
    global last_action_time
    # Ensure we don't trigger actions too fast
    if (time.time() - last_action_time) < ACTION_DELAY:
        return
        
    print(f"ACTION: Triggered '{action_name}'")
    if action_name == "pause_youtube":
        os.system("xdotool key space")
    elif action_name == "close_tab":
        os.system("xdotool key ctrl+w")
    elif action_name == "switch_tab":
        os.system("xdotool key ctrl+Tab")
        
    last_action_time = time.time()


# --- GESTURE RECOGNITION FUNCTIONS ---

def is_victory_sign(hand_landmarks):
    """
    More robust check for a victory/peace sign (✌️).
    Compares fingertips to the base of the finger (MCP joint) for stability.
    """
    # 
    try:
        # Get all landmark coordinates
        lm = hand_landmarks.landmark

        # Condition 1: Index and Middle fingers are extended UP.
        # Their tips should be above their MCP joints (base of the finger).
        index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

        # Condition 2: Ring and Pinky fingers are curled DOWN.
        # Their tips should be below their MCP joints.
        ring_down = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_down = lm[mp_hands.HandLandmark.PINKY_FINGER_TIP].y > lm[mp_hands.HandLandmark.PINKY_FINGER_MCP].y
        
        # Condition 3: The thumb is tucked in.
        # The thumb tip should be below the ring finger's PIP joint.
        thumb_tucked = lm[mp_hands.HandLandmark.THUMB_TIP].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y

        if index_up and middle_up and ring_down and pinky_down and thumb_tucked:
            return True
    except Exception:
        return False
    return False

def is_palm_open(hand_landmarks):
    """
    Checks if the hand is open in a 'stop' or 'palm' gesture (✋).
    All four main fingers should be extended.
    """
    try:
        lm = hand_landmarks.landmark
        
        # We check if the fingertips are above their PIP joint (the middle joint)
        index_up = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_up = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_up = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_up = lm[mp_hands.HandLandmark.PINKY_FINGER_TIP].y < lm[mp_hands.HandLandmark.PINKY_FINGER_PIP].y

        if index_up and middle_up and ring_up and pinky_up:
            return True
    except Exception:
        return False
    return False


# --- MAIN LOOP ---
print("[INFO] Starting video stream... Press 'q' to quit.")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    # Store frame dimensions
    h, w, c = frame.shape
    
    current_time = time.time()
    action_ready = (current_time - last_action_time) > ACTION_DELAY

    # --- Hand Gesture Detection ---
    # Convert color for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_detected = "None"
    if results.multi_hand_landmarks:
        # We only use the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check for gestures only if the cooldown has passed
        if action_ready:
            if is_victory_sign(hand_landmarks):
                gesture_detected = "Victory Sign"
                # Assign an action to this gesture
                perform_action("close_tab")
            elif is_palm_open(hand_landmarks):
                gesture_detected = "Open Palm"
                # Assign a different action to this gesture
                perform_action("switch_tab")

    # Display Gesture Status
    cv2.putText(frame, f"Gesture: {gesture_detected}", (10, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)


    # Display Cooldown Status
    if not action_ready:
        cooldown_rem = ACTION_DELAY - (current_time - last_action_time)
        cv2.putText(frame, f"COOLDOWN: {cooldown_rem:.1f}s", (10, 30),
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
