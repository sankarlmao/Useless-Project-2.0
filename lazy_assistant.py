import cv2
import numpy as np
import tensorflow.keras
import pyautogui
import time
import os
import platform

# ==============================
# Load Teachable Machine model
# ==============================
try:
    model = tensorflow.keras.models.load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
except Exception as e:
    print("Error loading model or labels. Make sure keras_model.h5 and labels.txt are in the same folder.")
    print(e)
    exit()

# ==============================
# Setup webcam
# ==============================
cap = cv2.VideoCapture(0)
eye_closed_start = None  # Timer for eyes closed

# ==============================
# Helper: Sleep/Shutdown
# ==============================
def sleep_system():
    os_name = platform.system()
    if os_name == "Windows":
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")  # Sleep
    elif os_name == "Darwin":  # macOS
        os.system("pmset sleepnow")
    elif os_name == "Linux":
        os.system("systemctl suspend")
    else:
        print("Unsupported OS for sleep command.")

# ==============================
# Main loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    # Prepare frame for prediction
    img = cv2.resize(frame, (224, 224))
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    img = (img / 127.5) - 1  # Normalize to [-1, 1]

    prediction = model.predict(img)
    index = np.argmax(prediction)
    class_name = class_names[index].strip().lower()
    confidence_score = prediction[0][index]

    # Show prediction on screen
    cv2.putText(frame, f"{class_name} ({confidence_score*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === Part 1: Yawn detection ===
    if class_name == "yawn" and confidence_score > 0.85:
        print("[ACTION] Yawn detected → Pressing Space")
        pyautogui.press("space")
        time.sleep(1)  # Avoid multiple triggers

    # === Part 2: Eyes closed for >10 sec ===
    if class_name == "eyes_closed" and confidence_score > 0.85:
        if eye_closed_start is None:
            eye_closed_start = time.time()
        elif time.time() - eye_closed_start >= 10:
            print("[ACTION] Eyes closed >10s → Sleeping system")
            sleep_system()
            break
    else:
        eye_closed_start = None

    # === Part 3: Palm gesture ===
    if class_name == "palm" and confidence_score > 0.85:
        print("[ACTION] Palm detected → Switching browser tab")
        pyautogui.hotkey("ctrl", "tab")
        time.sleep(1)

    # Display webcam
    cv2.imshow("Lazy Assistant", frame)

    # Exit if Q pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
