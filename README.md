<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# Lazy_assistant 🎯


## Basic Details
### Team Name: [Cheesecake]


### Team Members
- Team Lead: [Sankar B] - [MITS]
- Member 1: [Linsa Biji] - [MITS]

### Project Description

This project uses your webcam to spot when you yawn or show certain hand gestures to control media and browser tabs without touching the keyboard. When you yawn, it pauses YouTube, a simple hand wave switches tabs, and showing the victory sign closes the tab. It’s a fun way to interact hands-free using some cool face and hand tracking tech.

 
### The Problem (that doesn't exist)
Why bother using your hands when you can just yawn or flash a peace sign to control your computer? Because lazy tech is the best tech!
 
### The Solution (that nobody asked for)
Control your computer with yawns and peace signs — no fingers required! It’s the laziest, quirkiest way to pause videos and switch tabs, just because you can.



## Technical Details
### Technologies/Components Used
For Software:

OpenCV: For capturing and processing video frames in real-time.
Dlib: To detect facial landmarks and calculate mouth aspect ratio for yawn detection.
MediaPipe Hands: For accurate hand tracking and gesture recognition.
xdotool: A Linux utility to simulate keyboard inputs like space, ctrl+tab, and ctrl+w.
Python: The main programming language used to integrate all components seamlessly.


### Implementation
For Software:
# Installation

sudo apt update
sudo apt install -y python3-pip python3-opencv xdotool
pip3 install dlib mediapipe scipy numpy

# Run
python3 -m venv venv(for debain based , just python -m venv venv for arch based distros)
source venv/bin/activate
python3 lazy_assistant.py

### Project Documentation
For Software:

Source Code: Contains the main Python script integrating OpenCV, Dlib, and MediaPipe for real-time yawn and hand gesture detection.
Dependencies: Requires Python 3, OpenCV, Dlib, MediaPipe, SciPy, NumPy, and xdotool installed on a Linux system.
Data Files: Includes the shape_predictor_68_face_landmarks.dat file for facial landmark detection.
Usage: Captures webcam video, detects yawns and hand gestures, and sends keyboard commands to control media playback and browser tabs.
Gesture Logic:
Yawn detection uses mouth aspect ratio (MAR) thresholding.
Hand gestures are detected with MediaPipe; a victory sign triggers tab close (Ctrl + W), other hand detections switch tabs (Ctrl + Tab).
User Interaction: Real-time visual feedback via OpenCV window displaying webcam feed with landmarks and gesture outlines.

# Screenshots (Add at least 3)
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/155389fe-65a0-4d28-9e4c-b07d4b455bc4" />
yawn detecting

This screenshot shows a custom-built AI program that detects when a user yawns and automatically pauses their YouTube video.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/84a06c8f-7af4-4261-af69-fd9426df0d46" />
Switching browser tab with one hand.

Here, the same AI project is shown using hand gesture recognition to switch browser tabs, demonstrating its 'lazy assistant' capabilities.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/510a4f8b-ca6a-4cef-8a58-7889f2ce7b47" />
Closing tab with two hands.
A demonstration of a hand gesture recognition program, part of a 'lazy assistant' project, used to close tab.


### Project Demo
# Video
[(https://drive.google.com/file/d/12A17TOKXQZdr1_WQnOqhTeqeTAZhvpXc/view?usp=drivesdk)]
*Explain what the video demonstrates*


## Team Contributions
- [Sankar]: [My work involved building its central feature—a real-time computer vision engine for yawn and gesture detection using Python, OpenCV, and MediaPipe]
- [Linsa]: [I used Roboflow to train the custom detection model, from preparing the dataset and training the model.]
  
---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)



