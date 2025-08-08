# 💤 Lazy Assistant – Because Why Not? ✋

Ever wished your computer could just... *do stuff* when you make random faces or wave at it?  
Meet **Lazy Assistant** – the totally unnecessary but weirdly satisfying project that reacts to your yawns, eye-shutting moments, and palm waves.

## 🤔 What It Does
- **Yawn** 😪 → Instantly hits `Space` for you (perfect for pausing YouTube mid-binge).
- **show a 'perfect' hand sign** 😴 → Sends your PC straight to sleep (and maybe you too).
- **Show your palm** ✋ → Switches to the next browser tab (Ctrl+Tab magic).

Basically, it’s like having a lazy, slightly creepy roommate that watches you through your webcam and pushes buttons for you.

## 🛠 What You’ll Need
- Python 3.8+  
- A webcam (built-in or USB)  
- A trained [Teachable Machine](https://teachablemachine.withgoogle.com/) model with:
  - `yawn`
  - `eyes_closed`
  - `palm`
- Two files from Teachable Machine:
  - `keras_model.h5` (the brain)
  - `labels.txt` (the brain’s dictionary)

## 📦 How to Set It Up
1. Clone this repo or just throw all the files in one folder.
2. Install all the Python magic:
   ```bash
   pip install opencv-python pyautogui tensorflow numpy pillow keyboard
