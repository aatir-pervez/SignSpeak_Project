# 🤟 SignSpeak: Real-Time Indian Sign Language to Speech Converter with Emotional Context Awareness

## Overview
SignSpeak is a real-time system that converts Indian Sign Language (ISL) gestures into speech with emotional context awareness.  
It uses computer vision and machine learning to detect hand gestures and facial expressions, generating meaningful spoken output.

---

## Features
- Real-time hand gesture recognition (10+ gestures)
- Emotion detection using MediaPipe FaceMesh
- Emotion-aware speech output (Happy / Sad / Neutral)
- English + Hindi support
- Mobile access (same WiFi)
- Performance tracking (latency + accuracy)
- Smart replies

---

## Tech Stack
- Python
- Flask
- OpenCV
- MediaPipe
- Scikit-learn
- gTTS
- NumPy

---

## System Flow

Webcam  
↓  
Hand Detection → Gesture Prediction  
↓  
Face Detection → Emotion  
↓  
Combine Both  
↓  
Text + Speech Output  

---

## How to Run

1. Open terminal  
2. Run:

pip install -r requirements.txt  
python app_live.py  

---

## Mobile Use

Open in phone browser:

http://YOUR-IP:5003/live  

(make sure same WiFi)

---

## Performance

- Accuracy: ~85–95%
- Latency: ~20–40 ms

---

## Authors

- Aatir Pervez  
- Kashif  
- Arin  

---

## Future Work

- Dynamic gestures  
- Mobile app  
- Deep learning models  