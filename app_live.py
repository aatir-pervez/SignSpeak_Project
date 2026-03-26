from flask import Flask, render_template, jsonify, redirect, url_for
import threading
import time
import json
import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from gtts import gTTS
import playsound
import csv
from datetime import datetime
from flask import request
import warnings
warnings.filterwarnings("ignore")


emotion_last = "Neutral"
emotion_count = 0

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------- CONFIG ----------------

conf_threshold = 0.35
STABLE_FRAMES_REQUIRED = 3


# ---------------- GLOBAL STATE ----------------

latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
latest_pred = {"label": "-", "conf": 0.0}
lock = threading.Lock()
running = True

total_predictions = 0
unknown_predictions = 0
latency_history = []
last_label = None
stable_count = 0
frame_count = 0



current_emotion = "Neutral"

auto_speak = True
current_lang = "en"

os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/session_log.csv"

# create file with header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "gesture", "confidence", "emotion", "speech_text"])

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "models/gesture_knn.joblib"
LABELS_PATH = "models/labels.json"

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run train_model.py")

pipe = joblib.load(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# ---------------- MEDIAPIPE ----------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
mp_draw = mp.solutions.drawing_utils

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]

    x0, y0, z0 = xs[0], ys[0], zs[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]
    zs = [z - z0 for z in zs]

    max_range = max(max(xs) - min(xs), max(ys) - min(ys))
    if max_range != 0:
        xs = [x / max_range for x in xs]
        ys = [y / max_range for y in ys]
        zs = [z / max_range for z in zs]

    return xs + ys + zs

# ---------------- TEXT TO SPEECH ----------------

smart_replies = {
    "Hello": {
        "Happy": "Hello 😊 Nice to see you!",
        "Sad": "Hello... are you feeling okay?",
        "Neutral": "Hello."
    },
    "Thank You": {
        "Happy": "Thank you so much 😊",
        "Sad": "Thanks... I appreciate it.",
        "Neutral": "Thank you."
    },
    "Sorry": {
        "Happy": "Sorry 😊 it's okay!",
        "Sad": "I'm really sorry...",
        "Neutral": "Sorry."
    },
    "Good Morning": {
        "Happy": "Good morning 😊 have a great day!",
        "Sad": "Good morning... take care.",
        "Neutral": "Good morning."
    },
    "I Love You": {
        "Happy": "I love you ❤️",
        "Sad": "I love you... stay strong.",
        "Neutral": "I love you."
    },
    "Help": {
          "Happy": "I'm here to help 😊",
          "Sad": "Let me help you...",
          "Neutral": "I can help."
    },
    "Stop": {
           "Happy": "Stop 😊 that's enough!",
           "Sad": "Please stop...",
           "Neutral": "Stop."
    },
    "Good": {
           "Happy": "That's good 😊",
           "Sad": "It's okay...",
           "Neutral": "Good."
    },
    "Bad": {
         "Happy": "Oops 😅 that's not good",
         "Sad": "That's bad...",
         "Neutral": "Bad."
    },
    "Fine": {
         "Happy": "I'm doing great 😊",
         "Sad": "I'm fine...",
         "Neutral": "I'm fine."
    }
}

def speak_label(label):
    global last_spoken, last_time

    #  Skip invalid cases
    if not auto_speak or label in ["-", "Unknown"]:
        return

    # 🔹 Get emotion-aware speech
    if label in smart_replies:
        speech_text = smart_replies[label].get(current_emotion, label)
    else:
        speech_text = label

    # 📝 Log session
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            round(latest_pred.get("conf", 0), 2),
            current_emotion,
            speech_text
        ])

    try:
        # 🔍 Debug (VERY IMPORTANT)
        # print("LABEL:", label)
        # print("EMOTION:", current_emotion)
        # print("SPEAKING:", speech_text)

        # 🔹 English speech (only speed changes, text already decided)
        if current_lang == "en":

            if current_emotion == "Sad":
                tts = gTTS(text=speech_text, lang="en", slow=True)

            elif current_emotion == "Happy":
                tts = gTTS(text=speech_text, lang="en", slow=False)

            else:
                tts = gTTS(text=speech_text, lang="en", slow=False)

        # 🔹 Hindi speech
        else:
            hindi_map = {
                "Hello": "नमस्ते",
                "Good Morning": "सुप्रभात",
                "Thank You": "धन्यवाद",
                "Yes": "हाँ",
                "No": "नहीं",
                "Please": "कृपया",
                "Sorry": "मुझे माफ़ करें",
                "Welcome": "स्वागत है",
                "Good Night": "शुभ रात्रि",
                "I Love You": "मैं तुमसे प्यार करता हूँ",
                "Help": "मदद करो",
                "Stop": "रुको",
                "Good": "अच्छा",
                "Bad": "बुरा",
                "Fine": "ठीक हूँ",
            }

            hindi_text = hindi_map.get(label, label)
            tts = gTTS(text=hindi_text, lang="hi", slow=True)

        # 🔊 Play audio
        filename = "temp_speech.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)

        last_spoken = label
        last_time = time.time()

    except Exception as e:
        print("Speech error:", e)

# ---------------- CAMERA LOOP ----------------

def camera_loop():
    global latest_frame, latest_pred, running
    global last_label, stable_count, current_emotion, frame_count
    global total_predictions, unknown_predictions, latency_history

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Camera not opened")
        return

    print("Camera started")

    while running:

        frame_count += 1

        start_time = time.time()
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------------- HAND DETECTION ----------------
        res = hands.process(rgb)

        # ---------------- FACE DETECTION (OPTIMIZED) ----------------
        face_res = face_mesh.process(rgb)
        
                # ---------------- SIMPLE & WORKING EMOTION ----------------

        detected_emotion = "Neutral"

        if face_res and face_res.multi_face_landmarks:
           
            face = face_res.multi_face_landmarks[0]

            # 🔹 Simple Mouth-based detection
            upper_lip = face.landmark[13].y
            lower_lip = face.landmark[14].y
            mouth_gap = lower_lip - upper_lip

            # 🔥 Simple logic (works reliably)
            if mouth_gap > 0.02:
                detected_emotion = "Happy"

            elif mouth_gap < 0.015:
                detected_emotion = "Sad"

            else:
                detected_emotion = "Neutral"

       

        # ---------------- STABILITY FILTER ----------------
        global emotion_last, emotion_count

        if detected_emotion == emotion_last:
            emotion_count += 1
        else:
            emotion_count = 1
            emotion_last = detected_emotion

        # Accept only stable emotion
        if emotion_count >= 2:
            current_emotion = detected_emotion

        if frame_count % 30 == 0:
           print("Detected:", detected_emotion, "| Final:", current_emotion)

        # ---------------- GESTURE PREDICTION ----------------
        pred_label = "-"
        pred_conf = 0.0

        if res.multi_hand_landmarks:

            lms = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            feats = np.array(
                extract_features(lms.landmark),
                dtype=np.float32
            ).reshape(1, -1)

            try:
                pred_idx = int(pipe.predict(feats)[0])
                pred_label = labels[pred_idx]

                if hasattr(pipe, "predict_proba"):
                    probs = pipe.predict_proba(feats)[0]
                    pred_conf = float(np.max(probs))
                else:
                    pred_conf = 0.8

            except Exception:
                pred_label = "Unknown"
                pred_conf = 0.0

        else:
            pred_label = "Unknown"
            pred_conf = 0.0

        # ---------------- CONFIDENCE FILTER ----------------
        if pred_conf < conf_threshold:
            pred_label = "Unknown"

        # ---------------- STABILITY CHECK ----------------
        if pred_label == last_label:
            stable_count += 1
        else:
            stable_count = 1
            last_label = pred_label

        if stable_count < 1:
            pred_label = "Unknown"

        # ---------------- SPEAK ----------------
        speak_label(pred_label)

        # ---------------- DRAW TEXT ----------------
        cv2.putText(
            frame,
            f"{pred_label} ({pred_conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # 🎨 Emotion Color Logic
        color = (255, 255, 255)  # default (white)

        if current_emotion == "Happy":
            color = (0, 255, 0)   # Green
        elif current_emotion == "Sad":
            color = (0, 0, 255)   # Red

        # 🔥 BIG EMOTION DISPLAY
        cv2.putText(
            frame,
            f"Emotion: {current_emotion}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        # ---------------- LATENCY ----------------
        latency_ms = round((time.time() - start_time) * 1000, 2)

        if res.multi_hand_landmarks:
            total_predictions += 1

            if pred_label == "Unknown":
                unknown_predictions += 1

        latency_history.append(latency_ms)
        if len(latency_history) > 100:
            latency_history.pop(0)

        # ---------------- UPDATE FRAME ----------------
        with lock:
            latest_frame = frame.copy()

            latest_pred = {
                "label": pred_label,
                "conf": round(pred_conf, 2),
                "latency": latency_ms
            }

        time.sleep(0.01)

    cap.release()

# ---------------- FLASK ----------------

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def root_redirect():
    return redirect(url_for("live_page"))

@app.route("/live")
def live_page():
    return render_template("live.html")

@app.route("/frame.jpg")
def frame_jpg():
    with lock:
        frame = latest_frame.copy()
    ok, buf = cv2.imencode(".jpg", frame)
    return (buf.tobytes(), 200, {"Content-Type": "image/jpeg"})




@app.route("/prediction")
def prediction():
    with lock:
        return jsonify({
            "label": latest_pred.get("label"),
            "conf": latest_pred.get("conf"),
            "latency": latest_pred.get("latency"),
            "fps": latest_pred.get("fps", 0),
            "emotion": current_emotion   # 🔥 THIS FIXES YOUR ISSUE
        })




@app.route("/report")
def report():
    return render_template(
        "report.html",
        conf_img="/static/reports/confusion_matrix.png",
        accuracy_text=open("static/reports/accuracy_report.txt").read()
    )


@app.route("/toggle_speak", methods=["POST"])
def toggle_speak():
    global auto_speak
    auto_speak = not auto_speak
    return jsonify({"auto_speak": auto_speak})

@app.route("/toggle_lang", methods=["POST"])
def toggle_lang():
    global current_lang
    current_lang = "hi" if current_lang == "en" else "en"
    return jsonify({"lang": current_lang})

@app.route("/set_emotion/<emotion>", methods=["POST"])
def set_emotion(emotion):
    global current_emotion
    current_emotion = emotion
    return jsonify({"emotion": current_emotion})

@app.route("/settings")
def settings_page():
    return render_template("settings.html", threshold=conf_threshold)

@app.route("/update_threshold", methods=["POST"])
def update_threshold():
    global conf_threshold
    value = request.form.get("threshold")
    try:
        conf_threshold = float(value)
    except:
        pass
    return redirect(url_for("settings_page"))

@app.route("/benchmark")
def benchmark():
    avg_latency = round(sum(latency_history)/len(latency_history), 2) if latency_history else 0
    success_rate = 0

    if total_predictions > 0:
        success_rate = round(
            ((total_predictions - unknown_predictions) / total_predictions) * 100,
            2
        )

    return render_template(
        "benchmark.html",
        total=total_predictions,
        unknown=unknown_predictions,
        avg_latency=avg_latency,
        success=success_rate
    )

def start_camera():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=5003, debug=False, use_reloader=False)