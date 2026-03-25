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


mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------- CONFIG ----------------

conf_threshold = 0.60
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
    "Hello": "Hello! Nice to see you.",
    "Thank You": "You're welcome!",
    "Good Morning": "Good Morning! Have a productive day.",
    "Sorry": "It's okay, no problem.",
    "I Love You": "Aww, that's sweet!"
}


def speak_label(label):
    global last_spoken, last_time

    if not auto_speak or label in ["-", "Unknown"]:
        return

    # 🔹 Smart reply
    speech_text = smart_replies.get(label, label)

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
        # 🔹 English speech with emotion
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
                "I Love You": "मैं तुमसे प्यार करता हूँ"
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
        
        res = hands.process(rgb)
        
        if frame_count % 5 == 0:
            face_res = face_mesh.process(rgb)
        else:
            face_res = None

        if face_res and face_res.multi_face_landmarks:
           face = face_res.multi_face_landmarks[0]

    # Example simple rule (student level)
           mouth_open = face.landmark[13].y - face.landmark[14].y

           if mouth_open < -0.02:
               current_emotion = "Happy"
           elif mouth_open > 0.02:
             current_emotion = "Sad"
           else:
               current_emotion = "Neutral"


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
            # 🔥 No hand detected
            pred_label = "Unknown"
            pred_conf = 0.0

        

        if pred_conf < conf_threshold:
            pred_label = "Unknown"

        # Stability check
        # 🔁 Simple Stability (2 consistent frames)

        if pred_label == last_label:
              stable_count += 1
        else:
              stable_count = 1
              last_label = pred_label

# Accept only if stable
        if stable_count < 2:
              pred_label = "Unknown"

        speak_label(pred_label)

        # Draw text
        cv2.putText(
            frame,
            f"{pred_label} ({pred_conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
             frame,
             f"Emotion: {current_emotion}",
             (10, 60),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.7,
             (255, 255, 0),
             2
        )

        # Update shared frame
        latency_ms = round((time.time() - start_time) * 1000, 2)

        global total_predictions, unknown_predictions, latency_history

        
        # Only count when a hand is actually detected
        if res.multi_hand_landmarks:

            total_predictions += 1

            if pred_label == "Unknown":
                  unknown_predictions += 1  
        
        
        latency_history.append(latency_ms)
        if len(latency_history) > 100:
            latency_history.pop(0)





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
        return jsonify(latest_pred)

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