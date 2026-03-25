import pyttsx3

engine = pyttsx3.init(driverName='nsss')  # Force macOS driver

# Pick a specific voice (Samantha is a safe default for US English)
engine.setProperty('voice', 'com.apple.voice.compact.en-US.Samantha')

engine.say("Hello Aatir , now I should be speaking with Samantha's voice.where do you live in kolkata")
engine.runAndWait()
  




#   # app_live.py
# from flask import Flask, render_template, jsonify
# import threading, time, json, os
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


# # ---- Load model & labels ----
# MODEL_PATH = "models/gesture_knn.joblib"
# LABELS_PATH = "models/labels.json"

# if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
#     raise SystemExit("Train the model first (run train_model.py).")

# pipe = joblib.load(MODEL_PATH)
# labels = json.load(open(LABELS_PATH))


# # ---- MediaPipe setup ----
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6
# )
# mp_draw = mp.solutions.drawing_utils


# def extract_features(landmarks):
#     xs = [lm.x for lm in landmarks]
#     ys = [lm.y for lm in landmarks]
#     zs = [lm.z for lm in landmarks]
#     x0, y0, z0 = xs[0], ys[0], zs[0]
#     xs = [x - x0 for x in xs]
#     ys = [y - y0 for y in ys]
#     zs = [z - z0 for z in zs]
#     return np.array(xs + ys + zs, dtype=np.float32).reshape(1, -1)


# # ---- Shared state ----
# latest_frame = np.zeros((360, 640, 3), dtype=np.uint8)   # safe default
# latest_pred = {"label": "-", "conf": 0.0}
# lock = threading.Lock()
# running = True


# def camera_loop():
#     global latest_frame, latest_pred, running
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("⚠️ Could not open camera. Using black frame only.")
#         return

#     while running:
#         ok, frame = cap.read()
#         if not ok:
#             continue

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         pred_label, pred_conf = "-", 0.0

#         if res.multi_hand_landmarks:
#             lms = res.multi_hand_landmarks[0]
#             mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
#             feats = extract_features(lms.landmark)

#             try:
#                 pred_idx = pipe.predict(feats)[0]
#                 pred_label = labels[str(pred_idx)] if str(pred_idx) in labels else str(pred_idx)

#                 # Try confidence estimation (works only if pipeline has scaler + knn)
#                 if "clf" in pipe.named_steps and "scaler" in pipe.named_steps:
#                     clf = pipe.named_steps["clf"]
#                     scaler = pipe.named_steps["scaler"]
#                     dists, _ = clf.kneighbors(scaler.transform(feats), n_neighbors=clf.n_neighbors)
#                     pred_conf = float(np.clip(1.0 / (1e-6 + dists.mean()), 0, 1))
#                 else:
#                     pred_conf = 0.7
#             except Exception as e:
#                 print("Prediction error:", e)
#                 pred_label, pred_conf = "-", 0.0

#         # draw HUD
#         cv2.putText(frame, f"{pred_label} ({pred_conf:.2f})", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#         with lock:
#             latest_frame = frame.copy()
#             latest_pred = {"label": pred_label, "conf": round(pred_conf, 2)}

#         time.sleep(0.01)

#     cap.release()


# # ---- Flask app ----
# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/live")
# def home():
#     return render_template("live.html")

# @app.route("/frame.jpg")
# def frame_jpg():
#     with lock:
#         frame = latest_frame.copy()
#     ok, buf = cv2.imencode(".jpg", frame)
#     return (buf.tobytes(), 200, {
#         "Content-Type": "image/jpeg",
#         "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
#     })

# @app.route("/prediction")
# def prediction():
#     with lock:
#         return jsonify(latest_pred)


# def start_camera():
#     t = threading.Thread(target=camera_loop, daemon=True)
#     t.start()
#     return t


# if __name__ == "__main__":
#     cam_thread = start_camera()
#     try:
#         app.run(debug=True, port=5001)
#     finally:
#         running = False
#         cam_thread.join(timeout=1)



































# phase two little bit production level code so the code runs just changed due to simplyfy 
# from flask import Flask, render_template, jsonify, redirect, url_for
# import threading, time, json, os
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# #  Paths 
# MODEL_PATH = "models/gesture_knn.joblib"
# LABELS_PATH = "models/labels.json"

# if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
#     raise SystemExit("Train the model first: python train_model.py")

# #  Load model & labels 
# pipe = joblib.load(MODEL_PATH)

# # labels.json saved by train_model.py is a simple list 
# with open(LABELS_PATH, "r") as f:
#     labels = json.load(f)  # list
# assert isinstance(labels, list) and len(labels) > 0, "labels.json must be a list"

# # MediaPipe 
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6,
# )
# mp_draw = mp.solutions.drawing_utils

# def extract_features(landmarks):
#     xs = [lm.x for lm in landmarks]
#     ys = [lm.y for lm in landmarks]
#     zs = [lm.z for lm in landmarks]
#     x0, y0, z0 = xs[0], ys[0], zs[0]
#     xs = [x - x0 for x in xs]
#     ys = [y - y0 for y in ys]
#     zs = [z - z0 for z in zs]
#     return np.array(xs + ys + zs, dtype=np.float32).reshape(1, -1)

# #  Shared State 
# latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# latest_pred = {"label": "-", "conf": 0.0}
# lock = threading.Lock()
# running = True

# def camera_loop():
#     global latest_frame, latest_pred, running
#     cap = cv2.VideoCapture(0)  # try 1 if no laptop camera
#     if not cap.isOpened():
#         print("⚠️ Could not open camera. Showing a black frame.")
#         return

#     while running:
#         ok, frame = cap.read()
#         if not ok:
#             time.sleep(0.01)
#             continue

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         pred_label, pred_conf = "-", 0.0

#         if res.multi_hand_landmarks:
#             lms = res.multi_hand_landmarks[0]
#             mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

#             feats = extract_features(lms.landmark)
#             try:
#                 pred_idx = int(pipe.predict(feats)[0])
#                 # labels is a list -> index directly
#                 if 0 <= pred_idx < len(labels):
#                     pred_label = labels[pred_idx]
#                 # crude confidence from KNN distances if available
#                 if "clf" in getattr(pipe, "named_steps", {}):
#                     clf = pipe.named_steps["clf"]
#                     if hasattr(clf, "kneighbors"):
#                         # if there's a scaler in the pipeline, use it
#                         X = feats
#                         if "scaler" in pipe.named_steps:
#                             X = pipe.named_steps["scaler"].transform(feats)
#                         dists, _ = clf.kneighbors(X, n_neighbors=clf.n_neighbors)
#                         # convert mean distance to a [0,1]-ish score
#                         mean_d = float(dists.mean())
#                         pred_conf = float(1.0 / (1.0 + mean_d))
#                     else:
#                         pred_conf = 0.7
#                 else:
#                     pred_conf = 0.7
#             except Exception as e:
#                 print("Prediction error:", e)
#                 pred_label, pred_conf = "-", 0.0

#         cv2.putText(
#             frame, f"{pred_label} ({pred_conf:.2f})", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
#         )

#         with lock:
#             latest_frame = frame.copy()
#             latest_pred = {"label": pred_label, "conf": round(pred_conf, 2)}

#         time.sleep(0.01)

#     cap.release()

# #  Flask 
# app = Flask(__name__, template_folder="templates", static_folder="static")

# @app.route("/")
# def root_redirect():
#     # Avoid 405 by redirecting straight to /live
#     return redirect(url_for("live_page"))

# @app.route("/live")
# def live_page():
#     return render_template("live.html")

# @app.route("/frame.jpg")
# def frame_jpg():
#     with lock:
#         frame = latest_frame.copy()
#     ok, buf = cv2.imencode(".jpg", frame)
#     return (buf.tobytes(), 200, {
#         "Content-Type": "image/jpeg",
#         "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
#     })

# @app.route("/prediction")
# def prediction():
#     with lock:
#         return jsonify(latest_pred)

# def start_camera():
#     t = threading.Thread(target=camera_loop, daemon=True)
#     t.start()
#     return t

# if __name__ == "__main__":
#     cam_thread = start_camera()
#     try:
#         # Separate port from app.py; no reloader so the camera thread isn't duplicated
#         app.run(host="127.0.0.1", port=5003, debug=False, use_reloader=False)
#     finally:
#         running = False
#         cam_thread.join(timeout=1)




























# Phase 3 we have completed with student level code and its prediciting the gesture and taking to the website and speaking the output as well 5 gesture are given 
# AND WE HAVE COMPLETED SPRINT 1 AND JUST LEFT WITH THE FEW EXPLAINATION LIKE WHAT TO EXPLAIN TP GUIDE NOW WE WILL MOVE TO SPRINT 2 


# APP_LIVE.PY


# from flask import Flask, render_template, jsonify, redirect, url_for
# import threading, time, json, os
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import warnings
# from gtts import gTTS
# import playsound
# import tempfile

# warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# #  Paths 
# MODEL_PATH = "models/gesture_knn.joblib"
# LABELS_PATH = "models/labels.json"

# if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
#     raise SystemExit("Train the model first: python train_model.py")

# #  Load model & labels 
# pipe = joblib.load(MODEL_PATH)

# # labels.json saved by train_model.py is a simple list 
# with open(LABELS_PATH, "r") as f:
#     labels = json.load(f)  # list
# assert isinstance(labels, list) and len(labels) > 0, "labels.json must be a list"

# # MediaPipe 
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.6,
#     min_tracking_confidence=0.6,
# )
# mp_draw = mp.solutions.drawing_utils

# def extract_features(landmarks):
#     xs = [lm.x for lm in landmarks]
#     ys = [lm.y for lm in landmarks]
#     zs = [lm.z for lm in landmarks]
#     x0, y0, z0 = xs[0], ys[0], zs[0]
#     xs = [x - x0 for x in xs]
#     ys = [y - y0 for y in ys]
#     zs = [z - z0 for z in zs]
#     return np.array(xs + ys + zs, dtype=np.float32).reshape(1, -1)

# # ---------------------------
# # 🗣️ Text to Speech Function
# # ---------------------------
# last_spoken = ""
# last_time = 0

# def speak_label(label):
#     global last_spoken, last_time
#     if label != "-" and label != last_spoken and time.time() - last_time > 3:
#         try:
#             with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
#                 tts = gTTS(text=label, lang='en')
#                 tts.save(fp.name)
#                 playsound.playsound(fp.name)
#             last_spoken = label
#             last_time = time.time()
#         except Exception as e:
#             print("Speech error:", e)

# #  Shared State 
# latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# latest_pred = {"label": "-", "conf": 0.0}
# lock = threading.Lock()
# running = True

# def camera_loop():
#     global latest_frame, latest_pred, running
#     cap = cv2.VideoCapture(0)  # try 1 if no laptop camera
#     if not cap.isOpened():
#         print("⚠️ Could not open camera. Showing a black frame.")
#         return

#     while running:
#         ok, frame = cap.read()
#         if not ok:
#             time.sleep(0.01)
#             continue

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         pred_label, pred_conf = "-", 0.0

#         if res.multi_hand_landmarks:
#             lms = res.multi_hand_landmarks[0]
#             mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

#             feats = extract_features(lms.landmark)
#             try:
#                 pred_idx = int(pipe.predict(feats)[0])
#                 # labels is a list -> index directly
#                 if 0 <= pred_idx < len(labels):
#                     pred_label = labels[pred_idx]
#                 # crude confidence from KNN distances if available
#                 if "clf" in getattr(pipe, "named_steps", {}):
#                     clf = pipe.named_steps["clf"]
#                     if hasattr(clf, "kneighbors"):
#                         # if there's a scaler in the pipeline, use it
#                         X = feats
#                         if "scaler" in pipe.named_steps:
#                             X = pipe.named_steps["scaler"].transform(feats)
#                         dists, _ = clf.kneighbors(X, n_neighbors=clf.n_neighbors)
#                         # convert mean distance to a [0,1]-ish score
#                         mean_d = float(dists.mean())
#                         pred_conf = float(1.0 / (1.0 + mean_d))
#                     else:
#                         pred_conf = 0.7
#                 else:
#                     pred_conf = 0.7
#             except Exception as e:
#                 print("Prediction error:", e)
#                 pred_label, pred_conf = "-", 0.0

#         # 🗣️ Speak detected label
#         speak_label(pred_label)

#         cv2.putText(
#             frame, f"{pred_label} ({pred_conf:.2f})", (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
#         )

#         with lock:
#             latest_frame = frame.copy()
#             latest_pred = {"label": pred_label, "conf": round(pred_conf, 2)}

#         time.sleep(0.01)

#     cap.release()

# #  Flask 
# app = Flask(__name__, template_folder="templates", static_folder="static")

# @app.route("/")
# def root_redirect():
#     # Avoid 405 by redirecting straight to /live
#     return redirect(url_for("live_page"))

# @app.route("/live")
# def live_page():
#     return render_template("live.html")

# @app.route("/frame.jpg")
# def frame_jpg():
#     with lock:
#         frame = latest_frame.copy()
#     ok, buf = cv2.imencode(".jpg", frame)
#     return (buf.tobytes(), 200, {
#         "Content-Type": "image/jpeg",
#         "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
#     })

# @app.route("/prediction")
# def prediction():
#     with lock:
#         return jsonify(latest_pred)

# def start_camera():
#     t = threading.Thread(target=camera_loop, daemon=True)
#     t.start()
#     return t

# if __name__ == "__main__":
#     cam_thread = start_camera()
#     try:
#         # Separate port from app.py; no reloader so the camera thread isn't duplicated
#         app.run(host="127.0.0.1", port=5003, debug=False, use_reloader=False)
#     finally:
#         running = False
#         cam_thread.join(timeout=1)


# CAPTURE_DATA.PY

# # capture_data.py
# import os, csv, time
# import cv2
# import mediapipe as mp

# SAVE_PATH = "data/landmarks.csv"
# LABELS = ["Hello", "Thank You", "Yes", "No", "Good Morning"]  # 5 static signs

# os.makedirs("data", exist_ok=True)

# # Write header once
# if not os.path.exists(SAVE_PATH):
#     with open(SAVE_PATH, "w", newline="") as f:
#         writer = csv.writer(f)
#         header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
#         writer.writerow(header)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
#                        min_detection_confidence=0.6, min_tracking_confidence=0.6)
# mp_draw = mp.solutions.drawing_utils

# def extract_features(landmarks):
#     # Normalize by wrist (index 0) to reduce translation effects
#     xs = [lm.x for lm in landmarks]
#     ys = [lm.y for lm in landmarks]
#     zs = [lm.z for lm in landmarks]
#     x0, y0, z0 = xs[0], ys[0], zs[0]
#     xs = [x - x0 for x in xs]
#     ys = [y - y0 for y in ys]
#     zs = [z - z0 for z in zs]
#     return xs + ys + zs

# cap = cv2.VideoCapture(0)
# current_label_idx = 0
# count_for_label = 0

# print("\nControls:")
# print("  [1..5]  -> choose label:", dict(zip(range(1,6), LABELS)))
# print("  [C]     -> capture one sample")
# print("  [A]     -> auto-capture 25 samples (2/sec)")
# print("  [Q]     -> quit\n")

# auto = False
# auto_left = 0

# while True:
#     ok, frame = cap.read()
#     if not ok:
#         print("Camera read failed.")
#         break
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     res = hands.process(rgb)

#     h, w = frame.shape[:2]
#     if res.multi_hand_landmarks:
#         for handLms in res.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#         lms = res.multi_hand_landmarks[0].landmark
#         feats = extract_features(lms)

#         if auto and auto_left > 0:
#             with open(SAVE_PATH, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([LABELS[current_label_idx]] + feats)
#             count_for_label += 1
#             auto_left -= 1
#             time.sleep(0.5)  # ~2 per sec

#     # HUD
#     txt = f"Label[{current_label_idx+1}]: {LABELS[current_label_idx]} | Samples for this run: {count_for_label}"
#     if auto: txt += f" | Auto left: {auto_left}"
#     cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

#     cv2.imshow("Capture Landmarks - SignSpeak", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
#         current_label_idx = int(chr(key)) - 1
#         count_for_label = 0
#         auto = False

#     elif key in [ord('c'), ord('C')]:
#         if res.multi_hand_landmarks:
#             with open(SAVE_PATH, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([LABELS[current_label_idx]] + feats)
#             count_for_label += 1
#             print(f"[+] Captured 1 sample for '{LABELS[current_label_idx]}'")

#     elif key in [ord('a'), ord('A')]:
#         auto = True
#         auto_left = 25
#         count_for_label = 0
#         print(f"[*] Auto-capturing 25 samples for '{LABELS[current_label_idx]}' ... hold steady")

#     elif key in [ord('q'), ord('Q')]:
#         break

# cap.release()
# cv2.destroyAllWindows()





# LIVE,HTML


# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8">
#   <title>Live Camera Feed</title>
#   <style>
#     body {
#       font-family: Arial, sans-serif;
#       background: #111;
#       color: #eee;
#       text-align: center;
#     }
#     h1 {
#       margin-top: 20px;
#     }
#     #video-container {
#       margin-top: 20px;
#     }
#     #frame {
#       width: 640px;
#       height: 480px;
#       border: 3px solid #555;
#       border-radius: 10px;
#     }
#     #prediction {
#       margin-top: 20px;
#       font-size: 20px;
#       font-weight: bold;
#       color: #00ff88;
#     }
#   </style>
# </head>
# <body>
#   <h1>📷 Live Camera Feed</h1>

#   <div id="video-container">
#     <img id="frame" src="/frame.jpg" alt="Live camera feed">
#   </div>

#   <div id="prediction">
#     Prediction: <span id="label">Loading...</span>
#   </div>

#   <script>
#     // Refresh the image every 100ms
#     function refreshFrame() {
#       const frame = document.getElementById("frame");
#       frame.src = "/frame.jpg?t=" + new Date().getTime();
#     }
#     setInterval(refreshFrame, 100);

#     // Refresh prediction every 1s
#     async function refreshPrediction() {
#       try {
#         let res = await fetch("/prediction");
#         let data = await res.json();
#         document.getElementById("label").textContent = data.label;
#       } catch (err) {
#         console.error("Prediction fetch error:", err);
#       }
#     }
#     setInterval(refreshPrediction, 1000);
#   </script>
# </body>
# </html>



