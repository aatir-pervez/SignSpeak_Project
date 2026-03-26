# capture_data.py
import os, csv, time
import cv2
import mediapipe as mp

# -----------------------------
# 🧠 File setup
# -----------------------------
SAVE_PATH = "data/landmarks.csv"
LABELS = [
    "Hello", "Thank You", "Yes", "No", "Good Morning",
    "I Love You", "Sorry", "Good Night", "Please", "Welcome", "Help", "Stop", "Good", "Bad", "Fine"
]  # 15 static signs

os.makedirs("data", exist_ok=True)

# create CSV with header if not exists
if not os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
        writer.writerow(header)

# -----------------------------
# ✋ MediaPipe Hand Setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils


# -----------------------------
# 📐 Feature Extraction with Normalization
# -----------------------------
def extract_features(landmarks):
    # Get all landmark coordinates
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]

    # Step 1: shift (normalize by wrist position)
    x0, y0, z0 = xs[0], ys[0], zs[0]
    xs = [x - x0 for x in xs]
    ys = [y - y0 for y in ys]
    zs = [z - z0 for z in zs]

    # Step 2: scale normalization — make all hands same relative size
    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys)
    )

    if max_range != 0:
        xs = [x / max_range for x in xs]
        ys = [y / max_range for y in ys]
        zs = [z / max_range for z in zs]

    # Step 3: return feature vector
    return xs + ys + zs



# -----------------------------
# 📸 Capture Setup
# -----------------------------
cap = cv2.VideoCapture(0)
current_label_idx = 0
count_for_label = 0

print("\nControls:")
print("  [1-9,0,-,=,u,i,o] -> choose label:", dict(zip(range(1, len(LABELS)+1), LABELS)))
print("  [C]     -> capture one sample")
print("  [A]     -> auto-capture 25 samples (2/sec)")
print("  [Q]     -> quit\n")

auto = False
auto_left = 0

# -----------------------------
# 🎥 Main Loop
# -----------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ Camera read failed.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        lms = res.multi_hand_landmarks[0].landmark
        feats = extract_features(lms)

        # auto-capture mode
        if auto and auto_left > 0:
            with open(SAVE_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([LABELS[current_label_idx]] + feats)
            count_for_label += 1
            auto_left -= 1
            time.sleep(0.5)  # 2 samples per second

    # display info
    txt = f"Label[{current_label_idx+1}]: {LABELS[current_label_idx]} | Samples this run: {count_for_label}"
    if auto:
        txt += f" | Auto left: {auto_left}"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Capture Landmarks - SignSpeak", frame)
    key = cv2.waitKey(1) & 0xFF

    
        # choose label
    if key in [ord(str(i)) for i in range(1, 10)]:
        current_label_idx = int(chr(key)) - 1
        count_for_label = 0
        auto = False

    elif key == ord('0'):  # 10th → Welcome
        current_label_idx = 9
        count_for_label = 0
        auto = False

    elif key == ord('-'):  # 11th → Help
        current_label_idx = 10
        count_for_label = 0
        auto = False

    elif key == ord('='):  # 12th → Stop
        current_label_idx = 11
        count_for_label = 0
        auto = False

    elif key == ord('u'):  # 13th → Good
        current_label_idx = 12
        count_for_label = 0
        auto = False

    elif key == ord('i'):  # 14th → Bad
        current_label_idx = 13
        count_for_label = 0
        auto = False

    elif key == ord('o'):  # 15th → Fine
        current_label_idx = 14
        count_for_label = 0
        auto = False
    # capture manually
    elif key in [ord('c'), ord('C')]:
        if res.multi_hand_landmarks:
            with open(SAVE_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([LABELS[current_label_idx]] + feats)
            count_for_label += 1
            print(f"[+] Captured 1 sample for '{LABELS[current_label_idx]}'")

    # auto-capture
    elif key in [ord('a'), ord('A')]:
        auto = True
        auto_left = 25
        count_for_label = 0
        print(f"[*] Auto-capturing 25 samples for '{LABELS[current_label_idx]}' ... hold steady")

    # quit
    elif key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()