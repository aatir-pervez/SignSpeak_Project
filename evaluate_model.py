# evaluate_model.py.               
# ye file tumhare trained model (gesture_knn.joblib + labels.json) ko load karegi
# aur ek confusion matrix + classification report generate karegi
# taaki tum accuracy aur mis-classification dekh sako.
import os, json, joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

DATA_CSV = "data/landmarks.csv"
MODEL_PATH = "models/gesture_knn.joblib"
LABELS_PATH = "models/labels.json"

# --- Load data and model ---
df = pd.read_csv(DATA_CSV)
y_true_str = df["label"].values
X = df.drop(columns=["label"]).values

pipe = joblib.load(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# Encode labels (same as training)
le = LabelEncoder()
le.fit(labels)
y_true = le.transform(y_true_str)
y_pred = pipe.predict(X)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - SignSpeak Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("static/reports", exist_ok=True)
plt.savefig("static/reports/confusion_matrix.png")
plt.close()

# --- Classification Report ---
report = classification_report(y_true, y_pred, target_names=labels)
print("\nModel Evaluation Report:\n")
print(report)

# Save text report
with open("static/reports/accuracy_report.txt", "w") as f:
    f.write(report)

print("\n✅ Saved confusion_matrix.png and accuracy_report.txt in static/reports/")
