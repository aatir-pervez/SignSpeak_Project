# train_model.py
import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

DATA_CSV = "data/landmarks.csv"
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_CSV)
print("\nClass Distribution:\n")
print(df["label"].value_counts())
y_str = df["label"].values
X = df.drop(columns=["label"]).values

le = LabelEncoder()
y = le.fit_transform(y_str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Validation accuracy: {acc:.3f}")

joblib.dump(pipe, "models/gesture_knn.joblib")
with open("models/labels.json", "w") as f:
    json.dump(list(le.classes_), f)

print("Saved: models/gesture_knn.joblib and models/labels.json")
