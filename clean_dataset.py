import pandas as pd

df = pd.read_csv("data/landmarks.csv")

remove_labels = [
    "Hello",
    "Thank You",
    "Yes",
    "No",
    "Good Morning",
    "Sorry"
]

df_clean = df[~df["label"].isin(remove_labels)]

df_clean.to_csv("data/landmarks.csv", index=False)

print("Old weak gesture samples removed.")
print(df_clean["label"].value_counts())