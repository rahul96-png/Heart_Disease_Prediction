import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/student-mat.csv", sep=";")

# -----------------------------
# Create Target Column
# Pass = 1 (G3 >= 10)
# Fail = 0 (G3 < 10)
# -----------------------------
df["Result"] = df["G3"].apply(lambda x: 1 if x >= 10 else 0)

# -----------------------------
# Select Features
# -----------------------------
X = df[[
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]]

# Target
y = df["Result"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Test Accuracy
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Student Performance Prediction")
print("=" * 50)
print(f"Accuracy : {accuracy:.2f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model trained successfully!")
print("✅ model.pkl created successfully.")
