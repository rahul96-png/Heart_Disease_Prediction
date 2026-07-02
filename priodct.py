import pickle
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Sample Input
# Change these values to test
# -----------------------------
sample = pd.DataFrame([{
    "age": 52,
    "sex": 1,
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3
}])

# -----------------------------
# Predict
# -----------------------------
prediction = model.predict(sample)

# -----------------------------
# Output
# -----------------------------
if prediction[0] == 1:
    print("⚠️ Heart Disease Detected")
else:
    print("✅ No Heart Disease Detected")