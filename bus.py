import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/heart.csv")

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# -----------------------------
# Best Model
# -----------------------------
model = grid.best_estimator_

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Heart Disease Prediction")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print("\nBest Parameters:")
print(grid.best_params_)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Model trained successfully.")
print("✅ model.pkl created.")