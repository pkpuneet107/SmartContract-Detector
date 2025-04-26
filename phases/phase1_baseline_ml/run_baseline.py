# run_baseline.py

from src.data.load_dataset import load_and_preprocess, split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load and preprocess
print("🔍 Loading and preprocessing data...")
X, y = load_and_preprocess(
    "data/raw/BCCC-VolSCs-2023_Secure.csv",
    "data/raw/BCCC-VolSCs-2023_Vulnerable.csv",
    save_path="data/processed/merged_cleaned.csv"
)

# Split
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Train Random Forest
print("🌲 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Validation evaluation
print("\n Validation Results:")
y_val_pred = model.predict(X_val)
print(classification_report(y_val, y_val_pred))

# Test evaluation
print("\n Test Results:")
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred))

# Save the trained model - store in .joblib for no retraining. 
os.makedirs("experiments", exist_ok=True)
joblib.dump(model, "experiments/random_forest_model.joblib")
print("Model saved to: experiments/random_forest_model.joblib")
