# phases/phase1_baseline_ml/analyze_features.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model
model_path = "phases/phase1_baseline_ml/random_forest_model.joblib"
model = joblib.load(model_path)

# Load data
df = pd.read_csv("data/processed/merged_cleaned.csv")
X = df.drop(columns=["label"]).select_dtypes(include=["number"]).astype("float32")

# Extract importances
importances = model.feature_importances_
features_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Save importances
os.makedirs("experiments", exist_ok=True)
features_df.to_csv("experiments/feature_importance.csv", index=False)
print("📁 Feature importances saved to: experiments/feature_importance.csv")

# Plot top 20
top_n = 20
top_features = features_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title(f"Top {top_n} Most Important Features")
plt.xlabel("Importance (Random Forest)")
plt.tight_layout()
plt.savefig("experiments/top_features.png")
print("📊 Plot saved to: experiments/top_features.png")
plt.show()