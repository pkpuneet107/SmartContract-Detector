# test_load_data.py

from src.data.load_dataset import load_and_preprocess, split_data

# Define file paths
secure_path = "data/raw/BCCC-VolSCs-2023_Secure.csv"
vulnerable_path = "data/raw/BCCC-VolSCs-2023_Vulnerable.csv"

# Run loading and preprocessing
print("Loading and preprocessing data...")
X, y = load_and_preprocess(secure_path, vulnerable_path)

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {y.value_counts().to_dict()}")

# Split the data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

print("Split successful!")
print(f"Train set: {X_train.shape}, {len(y_train)} labels")
print(f"Val set:   {X_val.shape}, {len(y_val)} labels")
print(f"Test set:  {X_test.shape}, {len(y_test)} labels")

X, y = load_and_preprocess(
    secure_path,
    vulnerable_path,
    save_path="data/processed/merged_cleaned.csv"
)
