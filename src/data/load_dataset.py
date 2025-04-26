# src/data/load_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_and_preprocess(path_secure, path_vulnerable, save_path=None):
    df_secure = pd.read_csv(path_secure)
    df_vulnerable = pd.read_csv(path_vulnerable)

    df_secure["label"] = 0
    df_vulnerable["label"] = 1

    df = pd.concat([df_secure, df_vulnerable], ignore_index=True)

    # Remove unwanted columns
    drop_cols = ["Unnamed: 0", "hash_id", "ast_nodetype", "ast_src"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    df = df.dropna()

    if save_path:
        df.to_csv(save_path, index=False)

    X = df.drop(columns=["label"]).select_dtypes(include=["number"]).astype("float32")
    y = df["label"]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nonzero_mask = ~np.all(X_scaled == 0, axis=1)
    X_scaled = X_scaled[nonzero_mask]
    y = y[nonzero_mask]

    return X_scaled, y


def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    # First split into train and temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split temp into val and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
