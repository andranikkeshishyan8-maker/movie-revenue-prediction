"""
preprocessing.py
----------------
Cleans and prepares the TMDB movies dataset for machine learning.

Features used:
    - budget        (numerical)
    - popularity    (numerical)
    - runtime       (numerical)
    - vote_average  (numerical)
    - vote_count    (numerical)
    - release_year  (derived from release_date)

Target:
    - revenue       (numerical — regression target)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(filepath="data/dataset.csv"):
    # ── Pandas Task 1: Load ───────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── Preprocessing Task 1: Remove unnecessary columns ─────────────────────
    keep_cols = ["title", "budget", "revenue", "popularity",
                 "runtime", "vote_average", "vote_count", "release_date"]
    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    # ── Preprocessing Task 2 / Pandas Task 6: Handle missing values ───────────
    df.dropna(subset=["budget", "revenue", "popularity",
                      "runtime", "vote_average", "vote_count"], inplace=True)

    # ── Pandas Task 11: Create new column — release_year ─────────────────────
    if "release_date" in df.columns:
        df["release_year"] = pd.to_datetime(
            df["release_date"], errors="coerce").dt.year

    # ── Preprocessing Task 3: Filter rows with valid budget and revenue ────────
    # Movies with 0 budget or 0 revenue are incomplete records
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

    # ── Pandas Task 12: Rename column for clarity ─────────────────────────────
    df.rename(columns={"vote_average": "rating",
                        "vote_count": "num_votes"}, inplace=True)

    # ── Pandas Task 11b: Create budget_to_revenue ratio column ───────────────
    df["budget_millions"] = df["budget"] / 1_000_000

    # ── NumPy Task 3: Log-transform skewed columns (budget & revenue) ─────────
    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])   # this is our target
    df["log_popularity"] = np.log1p(df["popularity"])

    # ── Preprocessing Task 4: Define features and target ─────────────────────
    feature_cols = ["log_budget", "log_popularity", "runtime",
                    "rating", "num_votes"]
    if "release_year" in df.columns:
        df["release_year"].fillna(df["release_year"].median(), inplace=True)
        feature_cols.append("release_year")

    X = df[feature_cols].copy()
    y = df["log_revenue"].copy()

    # ── Preprocessing Task 5: Check for any remaining NaNs ───────────────────
    X.fillna(X.median(), inplace=True)

    # ── NumPy Task 4: Confirm shapes with NumPy ───────────────────────────────
    print(f"Feature matrix shape: {np.array(X).shape}")
    print(f"Target vector shape:  {np.array(y).shape}")

    # ── Preprocessing Task 6 (check shape) ───────────────────────────────────
    print(f"\nFeatures: {feature_cols}")
    print(f"Target:   log_revenue (log-transformed box office revenue)")

    # ── Preprocessing Task 7: Scale features ─────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Scikit-Learn Task 1: Train/test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    return X_train, X_test, y_train, y_test, df, feature_cols, scaler


if __name__ == "__main__":
    preprocess_data()
