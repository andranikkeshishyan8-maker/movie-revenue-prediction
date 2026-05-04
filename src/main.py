"""
main.py
-------
Entry point for the TMDB Movie Revenue Prediction project.

Runs the full ML pipeline:
    1. Data exploration
    2. Preprocessing
    3. Visualization
    4. Model training
    5. Evaluation

Usage:
    python src/main.py
"""

import sys
import os

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from data_exploration import explore_data
from preprocessing import preprocess_data
from visualization import create_all_visualizations
from model import train_model
from evaluation import evaluate_model


def main():
    print("=" * 60)
    print("  TMDB Movie Revenue Prediction — Full ML Pipeline")
    print("=" * 60)

    DATASET_PATH = "data/dataset.csv"

    # ── Step 1: Explore ───────────────────────────────────────────────────────
    print("\n[STEP 1] Data Exploration")
    df_raw = explore_data(DATASET_PATH)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n[STEP 2] Preprocessing")
    X_train, X_test, y_train, y_test, df_clean, feature_cols, scaler = \
        preprocess_data(DATASET_PATH)

    # ── Step 3: Visualize ─────────────────────────────────────────────────────
    print("\n[STEP 3] Visualization")
    create_all_visualizations(df_raw)

    # ── Step 4: Train model ───────────────────────────────────────────────────
    print("\n[STEP 4] Model Training")
    model = train_model(X_train, y_train, model_type="decision_tree")

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print("\n[STEP 5] Evaluation")
    metrics = evaluate_model(model, X_test, y_test, df_clean)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  R² Score : {metrics['r2']:.4f}")
    print(f"  MAE (USD): ${metrics['mae_usd']:,.0f}")
    print("\n  Outputs saved to:")
    print("    outputs/plots/   — 3 PNG + 2 HTML visualizations")
    print("    outputs/results/ — metrics.txt + predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
