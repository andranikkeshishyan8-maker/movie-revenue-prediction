"""
model.py
--------
Trains a machine learning model to predict movie revenue.

ML Task:    REGRESSION
Model:      Linear Regression (baseline) + Decision Tree Regressor (main model)
Target:     log_revenue — log-transformed box office revenue

We train on log-transformed revenue because revenue is heavily right-skewed.
Predictions are later exponentiated back to actual USD values.
"""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def train_model(X_train, y_train, model_type="decision_tree"):
    """
    Train a regression model.

    Args:
        X_train:    Scaled training features
        y_train:    Log-transformed revenue target
        model_type: 'linear_regression' or 'decision_tree' (default)

    Returns:
        Trained model object
    """
    print(f"\n[Training model: {model_type}]")

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        # Decision Tree Regressor — main model
        # max_depth=6 prevents overfitting while capturing non-linear patterns
        model = DecisionTreeRegressor(max_depth=6, min_samples_split=10,
                                      random_state=42)

    model.fit(X_train, y_train)
    print(f"Model trained on {len(X_train)} samples.")

    # ── NumPy Task: Inspect training predictions ──────────────────────────────
    train_preds = model.predict(X_train)
    train_residuals = np.array(y_train) - train_preds
    print(f"Training residuals — mean: {np.mean(train_residuals):.4f}, "
          f"std: {np.std(train_residuals):.4f}")

    return model


if __name__ == "__main__":
    # Standalone test
    from preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, df, feature_cols, scaler = preprocess_data()
    model = train_model(X_train, y_train)
    print("Model ready.")
