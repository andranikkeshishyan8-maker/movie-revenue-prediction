"""
evaluation.py
-------------
Evaluates the trained regression model and saves results.

Metrics:
    - MAE  (Mean Absolute Error)  — average dollar error in log-space
    - MSE  (Mean Squared Error)   — penalizes large errors more
    - R²   (R-squared Score)      — proportion of variance explained (0–1)

Results saved to:
    outputs/results/metrics.txt
    outputs/results/predictions.csv

Interpretation:
    - R² close to 1.0  → model explains most of the variance in revenue
    - R² close to 0.0  → model barely better than predicting the mean
    - MAE tells us how far off predictions are on average (in log scale)
    - We also back-transform predictions to USD for intuitive comparison
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs/results", exist_ok=True)


def evaluate_model(model, X_test, y_test, df=None):
    """
    Evaluate the model on the test set and save results.

    Args:
        model:  Trained scikit-learn model
        X_test: Scaled test features
        y_test: True log-revenue values
        df:     Full DataFrame (for adding movie titles to predictions)

    Returns:
        dict of metrics
    """
    print("\n[Evaluating model...]")

    # ── Scikit-Learn Task 4: Make predictions ─────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Scikit-Learn Task 5: Compute metrics ──────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Back-transform from log scale to USD for human-readable error
    true_revenue  = np.expm1(np.array(y_test))
    pred_revenue  = np.expm1(y_pred)
    mae_usd = mean_absolute_error(true_revenue, pred_revenue)

    print(f"  MAE  (log scale):  {mae:.4f}")
    print(f"  RMSE (log scale):  {rmse:.4f}")
    print(f"  MSE  (log scale):  {mse:.4f}")
    print(f"  R²   score:        {r2:.4f}")
    print(f"  MAE  (USD):        ${mae_usd:,.0f}")

    # ── Model explanation ─────────────────────────────────────────────────────
    if r2 >= 0.70:
        quality = "strong"
    elif r2 >= 0.50:
        quality = "moderate"
    else:
        quality = "weak"
    explanation = (
        f"\nModel Evaluation Summary\n"
        f"========================\n"
        f"Task:  Regression — predicting box office revenue\n"
        f"Model: Decision Tree Regressor (max_depth=6)\n\n"
        f"Metrics on test set:\n"
        f"  MAE  (log scale) : {mae:.4f}\n"
        f"  RMSE (log scale) : {rmse:.4f}\n"
        f"  MSE  (log scale) : {mse:.4f}\n"
        f"  R²   score       : {r2:.4f}\n"
        f"  MAE  (USD)       : ${mae_usd:,.0f}\n\n"
        f"Interpretation:\n"
        f"  The R² score of {r2:.4f} indicates a {quality} fit.\n"
        f"  On average the model's revenue estimate is off by ~${mae_usd/1e6:.1f}M USD.\n"
        f"  Budget and popularity are the strongest predictors of revenue.\n\n"
        f"What could improve the model:\n"
        f"  - Adding genre, cast popularity, or director track record as features\n"
        f"  - Using a Random Forest or Gradient Boosting model\n"
        f"  - Collecting more movies (the current filtered set is ~1,000-2,000 rows)\n"
        f"  - Engineering features like 'sequel indicator' or 'release season'\n"
    )

    # ── Scikit-Learn Task 6: Save metrics ─────────────────────────────────────
    with open("outputs/results/metrics.txt", "w") as f:
        f.write(explanation)
    print("\nSaved: outputs/results/metrics.txt")

    # ── Scikit-Learn Task 7: Save predictions CSV ─────────────────────────────
    results_df = pd.DataFrame({
        "true_log_revenue":  np.array(y_test),
        "pred_log_revenue":  y_pred,
        "true_revenue_usd":  true_revenue,
        "pred_revenue_usd":  pred_revenue,
        "error_usd":         pred_revenue - true_revenue
    })
    results_df.to_csv("outputs/results/predictions.csv", index=False)
    print("Saved: outputs/results/predictions.csv")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mae_usd": mae_usd}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from preprocessing import preprocess_data
    from model import train_model
    X_train, X_test, y_train, y_test, df, feature_cols, scaler = preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, df)
