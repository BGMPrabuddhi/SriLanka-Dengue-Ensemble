"""Evaluation: metrics, the unified results table and stored test predictions.

Every model is scored on exactly the same test rows, on the ORIGINAL case scale,
and compared with the naive "next week = this week" baseline.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODELS_DIR

RESULTS_CSV = MODELS_DIR / "results.csv"
PREDICTIONS_CSV = MODELS_DIR / "test_predictions.csv"
RESULT_COLUMNS = ["model", "family", "MAE", "RMSE", "R2", "sMAPE", "train_time_s",
                  "MAE_vs_naive_pct", "RMSE_vs_naive_pct"]
NAIVE_NAME = "Naive (last week)"


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in %, 0..200. Weeks where actual and predicted are both 0 count as perfect.

    Plain MAPE divides by the actual value and explodes when a district has 0 cases
    (12% of district-weeks), so sMAPE is used instead.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(2 * np.abs(y_pred - y_true), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(100 * ratio.mean())


def regression_metrics(y_true, y_pred) -> dict:
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)   # case counts cannot be negative
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "sMAPE": smape(y_true, y_pred)}


def improvement(model_value: float, naive_value: float) -> float:
    """% reduction in error vs the naive baseline (positive = better than naive)."""
    return 100 * (naive_value - model_value) / naive_value


# ---------------------------------------------------------------- results table
def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def record_result(model: str, family: str, y_true, y_pred, train_time_s: float = 0.0) -> dict:
    """Compute test metrics and upsert one row in models/results.csv (keyed by model name).

    The naive baseline must be recorded first; improvements are recomputed for all rows
    every time, so the table is always consistent.
    """
    row = {"model": model, "family": family, **regression_metrics(y_true, y_pred),
           "train_time_s": round(float(train_time_s), 2)}
    res = load_results()
    res = pd.concat([res[res["model"] != model], pd.DataFrame([row])], ignore_index=True)
    naive = res[res["model"] == NAIVE_NAME]
    if len(naive):
        res["MAE_vs_naive_pct"] = improvement(res["MAE"], naive["MAE"].iloc[0])
        res["RMSE_vs_naive_pct"] = improvement(res["RMSE"], naive["RMSE"].iloc[0])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    res[RESULT_COLUMNS].round(4).to_csv(RESULTS_CSV, index=False)
    return row


def save_predictions(model: str, test: pd.DataFrame, y_pred):
    """Store test predictions (one column per model) for actual-vs-predicted plots in the app."""
    key = test[["district", "target_date", "target"]].reset_index(drop=True)
    if PREDICTIONS_CSV.exists():
        preds = pd.read_csv(PREDICTIONS_CSV, parse_dates=["target_date"])
    else:
        preds = key.copy()
    preds = key.merge(preds.drop(columns=["target"], errors="ignore"), on=["district", "target_date"], how="left")
    preds[model] = np.clip(np.asarray(y_pred, float), 0, None).round(2)
    preds.to_csv(PREDICTIONS_CSV, index=False)
