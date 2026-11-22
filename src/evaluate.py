# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Evaluation: metrics, the unified results table and stored test predictions.

Every model is scored on exactly the same test rows, on the ORIGINAL case scale,
and compared with the naive "next week = this week" baseline.

Besides overall errors we split the test MAE into
  * normal weeks  : actual cases <= the district's training 75th percentile
  * outbreak weeks: actual cases >  that threshold (the "High" risk weeks)
because a model can look good overall while failing exactly when it matters.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODELS_DIR

RESULTS_CSV = MODELS_DIR / "results.csv"
PREDICTIONS_CSV = MODELS_DIR / "test_predictions.csv"
VAL_PREDICTIONS_CSV = MODELS_DIR / "val_predictions.csv"
RESULT_COLUMNS = ["model", "family", "MAE", "RMSE", "R2", "sMAPE",
                  "MAE_vs_naive_pct", "RMSE_vs_naive_pct",
                  "MAE_normal", "MAE_outbreak", "val_MAE", "train_time_s"]
NAIVE_NAME = "Naive (last week)"
# "Neural net" holds the stand-alone ANN, which is a component of the RF+ANN hybrids, not an ensemble.
FAMILY_ORDER = ["Baseline", "Bagging", "Boosting", "Stacking", "Neural net", "Hybrid"]


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


def improvement(model_value, naive_value):
    """% reduction in error vs the naive baseline (positive = better than naive)."""
    return 100 * (naive_value - model_value) / naive_value


def outbreak_mask(df: pd.DataFrame) -> np.ndarray:
    """True where the actual target exceeds the district's training 75th percentile."""
    from .features import load_feature_config
    thr = load_feature_config()["risk_thresholds"]
    high = df["district"].map(lambda d: thr[d]["high"]).to_numpy()
    return df["target"].to_numpy() > high


# ---------------------------------------------------------------- results table
def load_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def record_result(model: str, family: str, test: pd.DataFrame, test_pred,
                  train_time_s: float = 0.0, val: pd.DataFrame | None = None, val_pred=None) -> dict:
    """Score a model on the test rows, upsert its row in models/results.csv and store predictions.

    `val`/`val_pred` (optional) add the validation MAE, which is what we use to SELECT models;
    the test numbers are only reported, never used for choosing.
    """
    test_pred = np.clip(np.asarray(test_pred, float), 0, None)
    y = test["target"].to_numpy()
    ob = outbreak_mask(test)
    row = {"model": model, "family": family, **regression_metrics(y, test_pred),
           "MAE_normal": mean_absolute_error(y[~ob], test_pred[~ob]),
           "MAE_outbreak": mean_absolute_error(y[ob], test_pred[ob]),
           "val_MAE": (mean_absolute_error(val["target"], np.clip(val_pred, 0, None))
                       if val is not None else np.nan),
           "train_time_s": round(float(train_time_s), 2)}

    res = load_results()
    res = pd.concat([res[res["model"] != model], pd.DataFrame([row])], ignore_index=True)
    naive = res[res["model"] == NAIVE_NAME]
    if len(naive):
        res["MAE_vs_naive_pct"] = improvement(res["MAE"], naive["MAE"].iloc[0])
        res["RMSE_vs_naive_pct"] = improvement(res["RMSE"], naive["RMSE"].iloc[0])
    res["_order"] = res["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    res = res.sort_values(["_order", "MAE"]).drop(columns="_order")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    res[RESULT_COLUMNS].round(4).to_csv(RESULTS_CSV, index=False)

    _store_predictions(PREDICTIONS_CSV, model, test, test_pred)
    if val is not None:
        _store_predictions(VAL_PREDICTIONS_CSV, model, val, val_pred)
    return row


def _store_predictions(path, model, df, y_pred):
    """One row per (district, target week), one column per model; used by the dashboard."""
    key = df[["district", "target_date", "target"]].reset_index(drop=True)
    if path.exists():
        preds = pd.read_csv(path, parse_dates=["target_date"]).drop(columns=["target"])
        preds = key.merge(preds.drop(columns=[model], errors="ignore"), on=["district", "target_date"], how="left")
    else:
        preds = key.copy()
    preds[model] = np.clip(np.asarray(y_pred, float), 0, None).round(2)
    preds.to_csv(path, index=False)


def load_predictions(split: str = "test") -> pd.DataFrame:
    path = PREDICTIONS_CSV if split == "test" else VAL_PREDICTIONS_CSV
    return pd.read_csv(path, parse_dates=["target_date"])


def results_table(res: pd.DataFrame | None = None) -> pd.DataFrame:
    """Nicely ordered results for display (family order, then MAE)."""
    res = load_results() if res is None else res
    res = res.assign(_o=res["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
    return res.sort_values(["_o", "MAE"]).drop(columns="_o").reset_index(drop=True)
