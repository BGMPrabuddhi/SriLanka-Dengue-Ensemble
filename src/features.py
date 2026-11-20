"""Feature engineering, chronological split, preprocessing pipeline and risk thresholds.

Timing convention (the key to avoiding leakage)
------------------------------------------------
Each row is one *forecast*: standing at the end of reporting week t ("origin"), we
predict cases in week t+1 ("target"). Every feature uses data from week t or earlier:

    cases_lag1 = cases in week t        (the most recent observed week)
    cases_lag4 = cases in week t-3
    rolling_mean_4 = mean of cases in weeks t-3 .. t
    rain_mm_lag1   = rainfall in week t,  rain_mm_lag8 = rainfall in week t-7
    week_sin/cos, year = calendar of the TARGET week (known in advance, so not leakage)

All per-district operations use groupby("district") so one region's history never
leaks into another's.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (DATA_PROCESSED, N_LAGS, RISK_PERCENTILES, TEST_START_YEAR,
                     TRAIN_END_YEAR, VAL_YEAR, WEATHER_LAGS, WEATHER_VARS)

WARMUP_WEEKS = 8                          # first weeks of each series have no history
EXCLUDE_LAST_N_WEEKS = 2                  # most recent weeks may still receive late reports

CASE_LAG_FEATURES = [f"cases_lag{k}" for k in range(1, N_LAGS + 1)]
ROLLING_FEATURES = ["rolling_mean_4", "rolling_std_4", "rolling_mean_8"]
WEATHER_LAG_FEATURES = [f"{v}_lag{k}" for v in WEATHER_VARS for k in WEATHER_LAGS]
CALENDAR_FEATURES = ["week_sin", "week_cos", "year"]
CATEGORICAL_FEATURES = ["district"]

FEATURES_CSV = DATA_PROCESSED / "features.csv"
FORECAST_CSV = DATA_PROCESSED / "forecast_features.csv"
FEATURE_CONFIG_JSON = DATA_PROCESSED / "feature_config.json"


def numeric_features(with_weather: bool = True) -> list[str]:
    weather = WEATHER_LAG_FEATURES if with_weather else []
    return CASE_LAG_FEATURES + ROLLING_FEATURES + weather + CALENDAR_FEATURES


def all_features(with_weather: bool = True) -> list[str]:
    return numeric_features(with_weather) + CATEGORICAL_FEATURES


def _calendar(dates: pd.Series) -> pd.DataFrame:
    """Same year/week rule as data.clean_dengue, plus cyclical encoding of the week."""
    mid = dates + pd.Timedelta(days=3)
    week = (mid.dt.dayofyear - 1) // 7 + 1
    angle = 2 * np.pi * week / 52.18      # 52.18 = average weeks per year
    return pd.DataFrame({"year": mid.dt.year, "week": week,
                         "week_sin": np.sin(angle), "week_cos": np.cos(angle)}, index=dates.index)


def build_panel(dengue: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (district, origin week t) with features, target (= cases at t+1) and baselines."""
    df = dengue.sort_values(["district", "start_date"]).reset_index(drop=True)
    if weather is not None:
        df = df.merge(weather[["district", "start_date", *WEATHER_VARS]],
                      on=["district", "start_date"], how="left")
    g = df.groupby("district", group_keys=False)

    out = pd.DataFrame({"district": df["district"], "origin_date": df["start_date"]})
    out["target_date"] = g["start_date"].shift(-1)
    # The last row of each district is the live forecast row: its target week is the next one.
    out["target_date"] = out["target_date"].fillna(out["origin_date"] + pd.Timedelta(days=7))
    out["target"] = g["cases"].shift(-1)

    # --- lagged cases (lag1 = week t itself)
    for k in range(1, N_LAGS + 1):
        out[f"cases_lag{k}"] = g["cases"].shift(k - 1)

    # --- rolling statistics over weeks <= t. NaN (missing reports) are skipped by pandas
    # as long as at least `min_periods` real values are in the window.
    def roll(window, fn, min_periods):
        return g["cases"].transform(lambda s: s.rolling(window, min_periods=min_periods).agg(fn))
    out["rolling_mean_4"] = roll(4, "mean", 2)
    out["rolling_std_4"] = roll(4, "std", 2)
    out["rolling_mean_8"] = roll(8, "mean", 4)

    # --- lagged weather (lag1 = week t)
    if weather is not None:
        for v in WEATHER_VARS:
            for k in WEATHER_LAGS:
                out[f"{v}_lag{k}"] = g[v].shift(k - 1)

    # --- calendar of the TARGET week
    cal = _calendar(out["target_date"])
    out["year"], out["week"] = cal["year"], cal["week"]
    out["week_sin"], out["week_cos"] = cal["week_sin"], cal["week_cos"]

    # --- baselines (use the last *observed* value if a week is missing)
    last_obs = g["cases"].ffill()
    out["naive_pred"] = last_obs                                   # "next week = this week"
    out["seasonal_pred"] = last_obs.groupby(df["district"]).shift(51)  # same week last year (t+1-52)

    out["history_weeks"] = g.cumcount() + 1
    return out


def split_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological split on the TARGET week's year. Never shuffled.

    Returns train / val / test (rows with a known target) and `forecast` (latest week per
    district, target unknown, used by the dashboard). The last EXCLUDE_LAST_N_WEEKS target
    weeks are excluded from test because they may still be revised by late reports.
    """
    last_date = panel["origin_date"].max()
    forecast = panel[panel["origin_date"] == last_date].copy()

    usable = panel[(panel["history_weeks"] > WARMUP_WEEKS) & panel["target"].notna()]
    recent = np.sort(usable["target_date"].unique())[-EXCLUDE_LAST_N_WEEKS:]
    return {
        "train": usable[usable["year"] <= TRAIN_END_YEAR].copy(),
        "val": usable[usable["year"] == VAL_YEAR].copy(),
        "test": usable[(usable["year"] >= TEST_START_YEAR) & ~usable["target_date"].isin(recent)].copy(),
        "excluded_recent": usable[usable["target_date"].isin(recent)].copy(),
        "forecast": forecast,
    }


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(num_features: list[str], scale: bool = False) -> ColumnTransformer:
    """Imputation (+ optional scaling) for numeric columns, one-hot for district.

    SimpleImputer(median, add_indicator=True) fills NaN lags (from missing reports) with the
    TRAINING median and adds a 0/1 "was missing" column for every feature that had NaNs in
    training. Because it sits inside the model Pipeline, it is fit on training data only,
    including within each TimeSeriesSplit fold during tuning.
    Scaling is only needed for distance/gradient based models (MLP); trees are scale-invariant.
    """
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num_features),
         ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False)


def with_target_transform(regressor, use_log: bool):
    """Optionally train on log1p(cases) and convert predictions back with expm1.

    log1p compresses outbreak spikes (2,600 cases vs a typical 9) so squared-error
    trees don't spend all their splits on a few extreme weeks.
    """
    if not use_log:
        return regressor
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)


def make_model_pipeline(regressor, num_features, use_log: bool, scale: bool = False):
    """preprocessor -> (optionally log-target-wrapped) regressor."""
    return Pipeline([("prep", make_preprocessor(num_features, scale)),
                     ("model", with_target_transform(regressor, use_log))])


# ---------------------------------------------------------------- risk levels
def compute_risk_thresholds(dengue: pd.DataFrame) -> dict[str, dict[str, float]]:
    """District-specific percentiles of weekly cases in the TRAINING period only."""
    train = dengue[(dengue["year"] <= TRAIN_END_YEAR) & dengue["cases"].notna()]
    thr = {}
    for district, s in train.groupby("district")["cases"]:
        thr[district] = {"medium": float(np.percentile(s, RISK_PERCENTILES["medium"])),
                         "high": float(np.percentile(s, RISK_PERCENTILES["high"]))}
    return thr


def risk_level(cases: float, district: str, thresholds: dict) -> str:
    """Low  : below the district's median week
       Medium: between median and 75th percentile
       High  : above the district's 75th percentile (a busier-than-usual week)."""
    t = thresholds[district]
    if cases > t["high"]:
        return "High"
    if cases > t["medium"]:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------- persistence
def save_feature_config(cfg: dict):
    FEATURE_CONFIG_JSON.write_text(json.dumps(cfg, indent=2, default=str))


def load_feature_config() -> dict:
    return json.loads(FEATURE_CONFIG_JSON.read_text())


def load_splits() -> dict[str, pd.DataFrame]:
    """Re-create the splits from the saved feature table (used by notebooks 03-07)."""
    panel = pd.read_csv(FEATURES_CSV, parse_dates=["origin_date", "target_date"])
    return split_panel(panel)
