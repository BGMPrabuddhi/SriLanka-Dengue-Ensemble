# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Model definitions, time-aware tuning, early stopping and stacking.

Every model is a sklearn Pipeline:
    preprocessor (median-impute + was-missing flags + one-hot district)
      -> TransformedTargetRegressor(log1p / expm1)   [decided in notebook 02]
        -> the ensemble regressor
so parameter names look like  model__regressor__n_estimators.
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import features as F
from .config import MODELS_DIR, RANDOM_STATE

RS = RANDOM_STATE
P = "model__regressor__"          # prefix to reach the regressor inside pipeline + log-target wrapper
BEST_PARAMS_JSON = MODELS_DIR / "best_params.json"
MAX_FILE_MB = 95                  # GitHub hard limit is 100 MB per file


# ---------------------------------------------------------------- data helpers
def load_data():
    """Feature config, chronological splits (each sorted by time) and the feature list."""
    cfg = F.load_feature_config()
    splits = {k: time_sorted(v) for k, v in F.load_splits().items()}
    feats = cfg["numeric_features"] + cfg["categorical_features"]
    return cfg, splits, feats


def time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """TimeSeriesSplit cuts by ROW POSITION, so rows must be in chronological order."""
    return df.sort_values(["target_date", "district"], kind="mergesort").reset_index(drop=True)


def make_pipeline(regressor, cfg, scale=False):
    return F.make_model_pipeline(regressor, cfg["numeric_features"], cfg["use_log_target"], scale)


# ---------------------------------------------------------------- model catalogue
# name -> (family, constructor, search space, n_iter for RandomizedSearchCV)
# Search spaces are deliberately small so each notebook runs in minutes on a laptop.
# min_samples_leaf >= 3 also keeps forests small enough for GitHub (< 100 MB per file).
CATALOGUE = {
    # ----------------------------- bagging (parallel, reduce variance)
    "Bagging (Decision Trees)": ("Bagging",
        lambda: BaggingRegressor(DecisionTreeRegressor(random_state=RS), n_estimators=100,
                                 n_jobs=-1, random_state=RS),
        {P + "n_estimators": [100, 200], P + "max_samples": [0.5, 0.8, 1.0],
         P + "max_features": [0.5, 0.8, 1.0], P + "estimator__min_samples_leaf": [3, 5, 10]}, 6),
    "Random Forest": ("Bagging",
        lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
        {P + "n_estimators": [100, 200, 300], P + "max_features": [0.3, 0.5, 0.7, "sqrt"],
         P + "min_samples_leaf": [3, 5, 10], P + "max_depth": [None, 25]}, 8),
    "Extra Trees": ("Bagging",
        lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
        {P + "n_estimators": [100, 200, 300], P + "max_features": [0.3, 0.5, 0.7, 1.0],
         P + "min_samples_leaf": [3, 5, 10], P + "max_depth": [None, 25]}, 8),
    # ----------------------------- boosting (sequential, reduce bias)
    "AdaBoost": ("Boosting",
        lambda: AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, random_state=RS),
                                  n_estimators=200, random_state=RS),
        {P + "learning_rate": [0.03, 0.1, 0.3], P + "estimator__max_depth": [3, 5],
         P + "loss": ["linear", "square"]}, 4),
    "Gradient Boosting": ("Boosting",
        lambda: GradientBoostingRegressor(n_estimators=200, random_state=RS),
        {P + "learning_rate": [0.05, 0.1], P + "max_depth": [3, 4, 5],
         P + "subsample": [0.7, 0.9], P + "min_samples_leaf": [5, 20]}, 4),
    "XGBoost": ("Boosting",
        lambda: XGBRegressor(n_estimators=400, tree_method="hist", n_jobs=-1, random_state=RS),
        {P + "learning_rate": [0.03, 0.05, 0.1], P + "max_depth": [3, 5, 7],
         P + "subsample": [0.7, 0.9], P + "colsample_bytree": [0.5, 0.8],
         P + "min_child_weight": [1, 5], P + "reg_lambda": [1, 5]}, 8),
    "LightGBM": ("Boosting",
        lambda: LGBMRegressor(n_estimators=400, subsample_freq=1, random_state=RS, verbose=-1),
        {P + "learning_rate": [0.03, 0.05, 0.1], P + "num_leaves": [15, 31, 63],
         P + "min_child_samples": [10, 30, 60], P + "subsample": [0.7, 0.9],
         P + "colsample_bytree": [0.5, 0.8]}, 8),
    "CatBoost": ("Boosting",
        lambda: CatBoostRegressor(iterations=500, verbose=0, random_seed=RS, thread_count=-1,
                                  allow_writing_files=False),
        {P + "learning_rate": [0.03, 0.1], P + "depth": [4, 6, 8], P + "l2_leaf_reg": [1, 3, 10]}, 5),
}


def tune(name, cfg, train, feats, n_splits=3, n_iter=None, verbose=True):
    """RandomizedSearchCV with TimeSeriesSplit: every fold trains on the past and validates
    on the following block of weeks, so tuning never uses future data."""
    _, make, space, default_iter = CATALOGUE[name]
    search = RandomizedSearchCV(make_pipeline(make(), cfg), space, n_iter=n_iter or default_iter,
                                cv=TimeSeriesSplit(n_splits=n_splits), scoring="neg_mean_absolute_error",
                                random_state=RS, n_jobs=1, refit=False)
    t0 = time.time()
    search.fit(train[feats], train["target"])
    if verbose:
        print(f"{name}: best CV MAE {-search.best_score_:.3f} in {time.time() - t0:.0f}s -> "
              f"{ {k.replace(P, ''): v for k, v in search.best_params_.items()} }")
    return search.best_params_, -search.best_score_, time.time() - t0


# ---------------------------------------------------------------- early stopping
def _log(y, cfg):
    return np.log1p(y) if cfg["use_log_target"] else y


def early_stopping_rounds(name, params, cfg, train, val, feats, patience=50):
    """Pick the number of boosting rounds on the 2022 validation year.

    XGBoost / LightGBM / CatBoost: native early stopping with eval_set.
    AdaBoost / sklearn GradientBoosting: no eval_set, so we use staged_predict, i.e. the
    validation error after 1, 2, ... n trees, and take the best n (same idea, done manually).
    Returns the best number of rounds.
    """
    prep = F.make_preprocessor(cfg["numeric_features"]).fit(train[feats])
    Xtr, Xva = prep.transform(train[feats]), prep.transform(val[feats])
    ytr, yva = _log(train["target"], cfg), _log(val["target"], cfg)
    reg = CATALOGUE[name][1]()
    reg.set_params(**{k.replace(P, ""): v for k, v in params.items()})

    if name == "XGBoost":
        reg.set_params(n_estimators=3000, early_stopping_rounds=patience)
        reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return int(reg.best_iteration) + 1
    if name == "LightGBM":
        reg.set_params(n_estimators=3000)
        reg.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(patience, verbose=False)])
        return int(reg.best_iteration_)
    if name == "CatBoost":
        reg.set_params(iterations=3000, early_stopping_rounds=patience)
        reg.fit(Xtr, ytr, eval_set=(Xva, yva))
        return int(reg.get_best_iteration()) + 1
    # AdaBoost / GradientBoosting: staged predictions on validation
    reg.set_params(n_estimators=400 if name == "Gradient Boosting" else 300)
    reg.fit(Xtr, ytr)
    inv = np.expm1 if cfg["use_log_target"] else (lambda z: z)
    errs = [mean_absolute_error(val["target"], np.clip(inv(p), 0, None)) for p in reg.staged_predict(Xva)]
    return int(np.argmin(errs)) + 1


def build_final(name, params, cfg, n_rounds=None):
    """Untrained final pipeline with tuned params (+ early-stopped number of rounds)."""
    pipe = make_pipeline(CATALOGUE[name][1](), cfg)
    pipe.set_params(**params)
    if n_rounds is not None:
        key = "iterations" if name == "CatBoost" else "n_estimators"
        pipe.set_params(**{P + key: n_rounds})
    return pipe


def fit_timed(pipe, train, feats):
    t0 = time.time()
    pipe.fit(train[feats], train["target"])
    return pipe, time.time() - t0


# ---------------------------------------------------------------- stacking
class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking with *time-ordered* out-of-fold predictions.

    sklearn's StackingRegressor needs CV folds that cover every row (a partition);
    TimeSeriesSplit does not (the first block is never a test fold), so sklearn refuses it.
    This class does the same job in time order:
      1. For each TimeSeriesSplit fold, fit every base model on the past block and predict
         the next block -> honest out-of-fold (OOF) predictions.
      2. Fit the meta-learner (RidgeCV) on those OOF predictions (rows of the first block,
         which have no OOF prediction, are skipped). It learns how much to trust each model.
      3. Refit every base model on all training data for use at prediction time.
    The meta-learner works on log1p(predictions) to match the log-target used everywhere.
    Rows of X must be in chronological order.
    """

    def __init__(self, estimators, final_estimator=None, n_splits=4):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        y = np.asarray(y, float)
        oof = np.full((len(X), len(self.estimators)), np.nan)
        for tr_idx, te_idx in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            for j, (_, est) in enumerate(self.estimators):
                m = clone(est).fit(X.iloc[tr_idx], y[tr_idx])
                oof[te_idx, j] = m.predict(X.iloc[te_idx])
        have = ~np.isnan(oof).any(axis=1)
        self.oof_predictions_ = oof
        meta = self.final_estimator if self.final_estimator is not None else RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.final_estimator_ = clone(meta).fit(self._meta_X(oof[have]), np.log1p(y[have]))
        self.estimators_ = [(n, clone(est).fit(X, y)) for n, est in self.estimators]
        return self

    @staticmethod
    def _meta_X(base_preds):
        return np.log1p(np.clip(base_preds, 0, None))

    def base_predictions(self, X) -> pd.DataFrame:
        return pd.DataFrame({n: est.predict(X) for n, est in self.estimators_})

    def predict(self, X):
        return np.expm1(self.final_estimator_.predict(self._meta_X(self.base_predictions(X).to_numpy())))

    @property
    def meta_weights_(self) -> pd.Series:
        return pd.Series(self.final_estimator_.coef_, index=[n for n, _ in self.estimators])


# ---------------------------------------------------------------- persistence
def save_model(obj, filename: str) -> float:
    """joblib.dump with compression; refuses files that GitHub would reject. Returns size in MB."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(obj, path, compress=3)
    mb = path.stat().st_size / 1e6
    if mb > MAX_FILE_MB:
        path.unlink()
        raise ValueError(f"{filename} is {mb:.0f} MB (> {MAX_FILE_MB} MB); use fewer/shallower trees")
    print(f"saved models/{filename} ({mb:.1f} MB)")
    return mb


def load_model(filename: str):
    return joblib.load(MODELS_DIR / filename)


REGISTRY_JSON = MODELS_DIR / "registry.json"


def register(role: str, name: str, filename: str, **extra):
    """Record which saved file plays which role (e.g. 'bagging_best'); the dashboard reads this."""
    reg = json.loads(REGISTRY_JSON.read_text()) if REGISTRY_JSON.exists() else {}
    reg[role] = {"name": name, "file": filename, **extra}
    REGISTRY_JSON.write_text(json.dumps(reg, indent=2, default=str))


def load_registry() -> dict:
    return json.loads(REGISTRY_JSON.read_text())


class WeightedPredictor:
    """w * RF + (1 - w) * ANN, built from two saved pipelines (hybrid (a))."""

    def __init__(self, rf, ann, weight_rf):
        self.rf, self.ann, self.weight_rf = rf, ann, weight_rf

    def predict(self, X):
        return self.weight_rf * self.rf.predict(X) + (1 - self.weight_rf) * self.ann.predict(X)


class RFtoANNPredictor:
    """ANN that takes the saved RF's prediction as an extra input (hybrid (b))."""

    def __init__(self, rf, ann):
        self.rf, self.ann = rf, ann

    def predict(self, X):
        return self.ann.predict(X.assign(rf_log_pred=np.log1p(np.clip(self.rf.predict(X), 0, None))))


def load_predictor(model_name: str):
    """Load any registered model by its results-table name as an object with .predict(X)."""
    reg = load_registry()
    entry = next((e for e in reg.values() if e["name"] == model_name), None)
    if entry is None:
        raise KeyError(f"{model_name!r} was not saved to models/ (only family winners and components are kept)")
    obj = load_model(entry["file"])
    if isinstance(obj, dict) and "weight_rf" in obj:
        return WeightedPredictor(load_model(obj["rf_file"]), load_model(obj["ann_file"]), obj["weight_rf"])
    if "rf_file" in entry:
        return RFtoANNPredictor(load_model(entry["rf_file"]), obj)
    return obj


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def save_best_params(name, params, n_rounds=None, cv_mae=None):
    """Tuned hyper-parameters, early-stopping rounds and the time-series CV MAE (2007-2021 folds)."""
    store = json.loads(BEST_PARAMS_JSON.read_text()) if BEST_PARAMS_JSON.exists() else {}
    store[name] = {"params": dict(params), "n_rounds": n_rounds, "cv_mae": cv_mae}
    BEST_PARAMS_JSON.write_text(json.dumps(store, indent=2, default=str))


def load_best_params(name):
    entry = json.loads(BEST_PARAMS_JSON.read_text())[name]
    return entry["params"], entry["n_rounds"]
