# AI-assistance disclosure: This file was generated with the help of Claude Code (Anthropic) and reviewed, run and tested by B. G. M. Prabuddhi.
"""Hybrid models.

1. RF + SARIMA (time series, Zhang-style residual hybrid)
   SARIMA captures the linear autocorrelation + seasonality of one weekly series;
   a Random Forest then learns what SARIMA systematically misses (its residuals) from
   lagged cases, weather and calendar features. Final forecast = SARIMA + RF(residual).
   Everything is done on the log1p scale, then converted back with expm1.

2. RF + ANN (tabular, all districts)
   (a) weighted average  w * RF + (1 - w) * ANN, with w tuned on the 2022 validation year;
   (b) RF -> ANN: the RF's (out-of-fold) prediction is added as an extra input to the ANN.
"""
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import features as F
from .config import RANDOM_STATE, WEATHER_VARS
from .data import national_weekly

RS = RANDOM_STATE


# ====================================================================== series helpers
def series_frame(dengue: pd.DataFrame, weather: pd.DataFrame | None, which: str):
    """Return (dengue-like frame, weather-like frame) for one series so features.build_panel
    can be reused unchanged. which = 'National' (sum of all regions) or a district name."""
    if which != "National":
        d = dengue[dengue["district"] == which].copy()
        w = None if weather is None else weather[weather["district"] == which].copy()
        return d, w
    cal = (dengue.drop_duplicates("start_date")[["start_date", "end_date", "year", "week"]]
           .sort_values("start_date").set_index("start_date"))
    cal["cases"] = national_weekly(dengue)
    cal["missing_report"] = cal["cases"].isna()
    d = cal.reset_index().assign(district="National")
    w = None
    if weather is not None:   # national weather = average over the 26 region centres
        w = weather.groupby("start_date")[WEATHER_VARS].mean().reset_index().assign(district="National")
    return d, w


# ====================================================================== SARIMA
def select_sarima(y_train: np.ndarray, orders=None, seasonal_orders=None):
    """Small grid search; keeps the (p,d,q)x(P,D,Q,s) with the lowest AIC on the training series."""
    orders = orders or [(p, d, q) for p, d, q in itertools.product([0, 1, 2], [0, 1], [0, 1, 2])]
    seasonal_orders = seasonal_orders or [(0, 0, 0, 0)]
    rows = []
    for order, sorder in itertools.product(orders, seasonal_orders):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y_train, order=order, seasonal_order=sorder, trend="c" if order[1] == 0 else "n",
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
            rows.append({"order": order, "seasonal_order": sorder, "aic": res.aic})
        except Exception as exc:  # some combinations fail to converge
            rows.append({"order": order, "seasonal_order": sorder, "aic": np.inf, "error": str(exc)[:60]})
    grid = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    return grid


def fit_sarima(y_train, order, seasonal_order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y_train, order=order, seasonal_order=seasonal_order, trend="c" if order[1] == 0 else "n",
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)


def one_step_predictions(train_res, y_full: np.ndarray) -> np.ndarray:
    """Walk-forward one-step-ahead forecasts over the FULL series with parameters fixed from training.

    `apply` re-runs the Kalman filter on the whole series with the trained parameters, so the
    prediction for week i uses only observations before week i (exactly like refreshing the
    forecast every week as new data arrive; parameters are not re-estimated).
    Missing weeks (NaN) are handled naturally by the filter.
    """
    full = train_res.apply(y_full)
    return np.asarray(full.get_prediction().predicted_mean)


# ====================================================================== RF + SARIMA
class RFSarimaHybrid:
    """SARIMA on log1p(cases) + Random Forest on SARIMA's one-step residuals.

    fit(panel_train, sarima_pred_by_date): RF learns  residual = log1p(actual) - sarima_pred
    from the panel features plus the two most recent residuals (known at forecast time).
    """

    def __init__(self, num_features, n_estimators=300, min_samples_leaf=5, max_features=0.5):
        self.num_features = num_features
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, n_jobs=-1, random_state=RS)

    @staticmethod
    def add_residual_lags(panel, sarima_pred: pd.Series, log_cases: pd.Series):
        """resid_lag1 = residual of week t (origin), resid_lag2 = week t-1. Known at forecast time."""
        resid = (log_cases - sarima_pred)
        p = panel.copy()
        p["sarima_pred"] = p["target_date"].map(sarima_pred)
        p["resid_lag1"] = p["origin_date"].map(resid)
        p["resid_lag2"] = p["origin_date"].map(resid.shift(1))
        p["target_resid"] = np.log1p(p["target"]) - p["sarima_pred"]
        return p

    @property
    def inputs(self):
        return self.num_features + ["resid_lag1", "resid_lag2"]

    def fit(self, p_train):
        p = p_train.dropna(subset=["target_resid", "sarima_pred"])
        X = p[self.inputs].fillna(p[self.inputs].median())
        self.fill_ = p[self.inputs].median()
        self.rf.fit(X, p["target_resid"])
        return self

    def predict_log(self, p):
        X = p[self.inputs].fillna(self.fill_)
        return p["sarima_pred"].to_numpy() + self.rf.predict(X)

    def predict(self, p):
        return np.clip(np.expm1(self.predict_log(p)), 0, None)


# ====================================================================== RF + ANN
COUNT_FEATURES = F.CASE_LAG_FEATURES + F.ROLLING_FEATURES   # heavily right-skewed case counts


def make_mlp_preprocessor(num_features):
    """Neural-net preprocessing: impute -> log1p (count features only) -> standardise; one-hot district.

    Trees only care about the ORDER of values, so raw counts are fine for them. A neural net
    multiplies inputs by weights: with raw counts (skewness ~10) an outbreak week is many standard
    deviations outside the training range, the network extrapolates linearly, and after expm1 the
    forecast explodes (our first attempt had CV MAE > 30,000). log1p puts counts on the same scale
    as the log1p target, so the ANN learns "log cases next week ~ log cases this week".
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    counts = [c for c in num_features if c in COUNT_FEATURES]
    others = [c for c in num_features if c not in COUNT_FEATURES]
    return ColumnTransformer([
        ("counts", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                             ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), counts),
        ("other", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), others),
        ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), F.CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def make_mlp_pipeline(cfg, extra_features=(), **mlp_kwargs):
    """MLP on log-scaled, standardised inputs with the same log1p target as every other model."""
    from sklearn.pipeline import Pipeline
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                       random_state=RS, **mlp_kwargs)
    return Pipeline([("prep", make_mlp_preprocessor(cfg["numeric_features"] + list(extra_features))),
                     ("model", F.with_target_transform(mlp, cfg["use_log_target"]))])


MLP_SPACE = {"model__regressor__hidden_layer_sizes": [(32,), (64, 32), (128, 64), (64, 32, 16)],
             "model__regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
             "model__regressor__learning_rate_init": [3e-4, 1e-3, 3e-3]}


def tune_mlp(cfg, train, feats, n_iter=6, n_splits=3):
    search = RandomizedSearchCV(make_mlp_pipeline(cfg), MLP_SPACE, n_iter=n_iter,
                                cv=TimeSeriesSplit(n_splits=n_splits), scoring="neg_mean_absolute_error",
                                random_state=RS, n_jobs=1, refit=False)
    t0 = time.time()
    search.fit(train[feats], train["target"])
    return search.best_params_, -search.best_score_, time.time() - t0


def best_weight(y_val, rf_val, ann_val, grid=np.linspace(0, 1, 21)):
    """w in [0,1] minimising validation MAE of  w*RF + (1-w)*ANN."""
    maes = np.array([np.mean(np.abs(y_val - (w * rf_val + (1 - w) * ann_val))) for w in grid])
    return float(grid[maes.argmin()]), pd.Series(maes, index=grid.round(2))


class RFtoANNRegressor(BaseEstimator, RegressorMixin):
    """ANN that receives the Random Forest's prediction as an extra input feature.

    If the ANN were trained on the RF's *in-sample* predictions it would learn to trust them
    blindly (a forest fits its own training data almost perfectly). So, as in stacking, the RF
    feature for training rows comes from time-ordered out-of-fold predictions; rows in the first
    block (no out-of-fold prediction) are dropped. Rows of X must be in chronological order.
    """

    def __init__(self, rf_pipeline, ann_pipeline, n_splits=4):
        self.rf_pipeline = rf_pipeline
        self.ann_pipeline = ann_pipeline
        self.n_splits = n_splits

    def fit(self, X, y):
        y = np.asarray(y, float)
        oof = np.full(len(X), np.nan)
        for tr_idx, te_idx in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            m = clone(self.rf_pipeline).fit(X.iloc[tr_idx], y[tr_idx])
            oof[te_idx] = m.predict(X.iloc[te_idx])
        have = ~np.isnan(oof)
        self.rf_ = clone(self.rf_pipeline).fit(X, y)
        Xa = X.assign(rf_log_pred=np.log1p(np.clip(oof, 0, None)))
        self.ann_ = clone(self.ann_pipeline).fit(Xa[have], y[have])
        return self

    def predict(self, X):
        Xa = X.assign(rf_log_pred=np.log1p(np.clip(self.rf_.predict(X), 0, None)))
        return self.ann_.predict(Xa)
