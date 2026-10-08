"""
Prediction module for DataInsight AI: forecast a time series forward, or train a
quick model to predict any column from the others.

Everything here is intentionally simple (a handful of sklearn/statsmodels
calls) — this gives a fast, honest first read on predictability, not a
replacement for real feature engineering or model selection.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_STATE = 42
# Training/evaluating (fit + cross-validation + permutation importance) on
# more rows than this gets slow without meaningfully improving a default
# model's fit — sample down before splitting rather than trying to scale
# to arbitrary dataset sizes.
MAX_TRAINING_ROWS = 50_000
# One-hot encoding a column with more distinct values than this explodes
# both memory and training time for little predictive benefit — such a
# column is excluded from training automatically (see prepare_training_data).
MAX_ONEHOT_CARDINALITY = 100

def _forest_size(n_rows: int | None) -> int:
    """Fewer trees for bigger training sets — past a few thousand rows per
    tree, more trees mostly add training time, not stability."""
    if n_rows and n_rows > 20_000:
        return 100
    return 300


def _boosting_size(n_rows: int | None) -> int:
    """GradientBoosting is inherently sequential (no n_jobs), so its
    estimator count matters even more at scale than the forest's."""
    if n_rows and n_rows > 20_000:
        return 50
    return 100  # sklearn's own default


# One entry per selectable model, per task type. Factories (not instances)
# take the training-set size so estimator counts can scale down for large
# data — every call gets a fresh, unfitted estimator.
MODEL_REGISTRY = {
    "classification": {
        "Logistic Regression": lambda n_rows=None: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": lambda n_rows=None: DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": lambda n_rows=None: RandomForestClassifier(n_estimators=_forest_size(n_rows), random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": lambda n_rows=None: GradientBoostingClassifier(n_estimators=_boosting_size(n_rows), random_state=RANDOM_STATE),
    },
    "regression": {
        "Linear Regression": lambda n_rows=None: LinearRegression(),
        "Decision Tree": lambda n_rows=None: DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": lambda n_rows=None: RandomForestRegressor(n_estimators=_forest_size(n_rows), random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": lambda n_rows=None: GradientBoostingRegressor(n_estimators=_boosting_size(n_rows), random_state=RANDOM_STATE),
    },
}


# ---------------------------------------------------------------------------
# Time-series forecasting
# ---------------------------------------------------------------------------

def can_forecast(df: pd.DataFrame, date_col: str, value_col: str, min_points: int = 5) -> tuple:
    n = df[[date_col, value_col]].dropna().shape[0]
    if n < min_points:
        return False, f"Need at least {min_points} non-missing (date, value) pairs — found {n}."
    return True, ""


# Roughly how many of each cadence fit in a month — used to translate a
# "next N months" choice into a period count for whatever cadence the data
# is actually in. Approximate by design: this is meant to feel simple, not
# to be calendar-exact.
_PERIODS_PER_MONTH = {
    "H": 730, "T": 43_800, "MIN": 43_800, "S": 2_628_000,
    "D": 30, "B": 21,
    "W": 4.345,
    "M": 1, "MS": 1,
    "Q": 1 / 3, "QS": 1 / 3,
    "Y": 1 / 12, "A": 1 / 12, "YS": 1 / 12, "AS": 1 / 12,
}


def periods_for_horizon(freq: str | None, horizon_months: float) -> int:
    """How many periods of `freq` fit in `horizon_months` months. Falls
    back to treating the cadence as monthly when frequency couldn't be
    detected — a reasonable default for "next N months" phrasing."""
    base = 1.0
    if freq:
        key = freq.upper().lstrip("0123456789").split("-")[0]
        base = _PERIODS_PER_MONTH.get(key, 1.0)
    return max(1, round(horizon_months * base))


def forecast_time_series(df: pd.DataFrame, date_col: str, value_col: str, periods: int | None = None, horizon_months: float | None = None) -> dict:
    """Fit a simple trend model (Holt's exponential smoothing) and project
    forward, either an explicit number of `periods` (custom mode) or a
    `horizon_months` translated into periods via the data's detected
    cadence. Returns actual + forecast series, a rough (not statistically
    rigorous) uncertainty band, and the % change from the last actual
    value to the end of the forecast."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if periods is None and horizon_months is None:
        raise ValueError("Provide either periods or horizon_months.")

    ok, reason = can_forecast(df, date_col, value_col)
    if not ok:
        raise ValueError(reason)

    series, freq, agg, period_name, dropped = build_period_series(df, date_col, value_col)
    if len(series) < 5:
        raise ValueError(f"Only {len(series)} complete {period_name} periods of data; need at least 5 to forecast.")
    if periods is None:
        periods = periods_for_horizon(freq, horizon_months)
    if freq is None:
        diffs = series.index.to_series().diff().dropna()
        median_gap = diffs.median() if len(diffs) else pd.Timedelta(days=1)
        future_index = [series.index[-1] + median_gap * (i + 1) for i in range(periods)]
    else:
        future_index = pd.date_range(series.index[-1], periods=periods + 1, freq=freq)[1:]

    seasonal_periods = None
    seasonal = None
    if freq and len(series) >= 24 and freq.startswith(("M", "MS")):
        seasonal, seasonal_periods = "add", 12
    elif freq and len(series) >= 14 and freq.startswith("D"):
        seasonal, seasonal_periods = "add", 7

    model = ExponentialSmoothing(
        series.values,
        trend="add",
        damped_trend=True,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    ).fit(optimized=True)

    forecast_values = model.forecast(periods)
    residual_std = float(np.std(model.resid)) if len(model.resid) else float(series.std())
    band = 1.28 * residual_std  # rough ~80% band around a simple trend fit

    actual_df = pd.DataFrame({"date": series.index, "value": series.values, "kind": "actual"})
    forecast_df = pd.DataFrame({
        "date": future_index,
        "value": forecast_values,
        "kind": "forecast",
        "low": forecast_values - band,
        "high": forecast_values + band,
    })

    # Compare like with like: the forecast horizon vs the same number of most
    # recent actual periods (e.g. next 3 months vs last 3 months). Comparing
    # against one single last value is noisy and misleading.
    recent = series.values[-periods:] if len(series) >= periods else series.values
    recent_avg = float(np.mean(recent))
    forecast_avg = float(np.mean(forecast_values))
    growth_pct = round(100 * (forecast_avg - recent_avg) / abs(recent_avg), 1) if recent_avg else None

    return {
        "actual": actual_df,
        "forecast": forecast_df,
        "freq_detected": f"{period_name} {'totals' if agg == 'sum' else 'averages'}" if agg else (freq or "irregular (using median gap)"),
        "seasonal": seasonal or "none",
        "periods": periods,
        "growth_pct": growth_pct,
        "aggregation": agg,                       # "sum", "mean", or None (data already one value per period)
        "period_name": period_name,               # "month", "week", "day", ...
        "forecast_total": float(np.sum(forecast_values)),
        "recent_total": float(np.sum(recent)),
        "dropped_partial_periods": dropped,
    }


# Columns whose values are a *level* (averaged over a period) rather than an
# *amount* (added up over a period). Revenue per month is a sum; price per month is an average.
_LEVEL_NAME = re.compile(
    r"price|rate|ratio|avg|average|mean|median|score|rating|temp|percent|pct|%|margin|age|index|balance|level|close|open|high|low",
    re.I,
)


def build_period_series(df: pd.DataFrame, date_col: str, value_col: str):
    """Turn raw rows into one value per period, the way an analyst would before forecasting.

    - Data that is already one row per regular period (e.g. a monthly report) is used as is.
    - Transaction-style data (many rows per day, or irregular dates) is grouped into
      months (>= 6 months of data), weeks (>= 8 weeks) or days. Amounts like revenue or
      units are SUMMED per period; levels like price or rating are AVERAGED.
    - Incomplete first/last periods (e.g. a month that has only 10 days of data) are
      dropped, because a half-recorded month looks like a sales collapse to a model.

    Returns (series, freq, aggregation, period_name, dropped_partial_periods).
    """
    data = df[[date_col, value_col]].dropna()
    dates = pd.to_datetime(data[date_col], errors="coerce")
    data = data.assign(**{date_col: dates}).dropna().sort_values(date_col)

    unique_dates = pd.DatetimeIndex(data[date_col].drop_duplicates())
    one_row_per_date = len(unique_dates) == len(data)
    regular_freq = pd.infer_freq(unique_dates) if len(unique_dates) >= 3 else None
    if one_row_per_date and regular_freq:
        series = data.set_index(date_col)[value_col]
        series.index = pd.DatetimeIndex(series.index)
        name = {"M": "month", "MS": "month", "W": "week", "D": "day", "B": "business day", "Q": "quarter", "QS": "quarter"}.get(
            regular_freq.upper().lstrip("0123456789").split("-")[0], "period")
        return series, regular_freq, None, name, 0

    agg = "mean" if _LEVEL_NAME.search(str(value_col)) else "sum"
    span_days = (data[date_col].max() - data[date_col].min()).days
    if span_days >= 180:
        rule, name = "MS", "month"
    elif span_days >= 56:
        rule, name = "W-SUN", "week"
    else:
        rule, name = "D", "day"

    series = data.set_index(date_col)[value_col].resample(rule).agg(agg)
    if agg == "mean":
        series = series.interpolate(limit_direction="both")

    dropped = 0
    first, last = data[date_col].min(), data[date_col].max()
    if rule == "MS":
        if first.day > 1 and len(series) > 6:
            series, dropped = series.iloc[1:], dropped + 1
        if not last.is_month_end and len(series) > 6:
            series, dropped = series.iloc[:-1], dropped + 1
    elif rule == "W-SUN":
        if first.dayofweek != 0 and len(series) > 6:   # week didn't start on Monday
            series, dropped = series.iloc[1:], dropped + 1
        if last.dayofweek != 6 and len(series) > 6:    # week didn't end on Sunday
            series, dropped = series.iloc[:-1], dropped + 1
    return series, rule, agg, name, dropped


# ---------------------------------------------------------------------------
# Predict any column from the others
# ---------------------------------------------------------------------------

def infer_task_type(series: pd.Series, max_classes: int = 15) -> str:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if series.nunique(dropna=True) > max_classes:
            return "regression"
    return "classification"


def _build_pipeline(numeric_features: list, categorical_features: list, estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def prepare_training_data(df: pd.DataFrame, target_col: str, id_like_cols: list | None = None, test_size: float = 0.2) -> dict:
    """Shared split/typing step used by every model in a comparison, so
    each model is trained and tested on exactly the same rows."""
    id_like_cols = id_like_cols or []
    data = df.dropna(subset=[target_col]).copy()
    feature_cols = [c for c in data.columns if c != target_col and c not in id_like_cols]
    if not feature_cols:
        raise ValueError("No usable feature columns are left after excluding the target and ID-like columns.")

    task = infer_task_type(data[target_col])

    sampled_from = None
    if len(data) > MAX_TRAINING_ROWS:
        sampled_from = len(data)
        if task == "classification":
            # Sample within each class so rare classes aren't wiped out by
            # a plain random sample.
            frac = MAX_TRAINING_ROWS / sampled_from
            data = data.groupby(target_col, group_keys=False).apply(
                lambda g: g.sample(max(1, round(len(g) * frac)), random_state=RANDOM_STATE)
            )
        else:
            data = data.sample(MAX_TRAINING_ROWS, random_state=RANDOM_STATE)

    X = data[feature_cols]
    y = data[target_col]

    # Datetime columns aren't usable as a bare categorical predictor (and a
    # column with many distinct raw timestamps would one-hot-encode into
    # thousands of columns), and any other very-high-cardinality text
    # column (a free-text field, an ID the user didn't think to exclude)
    # is the same trap — cap what one-hot encoding is allowed to touch
    # rather than let a single bad column blow up training time/memory.
    datetime_features = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(X[c])]
    numeric_features = [c for c in feature_cols if c not in datetime_features and pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    other_features = [c for c in feature_cols if c not in datetime_features and c not in numeric_features]
    categorical_features = [c for c in other_features if X[c].nunique(dropna=True) <= MAX_ONEHOT_CARDINALITY]
    excluded_features = datetime_features + [c for c in other_features if c not in categorical_features]
    feature_cols = numeric_features + categorical_features
    if not feature_cols:
        raise ValueError("No usable feature columns are left after excluding the target, ID-like, datetime, and very-high-cardinality columns.")
    X = X[feature_cols]

    stratify = y if (task == "classification" and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    return {
        "task": task,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "numeric_features": numeric_features, "categorical_features": categorical_features,
        "feature_cols": feature_cols,
        "excluded_features": excluded_features,
        "sampled_from": sampled_from,
        # Labels from the *full* (post-sampling) column, not just y_train —
        # a small test split can otherwise contain a class train never
        # saw, which breaks confusion_matrix/classification_report with an
        # explicit (too-narrow) labels list.
        "all_labels": sorted(y.dropna().unique().tolist(), key=str) if task == "classification" else None,
    }


def evaluate_classification(y_test, y_pred, y_proba=None, labels: list | None = None) -> dict:
    """Accuracy/precision/recall/F1/confusion matrix always; ROC-AUC only
    when class probabilities are available and computable for this label
    set (binary directly, multiclass via one-vs-rest)."""
    labels = labels if labels is not None else sorted(pd.Series(y_test).dropna().unique().tolist(), key=str)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision (weighted)": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
        "Recall (weighted)": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3),
        "F1 (weighted)": round(f1_score(y_test, y_pred, average="weighted"), 3),
    }
    if y_proba is not None:
        try:
            if len(labels) == 2:
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 3)
            else:
                metrics["ROC-AUC (ovr)"] = round(roc_auc_score(y_test, y_proba, multi_class="ovr", labels=labels), 3)
        except ValueError:
            pass  # not computable for this label distribution — omit rather than guess

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "metrics": metrics,
        "confusion_matrix": pd.DataFrame(cm, index=labels, columns=labels),
        "classification_report": pd.DataFrame(report).T.round(3),
    }


def evaluate_regression(y_test, y_pred, n_features: int) -> dict:
    """MAE/MSE/RMSE/R² always; adjusted R² only when there are enough test
    rows relative to feature count for it to mean anything."""
    n = len(y_test)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "R2": round(r2, 3),
        "MAE": round(mean_absolute_error(y_test, y_pred), 3),
        "MSE": round(mean_squared_error(y_test, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
    }
    if n > n_features + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        metrics["Adjusted R2"] = round(adj_r2, 3)
    return {"metrics": metrics}


def explain_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on the *original* (pre-encoding) columns —
    works identically for every model type in MODEL_REGISTRY, unlike
    tree-only `feature_importances_` or linear-only `coef_`, so one
    explainability path covers every model in a comparison. `n_repeats` is
    lowered by the caller for large test sets, where each repeat is
    already expensive and doesn't add much precision."""
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = pd.DataFrame({
        "source_column": X_test.columns,
        "importance": result.importances_mean,
        "std": result.importances_std,
    })
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def train_and_evaluate_one(prep: dict, model_name: str, cv_folds: int = 3) -> dict:
    """Train and evaluate a single model against an already-built
    `prepare_training_data(...)` split. Split out from `compare_models` so
    a caller (the Predict tab) can run this one model at a time and show
    real incremental progress instead of one long silent block."""
    task = prep["task"]
    registry = MODEL_REGISTRY[task]
    entry = {"model": model_name, "task": task}
    if model_name not in registry:
        entry["error"] = f"Unknown {task} model '{model_name}'."
        return entry

    try:
        estimator = registry[model_name](n_rows=len(prep["X_train"]))
        pipeline = _build_pipeline(prep["numeric_features"], prep["categorical_features"], estimator)

        start = time.time()
        pipeline.fit(prep["X_train"], prep["y_train"])
        entry["train_seconds"] = round(time.time() - start, 3)

        y_pred = pipeline.predict(prep["X_test"])

        if task == "classification":
            y_proba = pipeline.predict_proba(prep["X_test"]) if hasattr(pipeline.named_steps["model"], "predict_proba") else None
            evaluation = evaluate_classification(prep["y_test"], y_pred, y_proba, labels=prep["all_labels"])
            scoring = "accuracy"
        else:
            evaluation = evaluate_regression(prep["y_test"], y_pred, n_features=len(prep["feature_cols"]))
            scoring = "r2"
        entry.update(evaluation)

        try:
            cv_scores = cross_val_score(clone(pipeline), prep["X_train"], prep["y_train"], cv=min(cv_folds, len(prep["y_train"])), scoring=scoring)
            entry["cv_mean"] = round(float(cv_scores.mean()), 3)
            entry["cv_std"] = round(float(cv_scores.std()), 3)
        except Exception:
            entry["cv_mean"] = entry["cv_std"] = None  # too few samples/classes for CV — skip rather than crash

        n_repeats = 3 if len(prep["X_train"]) > 10_000 else 5
        entry["importance"] = explain_model(pipeline, prep["X_test"], prep["y_test"], n_repeats=n_repeats)
        entry["y_test"] = prep["y_test"].reset_index(drop=True)
        entry["y_pred"] = pd.Series(y_pred)
    except Exception as exc:
        entry["error"] = str(exc)
    return entry


def summarize_comparison(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r["model"], "train_seconds": r.get("train_seconds"), "cv_mean": r.get("cv_mean"), "cv_std": r.get("cv_std")}
        row.update(r.get("metrics", {}))
        if "error" in r:
            row["error"] = r["error"]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame, target_col: str, model_names: list, id_like_cols: list | None = None, test_size: float = 0.2, cv_folds: int = 3) -> dict:
    """Train and evaluate each requested model on the *same* train/test
    split, so results are directly comparable. Returns per-model results
    plus a flat summary table — no synthetic overall score, raw metrics
    only, so trade-offs stay visible. A thin convenience wrapper around
    `prepare_training_data` + `train_and_evaluate_one` for callers (tests,
    the report) that just want the finished result in one call — the
    Predict tab calls those two directly instead, for per-model progress."""
    prep = prepare_training_data(df, target_col, id_like_cols, test_size)
    results = [train_and_evaluate_one(prep, name, cv_folds=cv_folds) for name in model_names]

    return {
        "task": prep["task"],
        "n_train": len(prep["X_train"]),
        "n_test": len(prep["X_test"]),
        "feature_cols": prep["feature_cols"],
        "sampled_from": prep.get("sampled_from"),
        "results": results,
        "summary": summarize_comparison(results),
    }
