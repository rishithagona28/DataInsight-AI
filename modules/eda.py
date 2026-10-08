"""
EDA (exploratory data analysis) helpers for Ana.

All functions take a pandas DataFrame and return plain data structures
(DataFrames, dicts) that app.py renders with Streamlit. No Streamlit
imports here, so this module can be tested and reused on its own.
"""

from __future__ import annotations

import re

import pandas as pd


def basic_info(df: pd.DataFrame) -> dict:
    """High-level shape/quality facts about the dataset."""
    n_rows, n_cols = df.shape
    n_duplicates = int(df.duplicated().sum())
    total_cells = n_rows * n_cols
    missing_cells = int(df.isna().sum().sum())
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_duplicates": n_duplicates,
        "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / total_cells, 2) if total_cells else 0.0,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
    }


def column_types(df: pd.DataFrame) -> dict:
    """Bucket columns into numeric / categorical / datetime / boolean /
    high-cardinality (likely IDs or free text, excluded from most charts)."""
    numeric_cols: list = []
    datetime_cols: list = []
    boolean_cols: list = []
    categorical_cols: list = []
    high_cardinality_cols: list = []

    n_rows = max(len(df), 1)

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        # Try to detect date-like strings (common when pandas reads a
        # date column from CSV as plain text). Covers both legacy `object`
        # columns and pandas's newer default `str` dtype.
        if pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(col)
                continue

        n_unique = series.nunique(dropna=True)
        if n_unique / n_rows > 0.9 and n_unique > 50:
            high_cardinality_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "boolean": boolean_cols,
        "high_cardinality": high_cardinality_cols,
    }


def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2) if len(df) else missing * 0.0
    out = pd.DataFrame({"column": df.columns, "missing_count": missing.values, "missing_pct": pct.values})
    return out.sort_values("missing_count", ascending=False).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    summary = df[numeric_cols].describe().T
    summary["skew"] = df[numeric_cols].skew(numeric_only=True)
    return summary.round(3)


def categorical_summary(df: pd.DataFrame, categorical_cols: list, top_n: int = 5) -> dict:
    out = {}
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False).head(top_n)
        out[col] = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
    return out


def correlation_matrix(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(numeric_only=True).round(3)


def to_datetime_safe(series: pd.Series) -> pd.Series:
    """Parse a column into datetimes whether it's already datetime or text."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


# ---------------------------------------------------------------------------
# Data profiling: cardinality, outliers, column-role suggestions, and a
# combined data-quality report used by the Overview tab, the cleaning
# recommendations (modules/cleaning.py), and AI insights/report context.
# ---------------------------------------------------------------------------

_ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|[_-]key$|^key$|uuid|guid)", re.I)
_TARGET_NAME_RE = re.compile(r"^(target|label|class|outcome|churn|response|result|status|y)$", re.I)


def cardinality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Unique-value counts and % unique for every column."""
    n_rows = max(len(df), 1)
    rows = []
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "pct_unique": round(100 * n_unique / n_rows, 2),
        })
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """IQR-based outlier counts per numeric column (values outside
    Q1 - 1.5*IQR .. Q3 + 1.5*IQR)."""
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < low) | (series > high)).sum())
        if n_outliers:
            rows.append({
                "column": col,
                "n_outliers": n_outliers,
                "pct_outliers": round(100 * n_outliers / len(series), 2),
                "lower_bound": round(float(low), 3),
                "upper_bound": round(float(high), 3),
            })
    return pd.DataFrame(rows)


def suggest_id_columns(df: pd.DataFrame, col_types: dict) -> list:
    """Columns likely to be identifiers: already flagged high-cardinality,
    or named like one (id, _id, key, uuid...). A suggestion, not binding."""
    by_cardinality = set(col_types.get("high_cardinality", []))
    by_name = {c for c in df.columns if _ID_NAME_RE.search(str(c))}
    return [c for c in df.columns if c in by_cardinality or c in by_name]


def suggest_target_columns(df: pd.DataFrame, col_types: dict) -> list:
    """Columns whose name suggests they're a prediction target. Deliberately
    name-based only (high precision) — silent when nothing matches rather
    than guessing from shape alone."""
    return [c for c in df.columns if _TARGET_NAME_RE.search(str(c))]


def _severity(pct: float) -> str:
    if pct >= 20:
        return "high"
    if pct >= 5:
        return "medium"
    return "low"


def data_quality_report(df: pd.DataFrame, col_types: dict) -> list:
    """Structured list of flagged data-quality issues:
    {issue, severity, columns, detail}. Combines missing values,
    duplicates, outliers, ID-like columns, and inconsistent category
    text into one shared source of truth for the Overview tab, the
    cleaning recommendations, and AI insights/report context."""
    issues: list = []
    n_rows = len(df)

    if n_rows:
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            pct = round(100 * n_dupes / n_rows, 2)
            issues.append({
                "issue": "Duplicate rows",
                "severity": _severity(pct),
                "columns": [],
                "detail": f"{n_dupes} duplicate row(s) ({pct}% of the dataset).",
            })

    missing = missing_value_table(df)
    for _, row in missing.iterrows():
        if row["missing_count"] > 0:
            issues.append({
                "issue": f"Missing values in {row['column']}",
                "severity": _severity(row["missing_pct"]),
                "columns": [row["column"]],
                "detail": f"{int(row['missing_count'])} missing ({row['missing_pct']}%).",
            })

    outliers = outlier_summary(df, col_types.get("numeric", []))
    for _, row in outliers.iterrows():
        issues.append({
            "issue": f"Outliers in {row['column']}",
            "severity": _severity(row["pct_outliers"]),
            "columns": [row["column"]],
            "detail": f"{int(row['n_outliers'])} value(s) outside [{row['lower_bound']}, {row['upper_bound']}] ({row['pct_outliers']}%).",
        })

    for col in suggest_id_columns(df, col_types):
        issues.append({
            "issue": f"{col} looks like an identifier",
            "severity": "low",
            "columns": [col],
            "detail": "High cardinality or an ID-like name — consider excluding it from charts and models.",
        })

    for row in inconsistent_text_columns(df, col_types.get("categorical", [])):
        pct = 100 * (row["raw_unique"] - row["collapsed_unique"]) / row["raw_unique"]
        issues.append({
            "issue": f"Inconsistent text values in {row['column']}",
            "severity": _severity(pct),
            "columns": [row["column"]],
            "detail": f"{row['raw_unique']} distinct raw values collapse to {row['collapsed_unique']} after trimming whitespace/case — likely the same category written differently.",
        })

    return issues


def inconsistent_text_columns(df: pd.DataFrame, categorical_cols: list) -> list:
    """Categorical columns where trimming whitespace/case collapses the
    number of distinct values — a sign the same category is written
    inconsistently (e.g. "USA" vs " usa"). Shared by the quality report
    above and modules/cleaning.py's normalization recommendation."""
    rows = []
    for col in categorical_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        raw = series.nunique()
        collapsed = series.str.strip().str.lower().nunique()
        if collapsed < raw:
            rows.append({"column": col, "raw_unique": raw, "collapsed_unique": collapsed})
    return rows
