"""
Data cleaning recommendations for DataInsight AI.

Detects fixable issues (via modules/eda.py's profiling helpers) and turns
each into a concrete, named, reversible transform the user can review and
opt into — never applied automatically. No Streamlit imports here, so this
module can be tested and reused on its own; app.py owns the approval UI,
the transformation history, and keeping the original dataframe untouched.
"""

from __future__ import annotations

import pandas as pd

from . import eda


def _drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _fill_missing_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].fillna(df[col].median())
    return df


def _fill_missing_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    mode = df[col].mode(dropna=True)
    if len(mode):
        df[col] = df[col].fillna(mode.iloc[0])
    return df


def _drop_missing_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.dropna(subset=[col]).reset_index(drop=True)


def _cap_outliers(df: pd.DataFrame, col: str, low: float, high: float) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].clip(low, high)
    return df


def _normalize_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Canonicalize inconsistent text values: trim whitespace, then map
    each case-insensitive variant to its most common original spelling
    (so "USA" beats " usa" if "USA" appears more often), rather than
    forcing everything to lowercase."""
    df = df.copy()
    series = df[col].astype(str)
    key = series.str.strip().str.lower()
    mode_by_key = series.groupby(key).agg(lambda s: s.value_counts().idxmax())
    df[col] = key.map(mode_by_key)
    return df


def build_recommendations(df: pd.DataFrame, col_types: dict) -> list:
    """Return a list of {id, title, rationale, columns, default_checked,
    apply} recommendations. `apply` takes the current working dataframe
    and returns a new one — never mutates in place."""
    recs: list = []

    if int(df.duplicated().sum()) > 0:
        n = int(df.duplicated().sum())
        recs.append({
            "id": "drop_duplicates",
            "title": "Remove duplicate rows",
            "rationale": f"{n} row(s) are exact duplicates of another row.",
            "columns": [],
            "default_checked": True,
            "apply": _drop_duplicate_rows,
        })

    missing = eda.missing_value_table(df)
    for _, row in missing[missing["missing_count"] > 0].iterrows():
        col = row["column"]
        if col in col_types.get("numeric", []):
            recs.append({
                "id": f"fill_median_{col}",
                "title": f"Fill missing {col} with the median",
                "rationale": f"{int(row['missing_count'])} missing value(s) ({row['missing_pct']}%).",
                "columns": [col],
                "default_checked": False,
                "apply": lambda d, c=col: _fill_missing_numeric(d, c),
            })
        elif col in col_types.get("categorical", []):
            recs.append({
                "id": f"fill_mode_{col}",
                "title": f"Fill missing {col} with the most common value",
                "rationale": f"{int(row['missing_count'])} missing value(s) ({row['missing_pct']}%).",
                "columns": [col],
                "default_checked": False,
                "apply": lambda d, c=col: _fill_missing_categorical(d, c),
            })
        elif col in col_types.get("datetime", []) and row["missing_pct"] <= 20:
            recs.append({
                "id": f"drop_missing_{col}",
                "title": f"Drop rows with an unparseable/missing {col}",
                "rationale": f"{int(row['missing_count'])} row(s) ({row['missing_pct']}%) have no usable date.",
                "columns": [col],
                "default_checked": False,
                "apply": lambda d, c=col: _drop_missing_rows(d, c),
            })

    outliers = eda.outlier_summary(df, col_types.get("numeric", []))
    for _, row in outliers.iterrows():
        col, low, high = row["column"], row["lower_bound"], row["upper_bound"]
        recs.append({
            "id": f"cap_outliers_{col}",
            "title": f"Cap outliers in {col} to the typical range",
            "rationale": f"{int(row['n_outliers'])} value(s) fall outside [{low}, {high}] ({row['pct_outliers']}%).",
            "columns": [col],
            "default_checked": False,
            "apply": lambda d, c=col, lo=low, hi=high: _cap_outliers(d, c, lo, hi),
        })

    for row in eda.inconsistent_text_columns(df, col_types.get("categorical", [])):
        col = row["column"]
        recs.append({
            "id": f"normalize_text_{col}",
            "title": f"Normalize inconsistent text in {col}",
            "rationale": f"{row['raw_unique']} raw spellings collapse to {row['collapsed_unique']} once whitespace/case differences are ignored.",
            "columns": [col],
            "default_checked": True,
            "apply": lambda d, c=col: _normalize_text(d, c),
        })

    return recs
