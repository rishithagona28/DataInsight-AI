"""
AI-generated insights: turn EDA summary statistics (not raw rows) into a
short natural-language analysis using an OpenAI-compatible chat
completions endpoint (Gemini by default — see app.py's get_llm_client).
"""

from __future__ import annotations

import os

import pandas as pd

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

SYSTEM_PROMPT = (
    "You are a careful data analyst. You are given summary statistics "
    "about a dataset (never the raw rows) — shape, column types, missing "
    "data, numeric summary stats, top categorical values, and correlations. "
    "Write a concise, plain-language analysis for a non-technical reader: "
    "1) what the dataset appears to contain, 2) the most notable patterns "
    "or relationships, 3) data quality issues worth fixing (missing data, "
    "outliers, duplicates, suspicious values), and 4) 2-3 concrete next "
    "questions worth investigating. Use short paragraphs or a light list. "
    "Only state what the statistics actually support — do not invent facts "
    "about the data you were not given."
)


def build_summary_text(df: pd.DataFrame, col_types: dict, quality_issues: list | None = None, max_chars: int = 6000) -> str:
    """A compact, human-readable digest of the dataset for the model —
    never the raw data itself. `quality_issues` is the structured list
    from eda.data_quality_report, already computed programmatically —
    passing it in means the model interprets verified facts instead of
    re-deriving (or guessing at) them."""
    lines: list = []
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")

    if quality_issues:
        lines.append("\nData quality issues already detected programmatically:")
        for issue in quality_issues[:15]:
            lines.append(f"- [{issue['severity']}] {issue['issue']}: {issue['detail']}")

    lines.append("\nColumns and types:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        pct_missing = round(100 * n_missing / len(df), 1) if len(df) else 0.0
        lines.append(f"- {col} ({dtype}): {n_missing} missing ({pct_missing}%)")

    numeric = col_types.get("numeric", [])
    if numeric:
        lines.append("\nNumeric summary (mean, std, min, median, max):")
        desc = df[numeric].describe().T.round(3)
        for col, row in desc.iterrows():
            lines.append(
                f"- {col}: mean={row['mean']}, std={row['std']}, min={row['min']}, "
                f"median={row['50%']}, max={row['max']}"
            )

    categorical = col_types.get("categorical", [])
    if categorical:
        lines.append("\nTop categorical values:")
        for col in categorical[:10]:
            vc = df[col].value_counts(dropna=True).head(5)
            top = ", ".join(f"{idx} ({cnt})" for idx, cnt in vc.items())
            lines.append(f"- {col}: {top}")

    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True)
        cols = corr.columns.tolist()
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append((cols[i], cols[j], v))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        if pairs:
            lines.append("\nStrongest correlations:")
            for a, b, v in pairs[:8]:
                lines.append(f"- {a} vs {b}: {round(v, 3)}")

    text = "\n".join(lines)
    return text[:max_chars]


def generate_insights(client, df: pd.DataFrame, col_types: dict, quality_issues: list | None = None, model: str | None = None) -> str:
    summary = build_summary_text(df, col_types, quality_issues=quality_issues)

    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the dataset summary:\n\n{summary}"},
        ],
    )
    return response.choices[0].message.content or ""


MODEL_COMPARISON_SYSTEM_PROMPT = (
    "You are a careful data analyst explaining machine-learning model "
    "comparison results to a non-technical reader. You are given a table "
    "of already-computed metrics for each model (never raw predictions). "
    "Explain in plain language: 1) which model(s) look strongest and on "
    "which metric, 2) any clear trade-offs (e.g. a model that's fast but "
    "less accurate), 3) what the top predictive features suggest about the "
    "data, and 4) practical limitations of this comparison (small test set, "
    "no hyperparameter tuning, etc. — say so only if actually applicable). "
    "Never invent a number that isn't in the table, and never declare a "
    "single 'winner' score — describe the trade-offs instead."
)


def build_model_comparison_summary_text(comparison: dict, max_chars: int = 6000) -> str:
    """Compact, structured digest of an already-computed model comparison
    (modules/predict.py::compare_models output) — metrics table + top
    features per model, never raw predictions or training data."""
    lines: list = [f"Task type: {comparison['task']}", f"Trained on {comparison['n_train']} rows, tested on {comparison['n_test']} rows."]
    lines.append("\nMetrics per model:")
    lines.append(comparison["summary"].to_string(index=False))

    lines.append("\nTop predictive features per model (permutation importance):")
    for r in comparison["results"]:
        if "error" in r:
            lines.append(f"- {r['model']}: failed to train ({r['error']})")
            continue
        top = r["importance"].head(5)
        feats = ", ".join(f"{row['source_column']} ({row['importance']:.3f})" for _, row in top.iterrows())
        lines.append(f"- {r['model']}: {feats}")

    text = "\n".join(lines)
    return text[:max_chars]


def generate_model_comparison_summary(client, comparison: dict, model: str | None = None) -> str:
    summary = build_model_comparison_summary_text(comparison)

    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": MODEL_COMPARISON_SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the model comparison summary:\n\n{summary}"},
        ],
    )
    return response.choices[0].message.content or ""
