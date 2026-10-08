"""
Automated HTML report for Ana: assembles whatever has already been
computed this session (profiling, cleaning log, AI insights, model
comparison, a few key charts) into one self-contained, downloadable HTML
file. Nothing here recomputes analysis — it only formats results that
were already produced elsewhere, so generating a report costs no extra
LLM tokens and no extra model training.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from . import eda

_STYLE = """
<style>
  body { font-family: -apple-system, 'Segoe UI', system-ui, sans-serif; color: #0b0b0b; background: #fcfcfb; margin: 0; padding: 0 0 48px; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 28px; margin-bottom: 4px; }
  h2 { font-size: 20px; margin-top: 40px; border-bottom: 1px solid #e1e0d9; padding-bottom: 8px; }
  .muted { color: #898781; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e1e0d9; }
  th { background: #f2f1ec; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 4px; }
  .sev-high { background: #fbe5e4; color: #8a1f1e; }
  .sev-medium { background: #fdf1da; color: #7a4a00; }
  .sev-low { background: #eef1ee; color: #3a3a36; }
  .ai-text { white-space: pre-wrap; line-height: 1.5; background: #f2f1ec; padding: 16px; border-radius: 12px; }
  .chart { margin: 16px 0; }
</style>
"""


def _df_to_html(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Nothing to show.</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="", escape=True)


def _quality_issues_html(quality_issues: list) -> str:
    if not quality_issues:
        return "<p class='muted'>No data-quality issues detected.</p>"
    rows = "".join(
        f"<tr><td><span class='badge sev-{i['severity']}'>{i['severity']}</span></td>"
        f"<td>{html.escape(i['issue'])}</td><td>{html.escape(i['detail'])}</td></tr>"
        for i in quality_issues
    )
    return f"<table><tr><th>Severity</th><th>Issue</th><th>Detail</th></tr>{rows}</table>"


def _cleaning_log_html(cleaning_log: list) -> str:
    if not cleaning_log:
        return "<p class='muted'>No cleaning steps were applied — this report reflects the data as uploaded.</p>"
    rows = "".join(
        f"<tr><td>{html.escape(e['title'])}</td><td>{html.escape(e['columns'])}</td><td>{html.escape(e['applied_at'])}</td></tr>"
        for e in cleaning_log
    )
    return f"<table><tr><th>Step</th><th>Column(s)</th><th>Applied at</th></tr>{rows}</table>"


def _ai_text_html(text: str | None) -> str:
    if not text:
        return "<p class='muted'>Not generated this session.</p>"
    return f"<div class='ai-text'>{html.escape(text)}</div>"


def _charts_html(charts: list) -> str:
    parts = []
    for i, fig in enumerate(charts):
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False), config={"displaylogo": False}))
    return "".join(f"<div class='chart'>{p}</div>" for p in parts)


def build_report_html(
    df: pd.DataFrame,
    col_types: dict,
    quality_issues: list,
    cleaning_log: list | None = None,
    insights_text: str | None = None,
    comparison: dict | None = None,
    comparison_explanation: str | None = None,
    charts: list | None = None,
    dataset_name: str = "your dataset",
) -> str:
    info = eda.basic_info(df)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    numeric_html = _df_to_html(eda.numeric_summary(df, col_types.get("numeric", [])))
    cardinality_html = _df_to_html(eda.cardinality_table(df))

    comparison_section = "<p class='muted'>No model comparison run this session.</p>"
    if comparison is not None:
        comparison_section = (
            f"<p>Task type: <strong>{comparison['task']}</strong> · trained on {comparison['n_train']} rows, tested on {comparison['n_test']} rows.</p>"
            + _df_to_html(comparison["summary"], max_rows=20)
            + _ai_text_html(comparison_explanation)
        )

    body = f"""
<div class="wrap">
  <h1>AI Data Analyst report</h1>
  <p class="muted">{html.escape(dataset_name)} · generated {generated_at}</p>

  <h2>Dataset overview</h2>
  <table>
    <tr><th>Rows</th><td>{info['n_rows']:,}</td></tr>
    <tr><th>Columns</th><td>{info['n_cols']}</td></tr>
    <tr><th>Missing cells</th><td>{info['missing_pct']}%</td></tr>
    <tr><th>Duplicate rows</th><td>{info['n_duplicates']}</td></tr>
    <tr><th>Memory</th><td>{info['memory_mb']} MB</td></tr>
  </table>
  {cardinality_html}

  <h2>Data quality summary</h2>
  {_quality_issues_html(quality_issues)}

  <h2>Cleaning steps applied</h2>
  {_cleaning_log_html(cleaning_log or [])}

  <h2>Key statistics</h2>
  {numeric_html}

  <h2>Visualizations</h2>
  {_charts_html(charts or [])}

  <h2>AI insights</h2>
  {_ai_text_html(insights_text)}

  <h2>Machine learning model comparison</h2>
  {comparison_section}

  <h2>Limitations</h2>
  <ul>
    <li>Forecasts use a simple trend model (Holt's exponential smoothing) — exploratory, not statistically rigorous.</li>
    <li>Model comparisons use default hyperparameters and a single train/test split — a fast first read on predictability, not a tuned, production-ready model.</li>
    <li>AI-generated text is written from the programmatically computed numbers above, never from the raw rows — verify anything decision-critical against the tables here.</li>
  </ul>
</div>
"""
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>AI Data Analyst report</title>{_STYLE}</head><body>{body}</body></html>"
