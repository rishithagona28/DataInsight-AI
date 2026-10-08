"""
Ana — AI Data Analyst

Upload a CSV, get an automatic EDA + dashboard, AI-generated insights,
quick predictions, and a chat agent that answers questions about your
data by writing and running real pandas code (not guessing).
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from modules import charts, chat_agent, cleaning, eda, insights, llm, loader, predict, report

load_dotenv()

st.set_page_config(page_title= "AI Data Analyst", page_icon="📊", layout="wide")

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
# Gemini's OpenAI-compatible endpoint — talking to it through the `openai`
# client means the same code would also work against Groq, OpenRouter, a
# local Ollama server, or any other OpenAI-compatible provider by just
# changing this URL (and the key/model) in .env, with no code changes.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()

# Streamlit's built-in "script is running" toolbar indicator is a fixed
# running-man glyph with no theme hook (it's literally named
# "RunningManIcon" in Streamlit's own frontend source). Swap it for a
# themed ring that spins with the current text color, so it reads as a
# loading spinner rather than a sports icon.
RUNNING_INDICATOR_CSS = """
<style>
[data-testid="stStatusWidget"] svg { display: none; }
[data-testid="stStatusWidget"]::before {
    content: "";
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2.5px solid color-mix(in srgb, currentColor 20%, transparent);
    border-top-color: currentColor;
    animation: ana-ring-spin 0.7s linear infinite;
    display: inline-block;
}
@keyframes ana-ring-spin {
    to { transform: rotate(360deg); }
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_llm_client():
    """Built once per session, not on every script rerun. The key is read
    from the server-side environment here and never leaves this process —
    nothing in this app returns it to the browser. Points at Gemini's
    OpenAI-compatible endpoint by default (see LLM_BASE_URL above)."""
    import openai

    # max_retries=0: the library's built-in retry waits out rate limits (up to a
    # minute+). modules/llm.py instead switches straight to another model.
    return openai.OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL, max_retries=0, timeout=60)


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Cached on the file's bytes. See modules/loader.py for how messy files
    (Excel encodings, semicolons, title rows, bad rows, "₹ 1,200") are handled."""
    return loader.load_csv_bytes(data)


@st.cache_data(show_spinner=False)
def prepare_dataframe(df: pd.DataFrame):
    """Detect column types once and parse text date columns into real
    datetimes, so every tab works from the same, already-typed frame."""
    col_types = eda.column_types(df)
    df = df.copy()
    for col in col_types["datetime"]:
        df[col] = eda.to_datetime_safe(df[col])
    return df, col_types


@st.cache_data(show_spinner=False)
def compute_col_types(df: pd.DataFrame) -> dict:
    return eda.column_types(df)


@st.cache_data(show_spinner=False)
def compute_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    return eda.cardinality_table(df)


@st.cache_data(show_spinner=False)
def compute_quality_report(df: pd.DataFrame, col_types: dict) -> list:
    return eda.data_quality_report(df, col_types)


@st.cache_data(show_spinner=False)
def compute_training_prep(df: pd.DataFrame, target_col: str, id_like_cols: list) -> dict:
    return predict.prepare_training_data(df, target_col, id_like_cols=id_like_cols)


@st.cache_data(show_spinner=False)
def train_one_model(prep: dict, model_name: str) -> dict:
    """Cached per (data split, model) pair — re-running the same model on
    an unchanged selection is instant. Called inside a per-model st.status
    so a cache *miss* still shows real incremental progress instead of one
    long silent block for the whole comparison."""
    return predict.train_and_evaluate_one(prep, model_name)


def _invalidate_derived_state() -> None:
    """Anything computed from the working dataframe — clear it whenever
    that dataframe changes (new upload, or a cleaning step applied/reset)
    so a stale AI/ML result never shows next to different data."""
    st.session_state.pop("insights_text", None)
    st.session_state.pop("model_comparison", None)
    st.session_state.pop("model_comparison_target", None)
    st.session_state.pop("model_comparison_explanation", None)
    st.session_state.pop("dataset_context", None)
    st.session_state.pop("report_html", None)
    st.session_state.pop("last_forecast", None)
    st.session_state["chat_display"] = []
    st.session_state["chat_history"] = []


def render_cleaning(df: pd.DataFrame, col_types: dict) -> None:
    recs = cleaning.build_recommendations(df, col_types)

    log = st.session_state["cleaning_log"]
    if log:
        st.caption(f"{len(log)} cleaning step(s) applied so far:")
        st.dataframe(pd.DataFrame(log)[["title", "columns", "applied_at"]], width="stretch", hide_index=True)
        if st.button("Reset to original data"):
            st.session_state["working_df"] = st.session_state["raw_df"].copy()
            st.session_state["cleaning_log"] = []
            _invalidate_derived_state()
            st.rerun()

    if not recs:
        st.caption("No fixable data-quality issues detected — nothing to clean.")
        return

    st.subheader("Recommended cleaning steps")
    st.caption("Review each recommendation and pick the ones to apply. Nothing changes until you click Apply.")
    selected = []
    for rec in recs:
        checked = st.checkbox(rec["title"], value=rec["default_checked"], key=f"clean_{rec['id']}", help=rec["rationale"])
        if checked:
            selected.append(rec)

    if st.button("Apply selected changes", type="primary", disabled=not selected):
        working = st.session_state["working_df"]
        for rec in selected:
            working = rec["apply"](working)
            st.session_state["cleaning_log"].append({
                "title": rec["title"],
                "columns": ", ".join(rec["columns"]) or "(all rows)",
                "applied_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        st.session_state["working_df"] = working
        _invalidate_derived_state()
        st.rerun()


def render_overview(df: pd.DataFrame, col_types: dict) -> None:
    info = eda.basic_info(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info['n_rows']:,}")
    c2.metric("Columns", info["n_cols"])
    c3.metric("Missing cells", f"{info['missing_pct']}%")
    c4.metric("Duplicate rows", info["n_duplicates"])

    target_suggestions = eda.suggest_target_columns(df, col_types)
    if target_suggestions:
        st.caption(f"Possible prediction targets (by column name): {', '.join(target_suggestions)} — see the Predict tab.")

    st.subheader("Column types")
    kind_by_col = {c: kind for kind, cols in col_types.items() for c in cols}
    cardinality = compute_cardinality(df)
    cardinality["detected_type"] = cardinality["column"].map(kind_by_col)
    st.dataframe(
        cardinality[["column", "detected_type", "dtype", "n_unique", "pct_unique"]],
        width="stretch", hide_index=True,
    )

    st.subheader("Missing values")
    st.dataframe(eda.missing_value_table(df), width="stretch", hide_index=True)

    quality_issues = compute_quality_report(df, col_types)
    st.subheader("Data quality issues")
    if quality_issues:
        issues_df = pd.DataFrame(quality_issues)
        issues_df["columns"] = issues_df["columns"].apply(", ".join)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        issues_df = issues_df.sort_values(by="severity", key=lambda s: s.map(severity_order))
        st.dataframe(issues_df[["severity", "issue", "columns", "detail"]], width="stretch", hide_index=True)
    else:
        st.caption("No data quality issues detected.")

    if col_types["numeric"]:
        st.subheader("Numeric summary")
        st.dataframe(eda.numeric_summary(df, col_types["numeric"]), width="stretch")

    if col_types["categorical"]:
        st.subheader("Categorical columns (top values)")
        cat_summaries = eda.categorical_summary(df, col_types["categorical"])
        n_cols = min(3, len(cat_summaries)) or 1
        cols = st.columns(n_cols)
        for i, (name, table) in enumerate(cat_summaries.items()):
            with cols[i % n_cols]:
                st.caption(name)
                st.dataframe(table, hide_index=True, width="stretch", key=f"cat_summary_{name}")

    st.subheader("Raw data preview")
    st.dataframe(df.head(50), width="stretch")


def render_dashboard(df: pd.DataFrame, col_types: dict) -> None:
    numeric = col_types["numeric"]
    categorical = col_types["categorical"]
    datetime_cols = col_types["datetime"]

    # --- Summary cards from other tabs, if already generated this session
    # (reads session_state only — never recomputes) ---
    insights_text = st.session_state.get("insights_text")
    comparison = st.session_state.get("model_comparison")
    forecast = st.session_state.get("last_forecast")
    if insights_text or comparison or forecast:
        with st.container(horizontal=True):
            if insights_text:
                with st.container(border=True):
                    st.markdown(":material/auto_awesome: **AI insights**")
                    st.caption(insights_text[:180].rsplit(" ", 1)[0] + "…")
                    st.caption("See the AI insights tab for the full analysis.")
            if forecast:
                with st.container(border=True):
                    st.markdown(":material/trending_up: **Latest forecast**")
                    growth = forecast["growth_pct"]
                    st.caption(f"{forecast['value_col']}, {forecast['periods']} period(s): {growth:+.1f}% expected change" if growth is not None else f"{forecast['value_col']}, {forecast['periods']} period(s) forecast")
                    st.caption("See the Predict tab for the chart.")
            if comparison:
                with st.container(border=True):
                    st.markdown(":material/model_training: **Latest model comparison**")
                    best = comparison["summary"].iloc[0]["model"] if not comparison["summary"].empty else "—"
                    st.caption(f"Task: {comparison['task']} · {len(comparison['results'])} model(s), incl. {best}")
                    st.caption("See the Predict tab for full metrics.")

    # --- Filters (scoped to this tab only; other tabs keep using the full
    # working dataset) ---
    filtered = df
    with st.container(border=True):
        st.caption("Filters")
        filter_cols = st.columns(3)

        if datetime_cols:
            date_col = datetime_cols[0]
            valid_dates = df[date_col].dropna()
            if not valid_dates.empty:
                min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
                with filter_cols[0]:
                    date_range = st.date_input(f"{date_col} range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = date_range
                    filtered = filtered[(filtered[date_col].dt.date >= start) & (filtered[date_col].dt.date <= end)]

        cat_filter_cols = sorted(categorical, key=lambda c: df[c].nunique(dropna=True))[:2]
        for i, col in enumerate(cat_filter_cols, start=1):
            with filter_cols[i % 3]:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                chosen = st.multiselect(col, options, key=f"dash_filter_{col}")
                if chosen:
                    filtered = filtered[filtered[col].astype(str).isin(chosen)]

        search = st.text_input("Search (matches any text column)", key="dash_search")
        if search and categorical:
            mask = pd.Series(False, index=filtered.index)
            for col in categorical:
                mask = mask | filtered[col].astype(str).str.contains(search, case=False, na=False)
            filtered = filtered[mask]

    if filtered.empty:
        st.warning("No rows match the current filters.", icon=":material/filter_alt_off:")
        return

    # --- KPI row + anomaly badges ---
    with st.container(horizontal=True):
        st.metric("Rows shown", f"{len(filtered):,}", border=True)
        st.metric("Of total", f"{100 * len(filtered) / len(df):.0f}%" if len(df) else "0%", border=True)
        for col in numeric[:3]:
            st.metric(f"Avg {col}", round(float(filtered[col].mean()), 2) if filtered[col].notna().any() else "—", border=True)

    outliers = eda.outlier_summary(filtered, numeric)
    if not outliers.empty:
        with st.container(horizontal=True):
            for _, row in outliers.iterrows():
                st.badge(f"{int(row['n_outliers'])} outlier(s) in {row['column']}", icon=":material/warning:", color="orange")

    # --- Drill-down: click a category bar to filter the rest of the tab ---
    drilldown_col = cat_filter_cols[0] if cat_filter_cols else (categorical[0] if categorical else None)
    drilldown_fig = charts.safe_chart(charts.bar_top_categories, filtered, drilldown_col) if drilldown_col and filtered[drilldown_col].nunique(dropna=True) >= 2 else None
    if drilldown_fig:
        st.caption(f"Click a bar to drill into {drilldown_col} — click again (or Clear) to reset.")
        event = st.plotly_chart(drilldown_fig, on_select="rerun", selection_mode="points", key="dashboard_drilldown")
        selected_categories = []
        if event and event.selection and event.selection.get("points"):
            selected_categories = sorted({p.get("y") for p in event.selection["points"] if p.get("y") is not None})
        if selected_categories:
            st.badge(f"Drilled into {drilldown_col}: {', '.join(map(str, selected_categories))}", icon=":material/zoom_in:", color="blue")
            filtered = filtered[filtered[drilldown_col].astype(str).isin(map(str, selected_categories))]

    if len(numeric) >= 2:
        corr_fig = charts.safe_chart(charts.correlation_heatmap, eda.correlation_matrix(filtered, numeric))
        if corr_fig:
            st.subheader("Correlation")
            st.plotly_chart(corr_fig, width="stretch")

    figures = charts.build_dashboard_figures(filtered, col_types)
    if not figures:
        st.info("Not enough numeric or categorical columns to auto-generate a dashboard.")
        return

    cols = st.columns(2)
    for i, (section, fig) in enumerate(figures):
        with cols[i % 2]:
            st.plotly_chart(fig, width="stretch", key=f"dashboard_fig_{i}_{section}")


def _plain_language_summary(task: str, metrics: dict, target_col: str) -> str:
    """A templated (not AI-generated — costs no extra tokens) one-line
    translation of the raw metrics into plain language."""
    if task == "classification":
        acc = metrics.get("Accuracy")
        if acc is None:
            return "Trained, but accuracy couldn't be computed for this data."
        return f"Correctly predicts {target_col} about {acc:.0%} of the time on data it hasn't seen."
    mae = metrics.get("MAE")
    if mae is None:
        return "Trained, but error couldn't be computed for this data."
    line = f"Predictions are typically within {mae:,.2f} of the actual {target_col}."
    r2 = metrics.get("R2")
    if r2 is not None:
        line += f" Explains about {max(r2, 0):.0%} of the variation in {target_col}."
    return line


def render_predict(df: pd.DataFrame, col_types: dict) -> None:
    datetime_cols = col_types["datetime"]
    numeric_cols = col_types["numeric"]

    default_forecast = bool(datetime_cols and numeric_cols)
    mode = st.radio(
        "What do you want to predict?",
        ["Forecast a metric over time", "Predict a column from the others"],
        index=0 if default_forecast else 1,
        horizontal=True,
    )

    if mode == "Forecast a metric over time":
        if not datetime_cols or not numeric_cols:
            st.warning("I couldn't find both a date column and a numeric column to forecast. Try the other mode instead.")
            return
        date_col = st.selectbox("Date column", datetime_cols)
        value_col = st.selectbox("Metric to forecast", numeric_cols)

        horizon_choice = st.segmented_control("Forecast how far ahead?", ["Next 1 month", "Next 3 months", "Next 6 months", "Custom"], default="Next 3 months")
        horizon_months = {"Next 1 month": 1, "Next 3 months": 3, "Next 6 months": 6}.get(horizon_choice)
        custom_periods = st.slider("Custom periods to forecast forward", 1, 60, 12) if horizon_choice == "Custom" else None

        if st.button("Run forecast", type="primary", icon=":material/trending_up:"):
            ok, reason = predict.can_forecast(df, date_col, value_col)
            if not ok:
                st.error(reason, icon=":material/error:")
                return
            with st.skeleton(height=480):
                try:
                    result = predict.forecast_time_series(df, date_col, value_col, periods=custom_periods, horizon_months=horizon_months)
                except Exception as exc:
                    st.error(f"Couldn't fit a forecast for this column: {exc}", icon=":material/error:")
                    return

                with st.container(border=True):
                    st.caption(f"Detected frequency: {result['freq_detected']} · seasonality: {result['seasonal']} · {result['periods']} period(s) forecast")
                    if result["growth_pct"] is not None:
                        st.metric(f"Expected change in {value_col}", f"{result['growth_pct']:+.1f}%", delta=f"{result['growth_pct']:+.1f}%", border=True,
                                  help=f"Forecast for the next {result['periods']} {result.get('period_name') or 'period'}(s) compared with the last {result['periods']}.")

                    fig = go.Figure()
                    fig.add_scatter(x=result["actual"]["date"], y=result["actual"]["value"], mode="lines", name="Actual", line=dict(color=charts.CATEGORICAL_PALETTE[0]))
                    fig.add_scatter(x=result["forecast"]["date"], y=result["forecast"]["value"], mode="lines", name="Forecast", line=dict(color=charts.CATEGORICAL_PALETTE[1], dash="dash"))
                    fig.add_scatter(
                        x=pd.concat([result["forecast"]["date"], result["forecast"]["date"][::-1]]),
                        y=pd.concat([result["forecast"]["high"], result["forecast"]["low"][::-1]]),
                        fill="toself", fillcolor="rgba(235,104,52,0.15)", line=dict(width=0),
                        name="Approx. 80% band", showlegend=True,
                    )
                    fig.update_layout(title=f"{value_col}: actual vs. forecast", plot_bgcolor=charts.SURFACE, paper_bgcolor=charts.SURFACE)
                    st.plotly_chart(fig, width="stretch")
                    st.caption("This is a simple trend model for exploratory purposes, not a statistically rigorous forecast.")
                    st.dataframe(result["forecast"][["date", "value", "low", "high"]], hide_index=True, width="stretch")
            st.session_state["last_forecast"] = {"value_col": value_col, "growth_pct": result["growth_pct"], "periods": result["periods"]}
            st.toast("Forecast ready", icon=":material/check_circle:")

    else:
        # A raw timestamp isn't a sensible classification/regression target
        # (that's what Forecast mode is for) — don't offer datetime columns
        # here at all, rather than letting the model choke on one.
        predictable_cols = [c for c in df.columns if c not in col_types["datetime"]]
        if not predictable_cols:
            st.warning("Every column here looks like a date — nothing left to predict. Try Forecast mode instead.")
            return
        target_suggestions = [c for c in eda.suggest_target_columns(df, col_types) if c in predictable_cols]
        default_target_index = predictable_cols.index(target_suggestions[0]) if target_suggestions else 0
        target_col = st.selectbox("Column to predict", predictable_cols, index=default_target_index)
        id_like_default = [c for c in col_types.get("high_cardinality", []) if c != target_col]
        id_like = st.multiselect("Exclude any ID-like columns (auto-suggested)", options=list(df.columns), default=id_like_default)

        task_preview = predict.infer_task_type(df[target_col])
        available_models = list(predict.MODEL_REGISTRY[task_preview].keys())
        default_model = "Random Forest" if "Random Forest" in available_models else available_models[0]

        compare_mode = st.checkbox(
            "Compare more models",
            help="Train every available model with full metrics, cross-validation, and explainability. Off by default — one good model is usually enough, and it's faster.",
        )
        if compare_mode:
            selected_models = st.multiselect(f"Models to compare ({task_preview})", available_models, default=available_models)
            run_label = "Compare models"
        else:
            selected_models = [default_model]
            st.caption(f"Will train **{default_model}** — a well-rounded default. Check \"Compare more models\" above to see others.")
            run_label = "Predict"

        if st.button(run_label, type="primary", disabled=not selected_models, icon=":material/model_training:"):
            try:
                prep = compute_training_prep(df, target_col, id_like)
            except Exception as exc:
                st.error(f"Couldn't prepare this column for training: {exc}")
                return

            if prep["sampled_from"]:
                st.caption(f"Training on a sample of {len(prep['X_train']) + len(prep['X_test']):,} rows out of {prep['sampled_from']:,} for speed.")
            if prep["excluded_features"]:
                st.caption(f"Excluded from training (dates or very high-cardinality): {', '.join(prep['excluded_features'])}")

            results = []
            for name in selected_models:
                with st.status(f"Training {name}...") as status:
                    entry = train_one_model(prep, name)
                    results.append(entry)
                    status.update(label=(f"{name} failed" if "error" in entry else f"{name} trained in {entry['train_seconds']}s"), state=("error" if "error" in entry else "complete"))

            st.session_state["model_comparison"] = {
                "task": prep["task"], "n_train": len(prep["X_train"]), "n_test": len(prep["X_test"]),
                "feature_cols": prep["feature_cols"], "sampled_from": prep["sampled_from"],
                "results": results, "summary": predict.summarize_comparison(results),
            }
            st.session_state["model_comparison_target"] = target_col
            st.session_state.pop("model_comparison_explanation", None)

        comparison = st.session_state.get("model_comparison")
        if comparison and st.session_state.get("model_comparison_target") == target_col:
            st.caption(f"Task type: {comparison['task']} · trained on {comparison['n_train']} rows, tested on {comparison['n_test']}")

            single = len(comparison["results"]) == 1 and "error" not in comparison["results"][0]
            if single:
                r = comparison["results"][0]
                st.subheader(f"Prediction results: {r['model']}")
                st.success(_plain_language_summary(comparison["task"], r["metrics"], target_col), icon=":material/check_circle:")
            else:
                st.subheader("Model comparison")
                st.caption("Raw metrics only — pick the trade-off that matters to you, there's no single overall score.")
            st.dataframe(comparison["summary"], width="stretch", hide_index=True)

            if API_KEY:
                if st.button("Explain these results", icon=":material/auto_awesome:"):
                    with st.spinner("Analyzing the comparison..."):
                        try:
                            client = get_llm_client()
                            st.session_state["model_comparison_explanation"] = insights.generate_model_comparison_summary(client, comparison)
                        except Exception as exc:
                            st.error(f"Couldn't generate an explanation: {exc}")
                explanation = st.session_state.get("model_comparison_explanation")
                if explanation:
                    st.markdown(explanation)

            for r in comparison["results"]:
                with st.expander(r["model"]):
                    if "error" in r:
                        st.error(r["error"])
                        continue

                    imp = r["importance"].head(15)
                    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["source_column"], orientation="h", marker_color=charts.CATEGORICAL_PALETTE[0]))
                    fig.update_layout(title="Feature importance (permutation)", yaxis=dict(autorange="reversed"), plot_bgcolor=charts.SURFACE, paper_bgcolor=charts.SURFACE)
                    st.plotly_chart(fig, width="stretch", key=f"importance_{r['model']}")

                    if comparison["task"] == "classification":
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Confusion matrix (test set)")
                            st.dataframe(r["confusion_matrix"], width="stretch", key=f"cm_{r['model']}")
                        with c2:
                            st.caption("Classification report")
                            st.dataframe(r["classification_report"], width="stretch", key=f"report_{r['model']}")
                    else:
                        st.caption("Actual vs. predicted (test set)")
                        comp_df = pd.DataFrame({"actual": r["y_test"], "predicted": r["y_pred"]})
                        st.dataframe(comp_df.head(30), hide_index=True, width="stretch", key=f"actual_pred_{r['model']}")


def render_insights(df: pd.DataFrame, col_types: dict) -> None:
    if not API_KEY:
        st.warning("Add a GEMINI_API_KEY to your .env file to enable AI-generated insights.", icon=":material/key_off:")
        return

    label = "Regenerate insights" if "insights_text" in st.session_state else "Generate insights"
    if st.button(label, type="primary", icon=":material/auto_awesome:"):
        with st.skeleton(height=260):
            try:
                client = get_llm_client()
                quality_issues = compute_quality_report(df, col_types)
                st.session_state["insights_text"] = insights.generate_insights(client, df, col_types, quality_issues=quality_issues)
            except Exception as exc:
                st.error(f"Couldn't generate insights: {exc}", icon=":material/error:")
                return
        st.toast("Insights ready", icon=":material/check_circle:")

    if "insights_text" in st.session_state:
        with st.container(border=True):
            st.markdown(st.session_state["insights_text"])


CHAT_SUGGESTIONS = {
    ":material/summarize: Summarize this data": "Summarize this data in simple words, like you would for a manager.",
    ":material/leaderboard: What performs best?": "Which group or category performs best, and by how much?",
    ":material/trending_up: What happens next?": "What will the main numbers look like over the next 3 months?",
    ":material/lightbulb: Recommendations": "Based on this data, give me 3 practical recommendations.",
}


def _render_answer_extras(artifacts: list, key: str) -> None:
    """Show what supports an answer: any chart the user asked for, otherwise the
    final table behind the numbers (collapsed). Never the code."""
    figures = [a["result"] for a in artifacts if isinstance(a.get("result"), go.Figure)]
    for i, fig in enumerate(figures):
        st.plotly_chart(fig, width="stretch", key=f"{key}_fig{i}")
    if figures:
        return
    tables = [a["result"] for a in artifacts if isinstance(a.get("result"), (pd.DataFrame, pd.Series))]
    if tables:
        table = tables[-1].to_frame() if isinstance(tables[-1], pd.Series) else tables[-1]
        if 0 < len(table) <= 30:
            with st.expander("See the numbers", icon=":material/table_chart:"):
                st.dataframe(table, width="stretch", key=f"{key}_table")


def render_chat(df: pd.DataFrame, col_types: dict) -> None:
    if not API_KEY:
        st.warning("Add a GEMINI_API_KEY to your .env file to enable the chat agent.")
        return

    client = get_llm_client()

    if "chat_display" not in st.session_state:
        st.session_state["chat_display"] = []  # [{'role', 'text', 'artifacts'}]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # OpenAI-style chat message format (role/content/tool_calls)
    if "dataset_context" not in st.session_state:
        # Built once per dataset (see main()'s fingerprint reset), reused every
        # turn so the model already knows the key facts about the data.
        quality_issues = compute_quality_report(df, col_types)
        st.session_state["dataset_context"] = chat_agent.build_dataset_context(df, col_types, quality_issues)

    for t, turn in enumerate(st.session_state["chat_display"]):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            _render_answer_extras(turn.get("artifacts", []), key=f"chat_hist_{t}")

    # Clickable starter questions, shown only before the first message
    question = None
    if not st.session_state["chat_display"]:
        st.caption("Ask anything about your data in plain English, or try one of these:")
        picked = st.pills("Suggested questions", list(CHAT_SUGGESTIONS), label_visibility="collapsed", key="chat_suggestion")
        if picked:
            question = CHAT_SUGGESTIONS[picked]

    typed = st.chat_input("Ask a question about your data...", submit_mode="disable")
    question = typed or question
    if not question:
        return

    st.session_state["chat_display"].append({"role": "user", "text": question})
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.time()
        with st.status(":shimmer[Analyzing your data]", type="compact") as status:
            try:
                outcome = chat_agent.ask(client, df, st.session_state["chat_history"], dataset_context=st.session_state["dataset_context"])
            except llm.AllModelsBusy as exc:
                status.update(label="AI limit reached for now", state="error")
                st.session_state["chat_history"].pop()
                st.session_state["chat_display"].append({"role": "assistant", "text": f":orange[{exc}]"})
                st.rerun()
            except Exception as exc:  # noqa: BLE001 — show a friendly message instead of a traceback
                status.update(label="Something went wrong", state="error")
                st.session_state["chat_history"].pop()
                st.session_state["chat_display"].append({"role": "assistant", "text": f":red[Sorry, I couldn't answer that just now ({type(exc).__name__}). Please try again.]"})
                st.rerun()
            steps = len(outcome["artifacts"])
            label = f"Analyzed your data in {time.time() - started:.0f}s"
            if steps:
                label += f" · {steps} calculation{'s' if steps > 1 else ''}"
            status.update(label=label, state="complete")
        st.session_state["chat_history"] = outcome["history"]
        st.markdown(outcome["final_text"])
        turn_index = len(st.session_state["chat_display"])
        _render_answer_extras(outcome["artifacts"], key=f"chat_new_{turn_index}")
    st.session_state["chat_display"].append({"role": "assistant", "text": outcome["final_text"], "artifacts": outcome["artifacts"]})


def render_report(df: pd.DataFrame, col_types: dict, dataset_name: str) -> None:
    st.caption("Compiles whatever's already been computed this session — no extra model calls or training.")

    have_insights = "insights_text" in st.session_state
    have_comparison = "model_comparison" in st.session_state
    with st.container(horizontal=True):
        st.badge("AI insights" + (" ready" if have_insights else " not generated"), color="green" if have_insights else "gray")
        st.badge("Model comparison" + (" ready" if have_comparison else " not run"), color="green" if have_comparison else "gray")
        st.badge(f"{len(st.session_state['cleaning_log'])} cleaning step(s)", color="blue")

    if st.button("Generate report", type="primary", icon=":material/description:"):
        quality_issues = compute_quality_report(df, col_types)
        report_charts = []
        numeric = col_types.get("numeric", [])
        if len(numeric) >= 2:
            corr_fig = charts.safe_chart(charts.correlation_heatmap, eda.correlation_matrix(df, numeric))
            if corr_fig:
                report_charts.append(corr_fig)
        report_charts.extend(fig for _section, fig in charts.build_dashboard_figures(df, col_types)[:3])

        html_report = report.build_report_html(
            df, col_types, quality_issues,
            cleaning_log=st.session_state.get("cleaning_log"),
            insights_text=st.session_state.get("insights_text"),
            comparison=st.session_state.get("model_comparison"),
            comparison_explanation=st.session_state.get("model_comparison_explanation"),
            charts=report_charts,
            dataset_name=dataset_name,
        )
        st.session_state["report_html"] = html_report

    if "report_html" in st.session_state:
        st.download_button(
            "Download report (.html)",
            data=st.session_state["report_html"],
            file_name=f"{dataset_name.rsplit('.', 1)[0]}_report.html",
            mime="text/html",
            icon=":material/download:",
        )
        with st.expander("Preview"):
            with st.container(height=600):
                st.html(st.session_state["report_html"], unsafe_allow_javascript=True)


def render_landing() -> None:
    st.info("Upload a CSV in the sidebar to get started.", icon=":material/upload_file:")
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown(":material/cleaning_services: **Data quality & cleaning**")
            st.caption("Missing values, duplicates, and inconsistent text — detected and explained, applied only with your approval.")
        with st.container(border=True):
            st.markdown(":material/dashboard: **Interactive dashboard**")
            st.caption("Filters, drill-down, and auto-picked charts that stay fast even on large files.")
        with st.container(border=True):
            st.markdown(":material/model_training: **Predictions & forecasts**")
            st.caption("A one-click model for any column, or forecast a metric 1, 3, 6 months — or a custom horizon — ahead.")
        with st.container(border=True):
            st.markdown(":material/chat: **Chat with your data**")
            st.caption("Ask in plain language — answers, charts, and forecasts are computed, not guessed.")


def main() -> None:
    st.html(RUNNING_INDICATOR_CSS)

    with st.sidebar:
        st.markdown("## :material/analytics: Ana")
        st.caption("Your AI-powered data analyst")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"])
        st.space("small")
        if API_KEY:
            st.badge("AI features ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("AI features need a key", icon=":material/key_off:", color="orange")
            st.caption("Add `GEMINI_API_KEY` to your `.env` file to unlock AI insights, chat, and forecasting explanations.")

    st.title("Gemini-powered AI data analyst", icon=":material/analytics:")
    st.caption("Upload a CSV to get an automatic dashboard, AI insights, quick predictions, and a chat agent for your data.")

    if uploaded_file is None:
        render_landing()
        return

    try:
        raw_df, load_notes = load_csv(uploaded_file.getvalue())
    except loader.CSVLoadError as exc:
        st.error(f"Couldn't read **{uploaded_file.name}**: {exc}", icon=":material/error:")
        return
    except Exception as exc:
        st.error(f"Couldn't read **{uploaded_file.name}** as a table: {exc}", icon=":material/error:")
        return
    if load_notes:
        with st.expander(f"Loaded **{uploaded_file.name}** with {len(load_notes)} automatic adjustment(s)", icon=":material/auto_fix_high:"):
            for note in load_notes:
                st.markdown(f"- {note}")

    typed_df, _ = prepare_dataframe(raw_df)

    fingerprint = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("dataset_fingerprint") != fingerprint:
        st.session_state["dataset_fingerprint"] = fingerprint
        st.session_state["raw_df"] = typed_df
        st.session_state["working_df"] = typed_df.copy()
        st.session_state["cleaning_log"] = []
        _invalidate_derived_state()  # a new dataset invalidates anything computed from the old one

    df = st.session_state["working_df"]
    col_types = compute_col_types(df)

    with st.sidebar:
        st.badge(f"{df.shape[0]:,} rows · {df.shape[1]} columns", icon=":material/table_chart:", color="blue")
        if st.session_state["cleaning_log"]:
            st.caption(f"{len(st.session_state['cleaning_log'])} cleaning step(s) applied.")

    tabs = st.tabs([
        ":material/cleaning_services: Data quality & cleaning",
        ":material/query_stats: Overview & EDA",
        ":material/dashboard: Dashboard",
        ":material/model_training: Predict",
        ":material/auto_awesome: AI insights",
        ":material/chat: Chat with your data",
        ":material/description: Report",
    ])
    with tabs[0]:
        render_cleaning(df, col_types)
    with tabs[1]:
        render_overview(df, col_types)
    with tabs[2]:
        render_dashboard(df, col_types)
    with tabs[3]:
        render_predict(df, col_types)
    with tabs[4]:
        render_insights(df, col_types)
    with tabs[5]:
        render_chat(df, col_types)
    with tabs[6]:
        render_report(df, col_types, uploaded_file.name)


if __name__ == "__main__":
    main()
