"""
Auto-dashboard chart builders for DataInsight AI, using Plotly.

Color and form choices follow a fixed-order categorical palette, a single
blue sequential ramp for magnitude, and a blue<->red diverging pair with a
neutral midpoint for correlation — never a rainbow, never dual axes.

Performance: chart types that would otherwise embed one Plotly point per
row (scatter, histogram, box, trend lines) are capped, binned, or
resampled server-side before a Figure is built, so the JSON payload sent
to the browser stays small regardless of dataset size — a raw, unbinned
histogram or scatter over a few hundred thousand rows is several MB of
JSON per chart and is what actually freezes the browser tab, not the
Python-side computation (which is already fast).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Fixed-order categorical palette (colorblind-safe adjacent pairs). Always
# assign colors in this order — never cycle or reassign when a filter
# changes which series are on screen.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

DIVERGING_BLUE_RED = ["#0d366b", "#5598e7", "#f0efec", "#e34948", "#8a1f1e"]

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

BASE_LAYOUT = dict(
    plot_bgcolor=SURFACE,
    paper_bgcolor=SURFACE,
    font=dict(color=INK_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
    margin=dict(l=40, r=20, t=48, b=40),
)

# A raw per-row trace (scatter) above this many points is sampled before
# plotting — past a few thousand points markers just overplot anyway, so
# nothing meaningful is lost visually.
MAX_CHART_POINTS = 5000
# A trend line with more raw distinct timestamps than this is resampled
# to a coarser grain instead of plotting every row.
MAX_TREND_POINTS = 500


def _style(fig: go.Figure, show_legend: bool = False) -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, showlegend=show_legend)
    fig.update_xaxes(showgrid=False, linecolor=GRIDLINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=INK_MUTED))
    return fig


def safe_chart(builder, *args, **kwargs) -> go.Figure | None:
    """Build a chart, swallowing errors from any single bad column (all-NaN,
    a single unique value, an unexpected dtype) so one broken chart never
    takes the whole dashboard — or app.py's direct chart calls — down with
    it. Public: used both by build_dashboard_figures below and directly by
    app.py wherever it builds a chart outside that helper."""
    try:
        fig = builder(*args, **kwargs)
        return fig if fig is not None and len(fig.data) else None
    except Exception:
        return None


def histogram(df: pd.DataFrame, col: str, nbins: int = 30) -> go.Figure:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric data in {col} to histogram.")
    counts, edges = np.histogram(values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=CATEGORICAL_PALETTE[0], marker_line_width=0))
    fig.update_layout(title=f"Distribution of {col}")
    return _style(fig)


def bar_top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    vc = df[col].astype(str).value_counts().head(top_n).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=CATEGORICAL_PALETTE[0]))
    fig.update_layout(title=f"Top {min(top_n, len(vc))} values in {col}")
    return _style(fig)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        colorscale=[[0, DIVERGING_BLUE_RED[0]], [0.5, DIVERGING_BLUE_RED[2]], [1, DIVERGING_BLUE_RED[4]]],
        zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="corr"),
    ))
    fig.update_layout(title="Correlation between numeric columns")
    return _style(fig)


def _resample_for_trend(df: pd.DataFrame, date_col: str, value_col: str, max_points: int = MAX_TREND_POINTS) -> tuple[pd.DataFrame, bool]:
    """Aggregate to the coarsest grain that both fits under max_points and
    stays at least daily — exact means, not a sample, so the trend shape
    is preserved. Returns (plot_df, was_resampled)."""
    plot_df = df[[date_col, value_col]].dropna().sort_values(date_col)
    if plot_df.empty:
        return plot_df, False

    n_unique = plot_df[date_col].nunique()
    if n_unique <= max_points:
        agg = plot_df.groupby(date_col, as_index=False)[value_col].mean()
        return agg, False

    indexed = plot_df.set_index(date_col)[value_col]
    for freq in ("D", "W", "MS", "QS", "YS"):
        resampled = indexed.resample(freq).mean().dropna()
        if len(resampled) <= max_points or freq == "YS":
            return resampled.reset_index(), True
    return indexed.resample("YS").mean().dropna().reset_index(), True  # pragma: no cover — loop above always returns


def time_series_line(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    plot_df, resampled = _resample_for_trend(df, date_col, value_col)
    fig = go.Figure(go.Scatter(
        x=plot_df[date_col], y=plot_df[value_col], mode="lines",
        line=dict(color=CATEGORICAL_PALETTE[0], width=2),
    ))
    title = f"{value_col} over time"
    fig.update_layout(title=f"{title} (aggregated)" if resampled else title)
    return _style(fig)


def area_chart(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    plot_df, resampled = _resample_for_trend(df, date_col, value_col)
    fig = go.Figure(go.Scatter(
        x=plot_df[date_col], y=plot_df[value_col], mode="lines", fill="tozeroy",
        line=dict(color=CATEGORICAL_PALETTE[0], width=2),
        fillcolor="rgba(42,120,214,0.15)",
    ))
    title = f"{value_col} over time (cumulative view)"
    fig.update_layout(title=f"{title}, aggregated" if resampled else title)
    return _style(fig)


def scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None = None, max_points: int = MAX_CHART_POINTS) -> go.Figure:
    plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    sampled = len(plot_df) > max_points
    if sampled:
        plot_df = plot_df.sample(max_points, random_state=42)

    if color_col:
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, color_discrete_sequence=CATEGORICAL_PALETTE)
        fig = _style(fig, show_legend=True)
    else:
        fig = go.Figure(go.Scatter(x=plot_df[x_col], y=plot_df[y_col], mode="markers", marker=dict(color=CATEGORICAL_PALETTE[0], size=7, opacity=0.75)))
        fig = _style(fig)
    title = f"{y_col} vs. {x_col}"
    fig.update_layout(title=f"{title} (sample of {max_points:,} rows)" if sampled else title)
    return fig


def _box_stats(vals: pd.Series) -> dict:
    q1, median, q3 = vals.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    return {
        "q1": q1, "median": median, "q3": q3, "mean": vals.mean(),
        "low": max(vals.min(), q1 - 1.5 * iqr),
        "high": min(vals.max(), q3 + 1.5 * iqr),
    }


def box_plot(df: pd.DataFrame, value_col: str, group_col: str | None = None, max_groups: int = 8) -> go.Figure:
    """Built from precomputed quartiles/fences (exact, server-side) rather
    than handing Plotly the raw column — a box plot only ever needs the
    five-number summary, so there's no reason to ship every row."""
    if group_col:
        top_groups = df[group_col].astype(str).value_counts().head(max_groups).index.tolist()
        names, q1s, medians, q3s, lows, highs, means = [], [], [], [], [], [], []
        for g in top_groups:
            vals = df.loc[df[group_col].astype(str) == g, value_col].dropna()
            if vals.empty:
                continue
            s = _box_stats(vals)
            names.append(g)
            q1s.append(s["q1"]); medians.append(s["median"]); q3s.append(s["q3"])
            lows.append(s["low"]); highs.append(s["high"]); means.append(s["mean"])
        fig = go.Figure(go.Box(x=names, q1=q1s, median=medians, q3=q3s, lowerfence=lows, upperfence=highs, mean=means, marker_color=CATEGORICAL_PALETTE[0]))
        fig.update_layout(title=f"{value_col} by {group_col}")
    else:
        vals = df[value_col].dropna()
        if vals.empty:
            raise ValueError(f"No data in {value_col} to plot.")
        s = _box_stats(vals)
        fig = go.Figure(go.Box(x=[value_col], q1=[s["q1"]], median=[s["median"]], q3=[s["q3"]], lowerfence=[s["low"]], upperfence=[s["high"]], mean=[s["mean"]], marker_color=CATEGORICAL_PALETTE[0]))
        fig.update_layout(title=f"Spread of {value_col}")
    return _style(fig)


def donut_chart(df: pd.DataFrame, col: str, top_n: int = 6) -> go.Figure:
    vc = df[col].astype(str).value_counts()
    top = vc.head(top_n)
    if len(vc) > top_n:
        top = pd.concat([top, pd.Series({"Other": vc.iloc[top_n:].sum()})])
    fig = go.Figure(go.Pie(labels=top.index, values=top.values, hole=0.55, marker=dict(colors=CATEGORICAL_PALETTE)))
    fig.update_layout(title=f"Share of {col}")
    return _style(fig, show_legend=True)


def build_dashboard_figures(df: pd.DataFrame, col_types: dict, max_numeric: int = 6, max_categorical: int = 4) -> list:
    """Return a list of (section_title, figure) pairs for the dashboard tab.
    Every chart is built through `safe_chart` — a bad column is skipped,
    not a crash."""
    figures: list = []

    numeric = col_types.get("numeric", [])
    categorical = col_types.get("categorical", [])
    datetime_cols = col_types.get("datetime", [])

    # Time series first — the most informative view when dates are present.
    if datetime_cols and numeric:
        date_col = datetime_cols[0]
        for value_col in numeric[:3]:
            fig = safe_chart(time_series_line, df, date_col, value_col)
            if fig:
                figures.append(("Trend", fig))
        fig = safe_chart(area_chart, df, date_col, numeric[0])
        if fig:
            figures.append(("Cumulative trend", fig))

    if len(numeric) >= 2:
        fig = safe_chart(scatter, df, numeric[0], numeric[1])
        if fig:
            figures.append(("Relationship", fig))

    if numeric and categorical:
        fig = safe_chart(box_plot, df, numeric[0], categorical[0])
        if fig:
            figures.append(("Spread by category", fig))

    for col in numeric[:max_numeric]:
        fig = safe_chart(histogram, df, col)
        if fig:
            figures.append(("Distribution", fig))

    for col in categorical[:max_categorical]:
        n_unique = df[col].nunique(dropna=True)
        if n_unique < 2:
            continue
        fig = safe_chart(bar_top_categories, df, col)
        if fig:
            figures.append(("Top categories", fig))
        if n_unique <= 8:
            fig = safe_chart(donut_chart, df, col)
            if fig:
                figures.append(("Share", fig))

    return figures
