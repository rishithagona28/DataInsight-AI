"""
Client-ready dashboard, built automatically from any CSV.

A human analyst doesn't chart every column; they find the *story*:
"how much (measure), over time (date), split by what (dimension)".
This module does the same:

  1. detect_roles()  finds the date column, the measures (revenue, units, ...)
                     and the dimensions (region, product, ...)
  2. render()        draws a presentation-style dashboard:
                     filter bar -> KPI cards with trend lines -> trend chart + donut
                     -> clickable ranking bar + breakdown table -> key takeaways

Every number shown is computed with pandas (no AI involved), so it's exact.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import eda
from .predict import _LEVEL_NAME

# Column names that usually hold business amounts, in order of how much a client cares
_MONEY_HINT = re.compile(r"revenue|sales|amount|profit|income|gmv|turnover|total|value|spend|cost|fees?|margin|price", re.I)
_QTY_HINT = re.compile(r"qty|quantity|units|orders|volume|count", re.I)
# Columns that usually make good "break down by" groupings
_DIM_HINT = re.compile(r"region|category|segment|department|product|type|channel|city|state|country|store|branch|team", re.I)
# Numeric columns that are really labels, not amounts (a "year" column shouldn't be summed)
_NOT_A_MEASURE = re.compile(r"(^|_)(year|month|day|week|hour|zip|pin|pincode|postal|code|phone|mobile|lat|latitude|lon|lng|longitude)($|_)", re.I)

DATE_PRESETS = {"All time": None, "Last 12 months": 365, "Last 6 months": 182, "Last 3 months": 91, "Last 30 days": 30}


# =========================================================================== roles
def detect_roles(df: pd.DataFrame, col_types: dict) -> dict:
    id_like = set(eda.suggest_id_columns(df, col_types))
    measures = [
        c for c in col_types.get("numeric", [])
        if c not in id_like and not _NOT_A_MEASURE.search(str(c)) and df[c].nunique(dropna=True) > 1
    ]
    measures.sort(key=lambda c: 0 if _MONEY_HINT.search(str(c)) else 1 if _QTY_HINT.search(str(c)) else 2)

    dims = [c for c in col_types.get("categorical", []) + col_types.get("boolean", [])
            if 2 <= df[c].nunique(dropna=True) <= 40]
    dims.sort(key=lambda c: (not 2 <= df[c].nunique(dropna=True) <= 12, not _DIM_HINT.search(str(c)), df[c].nunique(dropna=True)))

    dates = [c for c in col_types.get("datetime", []) if df[c].nunique(dropna=True) >= 2]
    return {"measures": measures, "dimensions": dims, "date": dates[0] if dates else None}


def aggregation_for(measure: str) -> str:
    """Amounts are added up (revenue per region); levels are averaged (price per region)."""
    return "mean" if _LEVEL_NAME.search(str(measure)) else "sum"


# =========================================================================== formatting
def fmt(x, compact: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x)
    if compact:
        for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(x) >= div:
                return f"{x / div:,.1f}{suffix}"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def pct_change(now: float, before: float):
    if before is None or now is None or not before or np.isnan(before):
        return None
    return (now - before) / abs(before) * 100


def _style(fig: go.Figure, height: int = 320) -> go.Figure:
    """One consistent, clean look for every dashboard chart (colors come from the app theme)."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=13),
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False, title=None)
    return fig


# =========================================================================== aggregation helpers
def _agg(series: pd.Series, how: str) -> float:
    return float(series.sum() if how == "sum" else series.mean()) if len(series) else float("nan")


def _period_rule(dates: pd.Series) -> tuple[str, str]:
    span = (dates.max() - dates.min()).days if len(dates) else 0
    if span >= 180:
        return "MS", "month"
    if span >= 56:
        return "W-MON", "week"
    return "D", "day"


def _trend(df: pd.DataFrame, date_col: str, measure: str | None, how: str, rule: str, by: str | None = None) -> pd.DataFrame:
    """Measure (or row count) per period, optionally split by a dimension."""
    d = df.dropna(subset=[date_col])
    keys = [pd.Grouper(key=date_col, freq=rule)] + ([by] if by else [])
    if measure is None:
        out = d.groupby(keys, observed=True).size().rename("value")
    else:
        out = d.groupby(keys, observed=True)[measure].agg(how).rename("value")
    return out.reset_index()


# =========================================================================== charts
def trend_chart(tdf: pd.DataFrame, date_col: str, by: str | None, label: str, partial_first: bool, partial_last: bool) -> go.Figure:
    fig = go.Figure()
    if by:
        for name, g in tdf.groupby(by, observed=True):
            fig.add_scatter(x=g[date_col], y=g["value"], mode="lines+markers", name=str(name),
                            line=dict(width=2.5, shape="spline", smoothing=0.6), marker=dict(size=5),
                            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}<br>{label}: %{{y:,.0f}}<extra></extra>")
    else:
        fig.add_scatter(x=tdf[date_col], y=tdf["value"], mode="lines+markers", name=label, fill="tozeroy",
                        line=dict(width=3, shape="spline", smoothing=0.6), marker=dict(size=6),
                        hovertemplate=f"%{{x|%d %b %Y}}<br>{label}: %{{y:,.0f}}<extra></extra>")
        for flag, idx in ((partial_first, 0), (partial_last, -1)):
            if flag and len(tdf) >= 2:
                fig.add_annotation(x=tdf[date_col].iloc[idx], y=tdf["value"].iloc[idx], text="incomplete", showarrow=True,
                                   arrowhead=0, ay=-28, font=dict(size=11))
        fig.update_layout(showlegend=False)
    return _style(fig, 330)


def donut_chart(shares: pd.Series, center_label: str, center_value: str) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=shares.index.astype(str), values=shares.values, hole=0.64, sort=False,
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        marker=dict(line=dict(width=2, color="rgba(0,0,0,0)")),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<span style='font-size:22px'><b>{center_value}</b></span><br><span style='font-size:12px'>{center_label}</span>",
                       showarrow=False, x=0.5, y=0.5)
    fig = _style(fig, 330)
    fig.update_layout(legend=dict(orientation="v", x=1, y=0.5, yanchor="middle"))
    return fig


def ranking_chart(ranked: pd.Series, selected: list[str], label: str) -> go.Figure:
    names = ranked.index.astype(str)[::-1]
    values = ranked.values[::-1]
    opacity = [1.0 if (not selected or n in selected) else 0.35 for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker=dict(cornerradius=6, opacity=opacity),
        text=[fmt(v) for v in values], textposition="outside", cliponaxis=False,
        hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:,.0f}}<extra></extra>",
    ))
    fig = _style(fig, max(260, 34 * len(ranked) + 40))
    fig.update_xaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)")
    return fig


# =========================================================================== takeaways
def takeaways(view: pd.DataFrame, cur: pd.DataFrame, prev: pd.DataFrame | None, roles: dict, measure: str | None,
              how: str, dim: str | None, rank_dim: str | None, compare_text: str) -> list[str]:
    """Plain-English observations, computed (not generated), like an analyst's summary slide."""
    label = measure or "records"
    notes = []
    if prev is not None and len(prev):
        now = _agg(cur[measure], how) if measure else len(cur)
        before = _agg(prev[measure], how) if measure else len(prev)
        ch = pct_change(now, before)
        if ch is not None:
            word = "up" if ch >= 0 else "down"
            notes.append(f"**{label.capitalize()}** is **{word} {abs(ch):.1f}%** ({compare_text}: {fmt(before)} → {fmt(now)}).")
    if dim and measure:
        g = view.groupby(dim, observed=True)[measure].agg(how).sort_values(ascending=False)
        if len(g) >= 2:
            if how == "sum" and g.sum():
                notes.append(f"**{g.index[0]}** leads with **{g.iloc[0] / g.sum():.0%}** of total {label}; "
                             f"**{g.index[-1]}** is the smallest at {g.iloc[-1] / g.sum():.0%}.")
            else:
                notes.append(f"**{g.index[0]}** has the highest average {label} ({fmt(g.iloc[0])}), "
                             f"vs {fmt(g.iloc[-1])} for **{g.index[-1]}**.")
    if rank_dim and measure and how == "sum" and rank_dim != dim:
        g = view.groupby(rank_dim, observed=True)[measure].sum().sort_values(ascending=False)
        if len(g) >= 5 and g.sum():
            top_n = max(1, round(len(g) * 0.2))
            notes.append(f"The top {top_n} of {len(g)} **{rank_dim}** values bring in **{g.head(top_n).sum() / g.sum():.0%}** of {label}.")
    date_col = roles.get("date")
    if date_col and measure and how == "sum":
        rule, period = _period_rule(view[date_col].dropna())
        t = _trend(view, date_col, measure, how, rule)
        if len(t) >= 3:
            best = t.loc[t["value"].idxmax()]
            fmt_date = "%b %Y" if period == "month" else "%d %b %Y"
            notes.append(f"The best {period} was **{best[date_col]:{fmt_date}}** with {fmt(best['value'])} {label}.")
    return notes


# =========================================================================== main
def render(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """Draw the dashboard. Returns the filtered data (other sections can reuse it)."""
    roles = detect_roles(df, col_types)
    measures, dims, date_col = roles["measures"], roles["dimensions"], roles["date"]

    # ------------------------------------------------------------------ controls
    with st.container(border=True, key="card_controls"):
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        measure = c1.selectbox("Measure", measures, key="dash_measure") if measures else None
        if measure is None:
            c1.caption("No numeric amounts found: counting records.")
        dim = c2.selectbox("Break down by", dims, key="dash_dim") if dims else None
        rank_options = [d for d in dims if d != dim] or dims
        rank_dim = c3.selectbox("Rank", rank_options, key="dash_rank_dim") if rank_options else None

        f1, f2 = st.columns([1.6, 2])
        preset = None
        if date_col:
            span = (df[date_col].max() - df[date_col].min()).days
            options = [k for k, days in DATE_PRESETS.items() if days is None or days < span]
            preset = f1.segmented_control("Period", options, default="All time", key="dash_period")
        chosen = []
        if dim:
            values = df[dim].dropna().astype(str).value_counts().index.tolist()
            if len(values) <= 8:
                chosen = f2.pills(f"Filter {dim}", values, selection_mode="multi", key="dash_dim_filter")
            else:
                chosen = f2.multiselect(f"Filter {dim}", values, key="dash_dim_filter")

    how = aggregation_for(measure) if measure else "sum"
    label = measure or "records"

    # ------------------------------------------------------------------ filtering
    base = df
    if chosen:
        base = base[base[dim].astype(str).isin(chosen)]
    prev = None
    if date_col and preset and DATE_PRESETS.get(preset):
        end = df[date_col].max()
        start = end - pd.Timedelta(days=DATE_PRESETS[preset])
        prev = base[(base[date_col] > start - pd.Timedelta(days=DATE_PRESETS[preset])) & (base[date_col] <= start)]
        base = base[(base[date_col] > start) & (base[date_col] <= end)]
    mid = None
    all_time = date_col and not (preset and DATE_PRESETS.get(preset))
    if all_time:
        # "All time": % changes compare the recent half of the data with the earlier half
        mid = df[date_col].min() + (df[date_col].max() - df[date_col].min()) / 2
        prev = base[base[date_col] <= mid]

    # Cross-filter: bars clicked in the ranking chart filter everything else
    sel_key = f"dash_rank_click_{st.session_state.get('dash_click_version', 0)}"
    selected: list[str] = []
    event = st.session_state.get(sel_key)
    selection = (event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)) if event else None
    if selection and rank_dim:
        selected = sorted({str(p.get("y")) for p in selection.get("points", []) if p.get("y") is not None})
    view = base[base[rank_dim].astype(str).isin(selected)] if selected else base
    if selected and prev is not None:
        prev = prev[prev[rank_dim].astype(str).isin(selected)]

    if view.empty:
        st.warning("No data matches these filters. Try a longer period or clear a filter.", icon=":material/filter_alt_off:")
        return view

    if selected:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(f"Showing {rank_dim}: {', '.join(selected)}", icon=":material/filter_alt:", color="blue")
            if st.button("Clear selection", icon=":material/close:", type="tertiary"):
                st.session_state["dash_click_version"] = st.session_state.get("dash_click_version", 0) + 1
                st.rerun()

    # ------------------------------------------------------------------ KPI cards
    rule, period = _period_rule(view[date_col].dropna()) if date_col else (None, None)
    spark_value = _trend(view, date_col, measure, how, rule)["value"].tolist() if date_col else None
    spark_count = _trend(view, date_col, None, how, rule)["value"].tolist() if date_col else None
    has_prev = prev is not None and len(prev) > 0

    # What the % change compares: the selected period vs the one before it, or for
    # "All time" the recent half vs the earlier half (the card still shows the full total).
    cur = view[view[date_col] > mid] if mid is not None else view
    compare_text = "vs earlier half" if mid is not None else "vs previous period"

    total_now = _agg(view[measure], how) if measure else len(view)
    avg_now = _agg(view[measure], "mean") if measure else None
    value_cur = (_agg(cur[measure], how) if measure else len(cur))
    value_prev = (_agg(prev[measure], how) if measure else len(prev)) if has_prev else None

    def delta(now, before):
        ch = pct_change(now, before)
        return f"{ch:+.1f}%" if ch is not None else None

    with st.container(horizontal=True, key="card_kpis"):
        st.metric(f"{'Total' if how == 'sum' else 'Average'} {label}", fmt(total_now), delta(value_cur, value_prev),
                  border=True, chart_data=spark_value, chart_type="area", delta_description=compare_text if has_prev else None,
                  help=f"Exact: {fmt(total_now, compact=False)}")
        st.metric("Records", f"{len(view):,}", delta(len(cur), len(prev) if has_prev else None), border=True,
                  chart_data=spark_count, chart_type="bar", delta_description=compare_text if has_prev else None)
        if measure and how == "sum":
            st.metric(f"Average {label} per record", fmt(avg_now),
                      delta(_agg(cur[measure], "mean"), _agg(prev[measure], "mean") if has_prev else None), border=True,
                      delta_description=compare_text if has_prev else None)
        if dim and measure:
            g = view.groupby(dim, observed=True)[measure].agg(how).sort_values(ascending=False)
            if len(g):
                share = f"{g.iloc[0] / g.sum():.0%} of total" if how == "sum" and g.sum() else f"avg {fmt(g.iloc[0])}"
                st.metric(f"Top {dim}", str(g.index[0]), share, delta_color="off", delta_arrow="off", border=True)

    # ------------------------------------------------------------------ trend + donut
    left, right = st.columns([1.7, 1])
    with left:
        with st.container(border=True, key="card_trend"):
            head, toggle = st.columns([2, 1.3], vertical_alignment="center")
            head.markdown(f"**{label.capitalize()} over time**" if date_col else f"**{label.capitalize()}**")
            split = None
            if date_col and dim and view[dim].nunique() <= 8:
                mode = toggle.segmented_control("View", ["Total", f"By {dim}"], default="Total", key="dash_trend_mode",
                                                label_visibility="collapsed")
                split = dim if mode == f"By {dim}" else None
            if date_col:
                tdf = _trend(view, date_col, measure, how, rule, by=split)
                first, last = view[date_col].min(), view[date_col].max()
                partial_first = period == "month" and first.day > 3
                partial_last = period == "month" and last.day < 28
                st.plotly_chart(trend_chart(tdf, date_col, split, label, partial_first, partial_last), key="dash_trend_chart",
                                config={"displaylogo": False})
                edges = [w for w, f in (("first", partial_first), ("last", partial_last)) if f]
                st.caption(f"{'Total' if how == 'sum' else 'Average'} {label} per {period}."
                           + (f" The {' and '.join(edges)} month{'s are' if len(edges) > 1 else ' is'} only partly covered by the data." if edges else ""))
            else:
                st.caption("No date column found, so there's no trend over time.")
    with right:
        with st.container(border=True, key="card_donut"):
            if dim:
                st.markdown(f"**Share by {dim}**")
                g = (view.groupby(dim, observed=True)[measure].agg(how) if measure else view[dim].value_counts()).sort_values(ascending=False)
                if how == "sum" or not measure:
                    shares = g.head(6)
                    if len(g) > 6:
                        shares = pd.concat([shares, pd.Series({"Other": g.iloc[6:].sum()})])
                    st.plotly_chart(donut_chart(shares, f"total {label}", fmt(g.sum())), key="dash_donut",
                                    config={"displaylogo": False})
                    st.caption("Click a name in the legend to hide or show it.")
                else:
                    st.caption(f"{label} is an average, so shares don't add up; see the ranking instead.")
            else:
                st.caption("No category column to split by.")

    # ------------------------------------------------------------------ ranking + breakdown
    left, right = st.columns([1.2, 1])
    with left:
        with st.container(border=True, key="card_rank"):
            if rank_dim:
                st.markdown(f"**Top {rank_dim} by {label}**")
                g = (base.groupby(rank_dim, observed=True)[measure].agg(how) if measure else base[rank_dim].value_counts())
                ranked = g.sort_values(ascending=False).head(10)
                st.plotly_chart(ranking_chart(ranked, selected, label), key=sel_key, on_select="rerun",
                                selection_mode="points", config={"displaylogo": False})
                st.caption("Click a bar to filter the whole dashboard to it.")
    with right:
        with st.container(border=True, key="card_table"):
            if dim:
                st.markdown(f"**{dim.capitalize()} breakdown**")
                grp = view.groupby(dim, observed=True)
                table = pd.DataFrame({"Records": grp.size()})
                if measure:
                    table[label] = grp[measure].agg(how)
                    if how == "sum":
                        table["Share"] = table[label] / table[label].sum()
                table = table.sort_values(label if measure else "Records", ascending=False).reset_index()
                cfg = {"Share": st.column_config.ProgressColumn("Share", format="percent", min_value=0, max_value=1)}
                if measure:
                    cfg[label] = st.column_config.NumberColumn(label, format="compact")
                st.dataframe(table, hide_index=True, column_config=cfg, height=min(36 * (len(table) + 1) + 4, 330))

    # ------------------------------------------------------------------ takeaways
    notes = takeaways(view, cur, prev if has_prev else None, roles, measure, how, dim, rank_dim, compare_text)
    if notes:
        with st.container(border=True, key="card_takeaways"):
            st.markdown(":material/lightbulb: **Key takeaways**")
            st.markdown("\n".join(f"- {n}" for n in notes))

    st.download_button("Download this view (CSV)", view.to_csv(index=False).encode(), file_name="dashboard_view.csv",
                       mime="text/csv", icon=":material/download:")
    return view
