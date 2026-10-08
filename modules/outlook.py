"""
Future-focused business outlook.

Two layers, deliberately separated:

  1. compute_signals()  -- CODE calculates every forward-looking number:
       forecast for the next months (with a likely range), momentum (last 3
       periods vs the 3 before), year-over-year change, seasonal peaks,
       rising/declining segments, concentration risk, volatility.
  2. generate_brief()   -- the LLM turns those numbers into a short business brief
       (outlook, opportunities, risks, actions). It is told to use ONLY the
       numbers it is given, and must answer in a fixed JSON structure that we
       validate with Pydantic ("structured output"), so the app can render it
       as cards and never has to parse free text.

The LLM explains; it never calculates. That's what keeps the brief trustworthy.
"""
from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from . import dashboard, llm, predict

HORIZON_MONTHS = 3
_MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"]


# =========================================================================== 1. signals (pure code)
def _pct(now: float, before: float):
    if before is None or not np.isfinite(before) or before == 0:
        return None
    return round((now - before) / abs(before) * 100, 1)


def compute_signals(df: pd.DataFrame, col_types: dict) -> dict:
    roles = dashboard.detect_roles(df, col_types)
    measure = roles["measures"][0] if roles["measures"] else None
    date_col = roles["date"]
    dim = roles["dimensions"][0] if roles["dimensions"] else None
    how = dashboard.aggregation_for(measure) if measure else "sum"
    sig: dict = {"measure": measure, "date_col": date_col, "dimension": dim, "how": how,
                 "rows": len(df), "caveats": []}

    if measure is None:
        sig["caveats"].append("No numeric amount column was found, so there is nothing to forecast.")
        return sig

    # Concentration: how dependent is the business on one group?
    if dim and how == "sum":
        shares = df.groupby(dim, observed=True)[measure].sum().sort_values(ascending=False)
        total = shares.sum()
        if total and len(shares) >= 2:
            sig["concentration"] = {"top_group": str(shares.index[0]), "top_share_pct": round(shares.iloc[0] / total * 100, 1),
                                    "groups": len(shares)}

    if date_col is None:
        sig["caveats"].append("There is no date column, so trends and forecasts can't be calculated.")
        return sig

    series, rule, agg, period, dropped = predict.build_period_series(df, date_col, measure)
    sig.update(period=period, periods_of_history=len(series),
               history_start=f"{series.index.min():%b %Y}", history_end=f"{series.index.max():%b %Y}")
    if dropped:
        sig["caveats"].append(f"{dropped} incomplete {period}(s) at the edges of the data were left out of the trend.")
    if len(series) < 6:
        sig["caveats"].append(f"Only {len(series)} {period}s of history: too short for a reliable forecast.")
        return sig

    # Momentum: last 3 periods vs the 3 before them
    if len(series) >= 6:
        sig["momentum_pct"] = _pct(series.iloc[-3:].sum(), series.iloc[-6:-3].sum())

    # Year over year (monthly data with >= 15 months)
    if period == "month" and len(series) >= 15:
        sig["yoy_pct"] = _pct(series.iloc[-3:].sum(), series.iloc[-15:-12].sum())

    # Seasonality: average by calendar month (needs a full year)
    if period == "month" and len(series) >= 12:
        by_month = series.groupby(series.index.month).mean()
        sig["seasonal_peak"] = _MONTH_NAMES[int(by_month.idxmax()) - 1]
        sig["seasonal_low"] = _MONTH_NAMES[int(by_month.idxmin()) - 1]
        sig["seasonal_swing_pct"] = _pct(by_month.max(), by_month.min())

    # Volatility: how bumpy the periods are (coefficient of variation)
    if series.mean():
        sig["volatility_pct"] = round(float(series.std() / series.mean() * 100), 1)

    # Forecast with a likely range
    try:
        fc = predict.forecast_time_series(df, date_col, measure, horizon_months=HORIZON_MONTHS)
        f = fc["forecast"]
        sig["forecast"] = {
            "periods": fc["periods"], "period": fc.get("period_name") or period,
            "next_total": float(fc["forecast_total"]), "recent_total": float(fc["recent_total"]),
            "change_pct": fc["growth_pct"],
            "low_total": float(f["low"].sum()), "high_total": float(f["high"].sum()),
            "aggregation": fc.get("aggregation"),
        }
        sig["_forecast_frame"] = fc  # for the chart (not sent to the LLM)
    except Exception as exc:  # noqa: BLE001 — a failed forecast shouldn't break the page
        sig["caveats"].append(f"The forecast couldn't be calculated ({exc}).")

    # Segments: which groups are speeding up or slowing down?
    if dim and df[dim].nunique() <= 25:
        d = df.dropna(subset=[date_col])
        grid = d.groupby([pd.Grouper(key=date_col, freq=rule), dim], observed=True)[measure].agg(how).unstack(dim)
        grid = grid.reindex(series.index)  # same complete periods as the main series
        if how == "sum":
            grid = grid.fillna(0)
        segs = []
        for g in grid.columns:
            s = grid[g].dropna()
            if len(s) >= 6 and s.iloc[-6:-3].sum() > 0:
                segs.append({"group": str(g), "momentum_pct": _pct(s.iloc[-3:].sum(), s.iloc[-6:-3].sum()),
                             "recent_total": float(s.iloc[-3:].sum())})
        segs.sort(key=lambda x: x["momentum_pct"] if x["momentum_pct"] is not None else 0, reverse=True)
        sig["segments"] = segs
    return sig


def _f(x) -> str:
    return dashboard.fmt(x)


def signals_text(sig: dict) -> str:
    """The signals as plain text for the LLM: the ONLY facts it may use."""
    m, period = sig.get("measure"), sig.get("period", "period")
    lines = [f"Main measure: {m} ({'totals' if sig.get('how') == 'sum' else 'averages'}); {sig.get('rows'):,} rows."]
    if sig.get("periods_of_history"):
        lines.append(f"History: {sig['periods_of_history']} complete {period}s, {sig['history_start']} to {sig['history_end']}.")
    fc = sig.get("forecast")
    if fc:
        lines.append(f"FORECAST next {fc['periods']} {fc['period']}(s): {_f(fc['next_total'])} "
                     f"(likely range {_f(fc['low_total'])} to {_f(fc['high_total'])}), vs {_f(fc['recent_total'])} in the last "
                     f"{fc['periods']} {fc['period']}(s): {fc['change_pct']:+.1f}%.")
    for key, text in (("momentum_pct", f"Momentum: last 3 {period}s vs the 3 before"), ("yoy_pct", "Year over year: last 3 months vs same months last year")):
        if sig.get(key) is not None:
            lines.append(f"{text}: {sig[key]:+.1f}%.")
    if sig.get("seasonal_peak"):
        lines.append(f"Seasonality: strongest month on average is {sig['seasonal_peak']}, weakest is {sig['seasonal_low']} "
                     f"(peak is {sig['seasonal_swing_pct']:+.1f}% above the low).")
    if sig.get("volatility_pct") is not None:
        lines.append(f"Volatility: {period}ly values typically vary by about {sig['volatility_pct']:.0f}% around the average.")
    conc = sig.get("concentration")
    if conc:
        lines.append(f"Concentration: {conc['top_group']} is {conc['top_share_pct']:.1f}% of total {m} across {conc['groups']} {sig['dimension']} groups.")
    segs = sig.get("segments") or []
    if segs:
        lines.append(f"Segment momentum by {sig['dimension']} (last 3 {period}s vs 3 before): "
                     + "; ".join(f"{s['group']} {s['momentum_pct']:+.1f}%" for s in segs if s["momentum_pct"] is not None))
    for c in sig.get("caveats", []):
        lines.append(f"Caveat: {c}")
    return "\n".join(lines)


# =========================================================================== 2. AI brief (structured output)
class Action(BaseModel):
    title: str = Field(description="Short imperative, e.g. 'Stock up before the June peak'")
    why: str = Field(description="One sentence citing the numbers that justify it")
    impact: Literal["high", "medium", "low"]


class OutlookBrief(BaseModel):
    headline: str = Field(description="One sentence: the single most important thing about what's ahead")
    outlook: list[str] = Field(description="2-4 bullets on what is likely to happen next, with numbers")
    opportunities: list[str] = Field(description="1-3 bullets: segments or periods to lean into")
    risks: list[str] = Field(description="1-3 bullets: what could go wrong or needs watching")
    actions: list[Action] = Field(description="Exactly 3 concrete recommended actions")
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str = Field(description="One short sentence on why")


SYSTEM_PROMPT = """You are a senior business analyst writing a short, forward-looking brief for a busy manager.
You are given SIGNALS that were calculated exactly from their data. Rules:
- Use ONLY numbers that appear in the SIGNALS. Never invent, round differently, or extrapolate new numbers.
- Focus on the FUTURE: what is likely to happen next, what to lean into, what to watch, what to do.
- Plain business English. No jargon like "coefficient of variation" or "Holt's method".
- Do not add a currency symbol unless the signals use one.
- Be honest about uncertainty: base confidence on history length, volatility and the forecast range.
- Recommended actions must be specific and tied to a signal (a segment, a month, a risk)."""


def generate_brief(client, sig: dict) -> OutlookBrief:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "SIGNALS:\n" + signals_text(sig)}]
    schema = {"type": "json_schema", "json_schema": {"name": "outlook_brief", "schema": OutlookBrief.model_json_schema()}}
    last_error = None
    for _ in range(2):  # one retry if the model's JSON doesn't validate
        response, _model = llm.complete(client, messages=messages, max_tokens=1600, response_format=schema)
        raw = response.choices[0].message.content or ""
        try:
            return OutlookBrief.model_validate_json(raw)
        except ValidationError as exc:
            last_error = exc
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:  # tolerate text around the JSON
                try:
                    return OutlookBrief.model_validate(json.loads(raw[start:end + 1]))
                except (ValidationError, json.JSONDecodeError) as exc2:
                    last_error = exc2
    raise ValueError(f"The AI's answer wasn't in the expected format ({last_error}).")


def brief_to_markdown(brief: OutlookBrief) -> str:
    """Plain-text version, used by the dashboard summary card and the HTML report."""
    parts = [f"**{brief.headline}**", "", "**Outlook**"] + [f"- {x}" for x in brief.outlook]
    parts += ["", "**Opportunities**"] + [f"- {x}" for x in brief.opportunities]
    parts += ["", "**Risks to watch**"] + [f"- {x}" for x in brief.risks]
    parts += ["", "**Recommended actions**"] + [f"- **{a.title}** ({a.impact} impact): {a.why}" for a in brief.actions]
    parts += ["", f"_Confidence: {brief.confidence}. {brief.confidence_reason}_"]
    return "\n".join(parts)
