"""
DataInsight AI's visual design layer: custom CSS on top of the Streamlit theme
(.streamlit/config.toml keeps the base colors and fonts).

Only stable hooks are targeted: data-testid attributes and the `st-key-<key>`
classes Streamlit adds to keyed containers. Every dashboard card uses a key
starting with "card_", so one rule styles them all.
"""
from __future__ import annotations

import html

import streamlit as st

_PALETTE = {
    "light": dict(bg="#f6f7fb", card="rgba(255,255,255,0.82)", border="rgba(15,23,42,0.08)",
                  shadow="rgba(15,23,42,0.07)", glow_a="rgba(42,120,214,0.16)", glow_b="rgba(124,92,255,0.13)",
                  muted="#5b6472", tab_bg="rgba(15,23,42,0.04)"),
    "dark": dict(bg="#0d0f14", card="rgba(24,27,36,0.78)", border="rgba(255,255,255,0.07)",
                 shadow="rgba(0,0,0,0.35)", glow_a="rgba(91,158,230,0.20)", glow_b="rgba(139,122,224,0.18)",
                 muted="#a3a9b6", tab_bg="rgba(255,255,255,0.05)"),
}


def inject() -> None:
    mode = "dark" if (st.context.theme.type == "dark") else "light"
    p = _PALETTE[mode]
    st.html(f"""
<style>
:root {{
  --ana-a: #2a78d6; --ana-b: #7c5cff; --ana-c: #1baf7a;
  --ana-card: {p['card']}; --ana-border: {p['border']}; --ana-shadow: {p['shadow']}; --ana-muted: {p['muted']};
}}

/* Soft, colourful background glow instead of a flat page */
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(900px 520px at 88% -8%, {p['glow_a']}, transparent 62%),
    radial-gradient(760px 460px at -8% 28%, {p['glow_b']}, transparent 60%),
    {p['bg']};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, {p['glow_a']}, transparent 40%), var(--ana-card);
  backdrop-filter: blur(10px);
}}

/* Cards: glassy surface, rounded corners, gentle lift on hover, fade-in */
[class*="st-key-card_"] {{
  background: var(--ana-card);
  border: 1px solid var(--ana-border) !important;
  border-radius: 18px !important;
  box-shadow: 0 1px 2px var(--ana-shadow), 0 10px 28px var(--ana-shadow);
  backdrop-filter: blur(8px);
  transition: transform .18s ease, box-shadow .18s ease;
  animation: anaFadeUp .45s ease both;
}}
[class*="st-key-card_"]:hover {{
  transform: translateY(-2px);
  box-shadow: 0 2px 4px var(--ana-shadow), 0 16px 36px var(--ana-shadow);
}}

/* KPI tiles: coloured accent bar, bold numbers */
[data-testid="stMetric"] {{
  background: var(--ana-card);
  border-radius: 16px;
  position: relative; overflow: hidden;
  box-shadow: 0 1px 2px var(--ana-shadow), 0 8px 22px var(--ana-shadow);
  transition: transform .18s ease;
  animation: anaFadeUp .45s ease both;
}}
[data-testid="stMetric"]:hover {{ transform: translateY(-2px); }}
[data-testid="stMetric"]::before {{
  content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
  background: linear-gradient(90deg, var(--ana-a), var(--ana-b));
}}
[data-testid="stMetricValue"] {{ font-weight: 800; letter-spacing: -0.02em; }}
[data-testid="stMetricLabel"] p {{ font-weight: 600; color: var(--ana-muted); text-transform: uppercase;
  font-size: 0.74rem; letter-spacing: 0.06em; }}

/* Tabs as rounded pills, the active one filled with the brand gradient */
[data-testid="stTabs"] [role="tablist"] {{ gap: 6px; flex-wrap: wrap; }}
[data-testid="stTab"] {{
  border-radius: 999px !important; padding: 6px 14px !important;
  background: {p['tab_bg']}; transition: background .15s ease, color .15s ease;
}}
[data-testid="stTab"][aria-selected="true"] {{
  background: linear-gradient(90deg, var(--ana-a), var(--ana-b)); color: #fff !important;
}}
[data-testid="stTab"][aria-selected="true"] * {{ color: #fff !important; }}

/* Hero banner */
.ana-hero {{
  position: relative; overflow: hidden; border-radius: 22px; padding: 26px 28px; margin: 4px 0 6px;
  color: #fff;
  background: linear-gradient(120deg, #1d4ed8 0%, #2a78d6 40%, #7c5cff 100%);
  box-shadow: 0 18px 40px rgba(42,120,214,0.28);
  animation: anaFadeUp .5s ease both;
}}
.ana-hero::after {{
  content: ""; position: absolute; right: -60px; top: -80px; width: 280px; height: 280px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.28), transparent 65%);
}}
.ana-hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; padding: 0; }}
.ana-hero p {{ margin: 6px 0 14px; opacity: .92; font-size: 1rem; max-width: 720px; }}
.ana-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.ana-chip {{ background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
  padding: 4px 12px; font-size: 0.82rem; font-weight: 600; backdrop-filter: blur(4px); }}

@keyframes anaFadeUp {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{
  [class*="st-key-card_"], [data-testid="stMetric"], .ana-hero {{ animation: none; transition: none; }}
}}
</style>
""")


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = "".join(f'<span class="ana-chip">{html.escape(c)}</span>' for c in (chips or []))
    st.html(f"""<div class="ana-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>
<div class="ana-chips">{chips_html}</div></div>""")
