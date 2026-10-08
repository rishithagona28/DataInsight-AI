# DataInsight AI

An AI-powered data analyst platform: upload a dataset and it profiles,
cleans, visualizes, models, and explains it — the routine work a data
analyst intern would do — with an LLM (Gemini by default, free) used for
interpretation and explanation, never for the actual calculations.

## Status

All core features are built, across a seven-tab workflow:

- **Data quality & cleaning** — programmatically detects missing values,
  duplicates, outliers (IQR), ID-like columns, and inconsistent text
  (e.g. `"USA"` vs `" usa"`), then proposes a named, explained fix for
  each — nothing is applied until you check it and click Apply. Every
  applied step is logged with a timestamp, and "Reset to original data"
  always gets you back to the untouched upload.
- **Overview & EDA** — shape, column types, cardinality, missing data,
  summary stats, duplicate rows, categorical breakdowns, and possible
  prediction-target columns.
- **Dashboard** — KPI cards, outlier badges, date/category/search filters
  scoped to the tab, click-to-drill-down on the top-categories chart, and
  auto-generated Plotly charts (correlation heatmap, trend, area,
  scatter, box, histogram, bar, donut) picked from the data's shape.
- **Predict** — forecasts a numeric metric 1, 3, or 6 months ahead (or a
  custom horizon), showing the expected % change alongside the trend
  chart. For "predict a column," a single well-rounded model (Random
  Forest) trains by default with a plain-language result — check "Compare
  more models" to reveal the full multi-model comparison (Logistic/Linear
  Regression, Decision Tree, Random Forest, Gradient Boosting) with the
  metrics appropriate to the task (accuracy/precision/recall/F1/ROC-AUC/
  confusion matrix for classification; MAE/MSE/RMSE/R²/adjusted R² for
  regression), per-model training time, cross-validation, and
  permutation-importance explainability — no synthetic "overall score."
- **AI Insights** — sends a compact statistical summary and the
  programmatically detected data-quality issues (never the raw rows) to
  the LLM and gets back a plain-language analysis.
- **Chat with your data** — a chat agent that answers questions by
  writing and executing real pandas/Plotly code against the loaded
  dataframe — including building charts on request, and a dedicated
  forecast tool for "predict/forecast X for the next N months" questions
  (reuses the same forecasting function as the Predict tab) — so answers
  (and numbers) are computed, not guessed.
- **Report** — compiles everything already computed this session
  (profiling, cleaning log, AI insights, model comparison) into one
  downloadable, self-contained HTML report with embedded interactive
  charts. Generating it costs no extra LLM tokens or model training.

## Setup (one-time)

1. Make sure Python 3.10+ is installed (`python3 --version` in a terminal).
2. Open this folder in VS Code.
3. Open a terminal in VS Code (Terminal menu → New Terminal).
4. Create a virtual environment (skip if you already have one):
   ```
   python3 -m venv venv
   ```
5. Activate it:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
6. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
7. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to your real
   key — get a **free** one (no credit card) at
   https://aistudio.google.com/apikey. The app works fine without it, but
   **AI Insights** and **Chat** stay disabled until a key is set. The key
   is only ever read server-side via `os.environ`/`.env` — it is never
   sent to the browser or exposed in any response.

   The app talks to Gemini through its OpenAI-compatible endpoint (the
   `openai` package pointed at Google's URL — see `get_llm_client()` in
   `app.py`), so swapping to another OpenAI-compatible provider (Groq,
   OpenRouter, a local Ollama server) later only means changing
   `GEMINI_API_KEY`/`GEMINI_MODEL`/`LLM_BASE_URL` in `.env` — no code
   changes.

## Run the app

```
streamlit run app.py
```

This opens the app in your browser at http://localhost:8501 — upload any
CSV in the sidebar to get started.

## Project layout

```
app.py                  # Streamlit UI — seven tabs, wires the modules together
modules/
  eda.py                 # profiling: types, cardinality, outliers, quality report
  cleaning.py             # cleaning recommendations (approval-gated, named transforms)
  charts.py               # auto-dashboard chart builders (Plotly)
  predict.py              # forecasting + multi-model comparison/evaluation/explainability
  insights.py              # compact summaries + LLM calls (dataset insights, model comparison explanation)
  chat_agent.py             # the code-executing chat agent (OpenAI-compatible tool use, sandboxed)
  report.py                # assembles an already-computed session into one HTML report
```

## Performance at larger dataset sizes

The app is built for practical, manageable CSV sizes rather than
unlimited scale, but it's been specifically fixed and tested against a
synthetic 300,000-row / 19MB file (see the caveats below):

- **Charts**: `scatter`/`box_plot`/`time_series_line`/`area_chart` used to
  embed every raw row into the Plotly JSON sent to the browser (3–6.6MB
  *per chart* at 300K rows — the actual cause of the browser tab
  freezing). They now bin, sample, or aggregate server-side before
  building the figure — the full 15-chart dashboard now sends under
  200KB total, at any dataset size. A bad column in any single chart is
  caught and skipped (`charts.safe_chart`) rather than crashing the tab.
- **Model training**: rows above 50,000 are sampled before training
  (stratified for classification), a datetime or very-high-cardinality
  column is automatically excluded from one-hot encoding instead of
  exploding into thousands of columns, and tree-ensemble sizes scale down
  for larger training sets. Each model trains inside its own `st.status`
  step, so a multi-model comparison shows real incremental progress
  instead of one long silent block.
- Measured on the 300K-row file: full app render (all 7 tabs, dashboard
  charts, cleaning recommendations) in under 10 seconds; the default
  single-model Predict flow (sampled to 40,000 training rows) in under 35
  seconds — versus the original bug, where a single Random Forest
  comparison alone ran past 2 minutes without finishing.

## Cost & performance notes

- The LLM never sees raw rows — only compact, structured summaries
  (shape, dtypes, stats, already-computed data-quality flags) built by
  `eda.py`/`insights.py`. All aggregation, statistics, model training,
  and evaluation happen in code, not in a prompt.
- Profiling, cardinality, the data-quality report, and model comparisons
  are cached (`st.cache_data`) so revisiting a tab or re-running the same
  comparison doesn't recompute or retrain.
- The dataset's shape/columns are sent once per session as grounding
  context for the chat agent instead of being re-described on every turn.

## Notes & limitations

- The forecast is a simple trend model (Holt's exponential smoothing) meant
  for quick exploration, not a statistically rigorous forecast — treat the
  shown uncertainty band as a rough guide, not a confidence interval.
- Model comparisons use default hyperparameters and a single train/test
  split — a fast first read on predictability and trade-offs, not tuned,
  production-ready models. ID-like columns are auto-suggested for
  exclusion; review the list before training.
- Cleaning recommendations are heuristics (median/mode imputation, IQR
  outlier capping, whitespace/case normalization, duplicate removal) —
  review each one's rationale before applying; nothing is automatic.
- The chat agent executes model-written code in a restricted namespace
  (no imports, no dunder-attribute access, no file/network/system access)
  and pre-validates code with an AST check before running it — a
  meaningful safeguard, but not a full security sandbox. This is meant
  for personal/local use, not for exposing to untrusted users.
