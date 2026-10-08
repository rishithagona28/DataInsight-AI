"""
Chat agent: answers questions about the loaded dataframe by writing and
running real pandas/numpy/Plotly code against it (a tool-use loop against
any OpenAI-compatible chat completions endpoint — Gemini by default),
rather than guessing from a text description.
"""

from __future__ import annotations

import ast
import builtins as _builtins_module
import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import predict

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are Ana, a data analyst assistant. You answer questions about a "
    "pandas DataFrame called `df` that is already loaded — you never see "
    "the raw data directly, only what your tool calls return. To answer a "
    "question, call the run_python tool with short pandas/numpy code. "
    "Assign whatever you want to see to a variable named `result` (a "
    "number, string, Series, DataFrame, or a Plotly figure built with `px` "
    "or `go` — assign the figure itself, don't call .show()). It will be "
    "shown back to you and, for a figure, rendered as a chart for the "
    "user. Use print() for anything else you want visibility into. Keep "
    "code focused on one question at a time; you can call the tool more "
    "than once if the first attempt errors or needs refining. Once you "
    "have enough to answer, reply in plain language, citing the actual "
    "numbers you computed. Never fabricate a number you did not compute. "
    "When asked to 'create a chart' or 'show' something visually, build it "
    "with px/go and assign it to `result` rather than describing it in text. "
    "When asked to predict or forecast a metric for a future period (e.g. "
    "'forecast revenue for the next 6 months'), call the forecast_metric "
    "tool instead of writing your own trend code — it runs the same "
    "vetted forecasting model as the app's Predict tab and returns a real "
    "chart, so the answer is calculated, not improvised."
)

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python code against the pandas DataFrame `df` (pandas as "
            "pd, numpy as np, and Plotly as px/go are available). Assign your "
            "answer to a variable named `result` — a number, string, Series, "
            "DataFrame, or a Plotly Figure. Returns whatever was printed plus "
            "a text preview of `result`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
}

FORECAST_TOOL = {
    "type": "function",
    "function": {
        "name": "forecast_metric",
        "description": (
            "Forecast a numeric column forward in time using a date column "
            "already in `df`. Use this for any 'predict/forecast X for the "
            "next N months' style question instead of run_python — it calls "
            "the same trend model as the Predict tab and returns a chart."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {"type": "string", "description": "Name of the date/time column in df."},
                "value_column": {"type": "string", "description": "Name of the numeric column to forecast."},
                "horizon_months": {"type": "number", "description": "How many months ahead to forecast, e.g. 1, 3, or 6."},
            },
            "required": ["date_column", "value_column", "horizon_months"],
        },
    },
}

# A lightweight safeguard (not a full sandbox) — this is a local, personal
# tool, so the goal is to stop obviously-wrong tool calls (opening files,
# importing os/subprocess, network access, escaping via dunder attribute
# tricks), not to defend against a determined malicious model.
_ALLOWED_BUILTIN_NAMES = [
    "len", "range", "min", "max", "sum", "sorted", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "round", "abs", "enumerate", "zip", "map", "filter",
    "print", "isinstance", "type", "Exception", "ValueError", "KeyError", "TypeError",
    "True", "False", "None",
]
_SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _ALLOWED_BUILTIN_NAMES if hasattr(_builtins_module, name)}

_FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "input", "breakpoint"}


def _validate_code(code: str) -> str | None:
    """Static pre-check before exec(): reject imports, dunder attribute
    access (the standard `().__class__.__bases__...` sandbox-escape
    pattern), and references to names that could reach the filesystem or
    process even though they're absent from the restricted builtins.
    Returns an error message, or None if the code looks safe to run."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Imports are not allowed — pd, np, px, and go are already available."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"Access to '{node.attr}' is not allowed."
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"'{node.id}' is not allowed."
    return None


def run_python_tool(code: str, df: pd.DataFrame) -> dict:
    """Execute model-written code in a restricted namespace. Returns a dict
    with 'text' (what to send back to the model) and 'result' (the raw
    Python object, e.g. a DataFrame or Plotly Figure, for Streamlit to
    render)."""
    violation = _validate_code(code)
    if violation:
        return {"text": f"Code rejected: {violation}", "result": None, "ok": False}

    local_vars: dict = {}
    safe_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "px": px, "go": go, "df": df.copy()}

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, local_vars)
    except Exception as exc:  # noqa: BLE001 — deliberately broad; fed back to the model
        return {"text": f"Error while running that code: {type(exc).__name__}: {exc}", "result": None, "ok": False}

    printed = stdout.getvalue()
    result = local_vars.get("result")

    if isinstance(result, pd.DataFrame):
        preview = result.head(20).to_string() if len(result) else "(empty DataFrame)"
        text = f"{printed}\nresult (DataFrame, {result.shape[0]} rows x {result.shape[1]} cols, first 20 shown):\n{preview}"
    elif isinstance(result, pd.Series):
        preview = result.head(20).to_string()
        text = f"{printed}\nresult (Series, {len(result)} entries, first 20 shown):\n{preview}"
    elif isinstance(result, go.Figure):
        title = (result.layout.title.text if result.layout.title else None) or "untitled"
        text = f"{printed}\nresult: a chart was created (\"{title}\") and will be shown to the user."
    elif result is not None:
        text = f"{printed}\nresult: {result!r}"
    else:
        text = printed or "(no output — set a `result` variable or print something)"

    return {"text": text.strip(), "result": result, "ok": True}


def forecast_metric_tool(date_column: str, value_column: str, horizon_months: float, df: pd.DataFrame) -> dict:
    """Handler for the forecast_metric tool — runs the same
    predict.forecast_time_series the Predict tab uses, rather than
    trusting the model to write correct trend-fitting code from scratch,
    and returns the actual-vs-forecast comparison as a chart."""
    if date_column not in df.columns or value_column not in df.columns:
        available = ", ".join(df.columns)
        return {"text": f"Column not found. Available columns: {available}", "result": None, "ok": False}

    try:
        result = predict.forecast_time_series(df, date_column, value_column, horizon_months=horizon_months)
    except Exception as exc:
        return {"text": f"Couldn't forecast {value_column}: {exc}", "result": None, "ok": False}

    fig = go.Figure()
    fig.add_scatter(x=result["actual"]["date"], y=result["actual"]["value"], mode="lines", name="Actual")
    fig.add_scatter(x=result["forecast"]["date"], y=result["forecast"]["value"], mode="lines", name="Forecast", line=dict(dash="dash"))
    fig.update_layout(title=f"{value_column}: actual vs. forecast ({horizon_months} month horizon)")

    growth = result["growth_pct"]
    growth_txt = f"{growth:+.1f}%" if growth is not None else "not computable (last actual value was 0)"
    end_value = float(result["forecast"]["value"].iloc[-1])
    text = (
        f"Forecasted {value_column} {horizon_months} month(s) ahead "
        f"({result['periods']} period(s) at detected frequency '{result['freq_detected']}'). "
        f"Expected change vs. the last actual value: {growth_txt}. "
        f"Forecast end value: {end_value:.2f}. A chart is shown to the user."
    )
    return {"text": text, "result": fig, "ok": True}


def build_dataset_context(df: pd.DataFrame, col_types: dict, quality_issues: list | None = None) -> str:
    """A short, static block describing the loaded dataframe's shape and
    columns (not the rows) — sent once per session as part of the cached
    system prompt instead of the model having to spend a turn discovering
    columns, and instead of re-describing the dataset on every turn."""
    lines = [f"The loaded `df` has {df.shape[0]} rows and {df.shape[1]} columns.", "Columns (name: detected type):"]
    kind_by_col = {c: kind for kind, cols in col_types.items() for c in cols}
    for col in df.columns:
        lines.append(f"- {col}: {kind_by_col.get(col, 'unknown')} ({df[col].dtype})")
    if quality_issues:
        lines.append(f"{len(quality_issues)} data-quality issue(s) were already flagged programmatically (missing values, outliers, duplicates, etc.) — you can query `df` directly for the specifics if relevant.")
    return "\n".join(lines)


def _dispatch_tool_call(name: str, arguments: dict, df: pd.DataFrame) -> tuple[str, dict]:
    """Run one tool call by name. Returns (code_repr, outcome) — code_repr
    is what's shown to the user as the 'code' for this step."""
    if name == "forecast_metric":
        code_repr = f"forecast_metric(date_column={arguments.get('date_column')!r}, value_column={arguments.get('value_column')!r}, horizon_months={arguments.get('horizon_months')!r})"
        outcome = forecast_metric_tool(
            arguments.get("date_column", ""),
            arguments.get("value_column", ""),
            arguments.get("horizon_months", 3),
            df,
        )
    else:
        code_repr = arguments.get("code", "")
        outcome = run_python_tool(code_repr, df)
    return code_repr, outcome


def ask(client, df: pd.DataFrame, history: list, dataset_context: str = "", model: str | None = None) -> dict:
    """Run the tool-use loop for the latest user turn already appended to
    `history`, against any OpenAI-compatible chat completions endpoint
    (Gemini by default — see app.py's get_llm_client). Returns
    {'history': updated_history, 'final_text': str, 'artifacts': [
    {'code':..., 'result':...}, ... ]}."""
    messages = list(history)
    artifacts: list = []

    system_text = f"{SYSTEM_PROMPT}\n\n{dataset_context}" if dataset_context else SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": system_text}] + messages

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            max_tokens=1500,
            tools=[RUN_PYTHON_TOOL, FORECAST_TOOL],
            messages=full_messages,
        )

        message = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            serialized_calls = []
            for tc in tool_calls:
                entry = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                # Gemini (and possibly other providers) attaches an opaque
                # per-call signature under extra_content that must be
                # echoed back verbatim on the next turn, or it rejects the
                # request — round-trip it if present, no-op otherwise.
                extra = getattr(tc, "extra_content", None)
                if extra:
                    entry["extra_content"] = extra
                serialized_calls.append(entry)
            assistant_msg["tool_calls"] = serialized_calls
        messages.append(assistant_msg)
        full_messages.append(assistant_msg)

        if not tool_calls:
            return {"history": messages, "final_text": message.content or "", "artifacts": artifacts}

        for tc in tool_calls:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {}
            code_repr, outcome = _dispatch_tool_call(tc.function.name, arguments, df)
            artifacts.append({"code": code_repr, "result": outcome["result"]})
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": outcome["text"][:4000]}
            messages.append(tool_msg)
            full_messages.append(tool_msg)

    return {
        "history": messages,
        "final_text": "I wasn't able to finish within the step limit — try breaking your question into a smaller one.",
        "artifacts": artifacts,
    }
