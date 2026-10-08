"""
One place for every LLM call in Ana, with automatic model fallback.

Why this exists: Gemini's free tier limits requests *per model, per day*
(the default model allowed only 20 requests/day, i.e. ~5-8 chat questions).
When a model hits its limit we don't fail; we move on to the next model in
the chain, each of which has its own separate daily quota. A model that ran
out is skipped until its quota resets (Google tells us when, in the error).

Configure the order in .env with GEMINI_MODELS=model-a,model-b,...
"""
from __future__ import annotations

import os
import re
import time

import openai

DEFAULT_CHAIN = [
    "gemini-3.5-flash",          # smart and fast
    "gemini-3.5-flash-lite",     # lighter models: quicker, separate quotas
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",          # newest, but a very small free daily quota
]

# model name -> unix time when it may be tried again. Shared by the whole
# process on purpose: the quota belongs to the API project, not to a user.
_unavailable_until: dict[str, float] = {}


class AllModelsBusy(RuntimeError):
    """Every model in the chain is out of quota (or unavailable) right now."""


def model_chain() -> list[str]:
    configured = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()]
    if configured:
        return configured
    preferred = os.environ.get("GEMINI_MODEL", "").strip()
    chain = ([preferred] if preferred else []) + DEFAULT_CHAIN
    return list(dict.fromkeys(chain))  # de-duplicate, keep order


def _retry_seconds(err: Exception, default: float) -> float:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    return float(match.group(1)) if match else default


def complete(client, *, messages: list, tools: list | None = None, max_tokens: int = 1500,
             fast: bool = True) -> tuple:
    """Chat completion with fallback. Returns (response, model_name_used).

    fast=True asks reasoning models to think briefly (much quicker answers
    for everyday questions); providers that don't support that just ignore it.
    """
    now = time.time()
    tried = []
    for model in model_chain():
        if _unavailable_until.get(model, 0) > now:
            continue
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
        if tools:
            kwargs["tools"] = tools
        if fast:
            kwargs["reasoning_effort"] = "low"
        tried.append(model)
        try:
            try:
                return client.chat.completions.create(**kwargs), model
            except openai.BadRequestError as err:
                if "reasoning" not in str(err).lower():
                    raise
                kwargs.pop("reasoning_effort", None)  # provider doesn't support it: retry plainly
                return client.chat.completions.create(**kwargs), model
        except openai.RateLimitError as err:
            per_day = "PerDay" in str(err)
            wait = _retry_seconds(err, 24 * 3600 if per_day else 60)
            _unavailable_until[model] = time.time() + max(wait, 20)
        except (openai.NotFoundError, openai.PermissionDeniedError):
            _unavailable_until[model] = time.time() + 24 * 3600  # model not available to this key
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
            _unavailable_until[model] = time.time() + 30  # temporary problem: try another model now

    soonest = min((_unavailable_until.get(m, now) for m in model_chain()), default=now)
    minutes = max(1, round((soonest - time.time()) / 60))
    when = f"in about {minutes} minute(s)" if minutes < 90 else f"in about {round(minutes / 60)} hour(s)"
    raise AllModelsBusy(
        "All free AI models have reached their usage limit for now. "
        f"Ana's AI features will work again {when}. "
        "(To remove this limit, enable billing on your Gemini API key or add a second provider in .env.)"
    )


def strip_code_blocks(text: str) -> str:
    """Users want answers, not code: remove any fenced code a model includes."""
    cleaned = re.sub(r"```[\s\S]*?```", "", text or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
