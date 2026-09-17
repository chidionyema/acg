"""
LangSmith tracing for ACG.

Every request through the gateway becomes a parent run. Each layer attempt
(L0 cache, L1 free API, L2 CPU, L4 spot GPU) is a child run. Traces include
model, provider, tenant, tier, layer, token counts, and latency.

Enabled when LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is set.
No-op when either key is absent or langsmith is not installed — zero overhead.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog

log = structlog.get_logger()

_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
_PROJECT = os.getenv("LANGCHAIN_PROJECT", "acg")
_ENABLED = bool(_API_KEY)

_client: Any = None

if _ENABLED:
    try:
        from langsmith import Client
        _client = Client(api_key=_API_KEY)
        log.info("langsmith.enabled", project=_PROJECT)
    except ImportError:
        _ENABLED = False
        log.warning("langsmith.disabled", reason="package not installed; pip install langsmith")


@asynccontextmanager
async def request_trace(
    *,
    request_id: str,
    model_id: str,
    prompt: str,
    tenant: str,
    tier: str,
    intent: str,
):
    """Top-level trace for one /v1/chat/completions request."""
    if not _ENABLED or _client is None:
        yield None
        return

    run_id = request_id
    try:
        _client.create_run(
            id=run_id,
            name="acg.chat",
            run_type="chain",
            project_name=_PROJECT,
            inputs={
                "model": model_id,
                "prompt_chars": len(prompt),
                "tenant": tenant,
                "tier": tier,
                "intent": intent,
            },
            tags=["acg", f"tier:{tier}", f"intent:{intent}"],
        )
    except Exception as e:
        log.warning("langsmith.create_run.failed", err=str(e))
        yield None
        return

    t0 = time.monotonic()
    result: dict = {}
    try:
        yield result
        _client.update_run(
            run_id,
            outputs={
                "provider": result.get("provider", "unknown"),
                "cached": result.get("cached", False),
                "degraded": result.get("degraded", False),
                "usage": result.get("usage", {}),
                "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            },
        )
    except Exception as exc:
        try:
            _client.update_run(run_id, error=str(exc))
        except Exception:
            pass
        raise


@asynccontextmanager
async def layer_trace(
    *,
    parent_run_id: str | None,
    layer: str,
    provider: str,
    model_id: str,
    prompt_chars: int,
):
    """Child trace for a single layer attempt within a request."""
    if not _ENABLED or _client is None or parent_run_id is None:
        yield {}
        return

    run_id = str(uuid.uuid4())
    try:
        _client.create_run(
            id=run_id,
            name=f"acg.{layer}",
            run_type="llm",
            project_name=_PROJECT,
            parent_run_id=parent_run_id,
            inputs={"provider": provider, "model": model_id, "prompt_chars": prompt_chars},
        )
    except Exception as e:
        log.warning("langsmith.layer_run.failed", layer=layer, err=str(e))
        yield {}
        return

    t0 = time.monotonic()
    result: dict = {}
    try:
        yield result
        _client.update_run(
            run_id,
            outputs={**result, "latency_ms": round((time.monotonic() - t0) * 1000, 2)},
        )
    except Exception as exc:
        try:
            _client.update_run(run_id, error=str(exc))
        except Exception:
            pass
        raise
