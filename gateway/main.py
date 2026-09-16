import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .cache import cache
from .meter import meter, Ceiling
from .router import router
from .fallback import fallback
from .batch_worker import batch_worker
from .health import health_summary
from pool.free_apis import free_pool
from pool.cpu_llama import cpu
from matrix.loader import matrix

log = structlog.get_logger()

TIER_CEILINGS: dict[str, Ceiling] = {
    "free": Ceiling(
        tokens_per_hour=50_000, dollars_per_hour=0.0, watts_per_hour=50.0, rpm=60
    ),
    "pro": Ceiling(
        tokens_per_hour=1_000_000, dollars_per_hour=2.0, watts_per_hour=500.0, rpm=600
    ),
    "scale": Ceiling(
        tokens_per_hour=10_000_000,
        dollars_per_hour=20.0,
        watts_per_hour=5000.0,
        rpm=6000,
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache.warm()
    asyncio.create_task(free_pool.health_loop(15.0))
    await batch_worker.start()

    async def sever(tenant: str, reason: str) -> None:
        log.error("sever.tenant", tenant=tenant, reason=reason)

    await fallback.install_sever_hook(sever)
    log.info(
        "acg.ready",
        models=len(matrix.all()),
        free_providers=len(free_pool.providers),
        batch_worker=batch_worker.available(),
    )
    yield
    await batch_worker.stop()


app = FastAPI(
    title="Asymmetric Compute Grid", version="1.1.0", lifespan=lifespan
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    # "normal" = hot path (default), "low" = async batch queue (free-tier only)
    priority: str = "normal"
    # Optional LoRA adapter name to select on CPU models
    lora_adapter: Optional[str] = None


def tenant_from_key(api_key: str) -> tuple[str, str]:
    # Format: acg_<tier>_<tenant>_<secret>
    parts = api_key.split("_")
    if len(parts) < 4 or parts[0] != "acg":
        raise HTTPException(401, "invalid_api_key")
    return parts[1], parts[2]


def _extract_messages(req: ChatRequest) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) split from messages."""
    system = "\n".join(m.content for m in req.messages if m.role == "system")
    user = "\n".join(m.content for m in req.messages if m.role != "system")
    return system, user


@app.get("/health")
async def health():
    return health_summary()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m.id, "object": "model", "owned_by": "acg"}
            for m in matrix.all()
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, authorization: str = Header(...)):
    t0 = time.perf_counter()
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing_bearer")
    tier, tenant = tenant_from_key(authorization[7:])
    ceiling = TIER_CEILINGS.get(tier, TIER_CEILINGS["free"])

    if await meter.is_throttled(tenant):
        raise HTTPException(429, "tenant_throttled")

    system, prompt = _extract_messages(req)

    # ── Item 5: async batch queue for low-priority free-tier requests ────────
    if req.priority == "low" and tier == "free" and batch_worker.available():
        try:
            batch_model = "claude-haiku"
            out = await batch_worker.submit(
                model=batch_model,
                system=system,
                prompt=prompt,
                params={"temperature": req.temperature, "max_tokens": req.max_tokens},
            )
            out["acg"] = {
                "model_id": batch_model,
                "provider": "anthropic:batch",
                "tier": tier,
                "tenant": tenant,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "cached": False,
                "degraded": False,
                "intent": router.classify(prompt)[0],
                "priority": "low",
                "request_id": str(uuid.uuid4()),
            }
            return JSONResponse(out)
        except Exception as e:
            log.warning("batch_worker.fallthrough", err=str(e))
            # Fall through to hot path on batch failure

    # ── Hot path ─────────────────────────────────────────────────────────────
    if req.model == "auto":
        model = router.route(prompt, tenant_tier=tier)
    else:
        model = matrix.get(req.model)
        if not model:
            raise HTTPException(404, f"model_not_found:{req.model}")

    params = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "_tier": tier,
        **({"lora_adapter": req.lora_adapter} if req.lora_adapter else {}),
    }

    try:
        out = await fallback.generate(
            model, prompt, params, tenant, system=system
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    usage = out.get("usage", {})
    tokens = usage.get("total_tokens", len(prompt.split()) + req.max_tokens // 2)
    dollars = 0.0
    watts = tokens / 100.0
    breach = await meter.check(tenant, ceiling, tokens, dollars, watts)
    if breach:
        raise HTTPException(429, f"ceiling_exceeded:{breach}")

    out["acg"] = {
        "model_id": model.id,
        "provider": out.get("provider", "unknown"),
        "tier": tier,
        "tenant": tenant,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "cached": out.get("cached", False),
        "degraded": out.get("degraded", False),
        "intent": router.classify(prompt)[0],
        "difficulty": router.difficulty(prompt),
        "priority": req.priority,
        "request_id": str(uuid.uuid4()),
    }
    return JSONResponse(out)


@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest

    return StreamingResponse(
        iter([generate_latest()]), media_type="text/plain; version=0.0.4"
    )
