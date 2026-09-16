import asyncio
import os
from typing import Callable, Coroutine, Optional

import httpx
import structlog

from matrix.loader import matrix, Model
from .cache import cache
from .router import router
from pool.free_apis import free_pool
from pool.cpu_llama import cpu
from pool.spot import spot

log = structlog.get_logger()

GGUF_MAP: dict[str, str] = {
    "smollm3-3b": "smollm3-3b-Q4_K_M.gguf",
    "nanbeige-3b": "nanbeige4.1-3b-Q4_K_M.gguf",
    "zaya1-8b": "zaya1-8b-Q4_K_M.gguf",
    "falcon-h1r-7b": "falcon-h1r-7b-Q4_K_M.gguf",
    "ornith-1.5-9b": "ornith-1.5-9b-Q4_K_M.gguf",
}

# Minimum token count for Anthropic cache_control to take effect
_ANTHROPIC_CACHE_MIN_TOKENS = 1024

# Anthropic native API headers
_ANTHROPIC_HEADERS = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "prompt-caching-2024-07-31",
    "content-type": "application/json",
}


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


class Fallback:
    def __init__(self) -> None:
        self.sever_hook_installed = False

    async def install_sever_hook(
        self, sever_fn: Callable[..., Coroutine]
    ) -> None:
        from .meter import meter
        meter.sever_hook = sever_fn

    async def generate(
        self,
        model: Model,
        prompt: str,
        params: dict,
        tenant: str,
        timeout: float = 120.0,
        system: str = "",
    ) -> dict:
        # L0: ACG cache — covers ALL providers
        cached = await cache.get(model.id, prompt, params)
        if cached:
            cached["cached"] = True
            return cached

        # ── Item 1: pre-dispatch cost gate ──────────────────────────────────
        # Checked here, before any paid resource (spot GPU) is allocated
        from .meter import meter
        spend = await meter.peek_spend(tenant)
        from gateway.main import TIER_CEILINGS
        ceiling = TIER_CEILINGS.get(params.get("_tier", "free"))
        if ceiling and spend["dollars"] >= ceiling.dollars_per_hour * 0.95:
            log.warning(
                "fallback.cost_gate",
                tenant=tenant,
                dollars=spend["dollars"],
                ceiling=ceiling.dollars_per_hour,
            )
            raise RuntimeError("cost_ceiling_approaching")

        # L1: free API pool ──────────────────────────────────────────────────
        prov = free_pool.pick(model.id)
        if prov:
            try:
                out = await self._call_free(prov, model.id, prompt, params, timeout, system)
                await cache.put(model.id, prompt, params, out)
                return out
            except Exception as e:
                log.warning("L1.fail", provider=prov.name, err=str(e))
                prov.consecutive_fails += 1
                if prov.consecutive_fails >= 3:
                    prov.healthy = False

        # L2: CPU — llama.cpp on Ampere ──────────────────────────────────────
        if "local_cpu" in model.providers and model.id in GGUF_MAP:
            try:
                adapter = params.get("lora_adapter")
                out = await self._call_cpu(model.id, prompt, params, timeout, system, adapter)
                await cache.put(model.id, prompt, params, out)
                return out
            except Exception as e:
                log.warning("L2.fail", model=model.id, err=str(e))

        # ── Item 1 continued: gate fires hard before GPU spend ───────────────
        if ceiling and spend["dollars"] >= ceiling.dollars_per_hour:
            raise RuntimeError("cost_ceiling_hard")

        # L4: spot GPU ───────────────────────────────────────────────────────
        if "spot" in model.providers or "local_gpu" in model.providers:
            inst = await spot.bid(model.id, int(model.vram_q4_gb))
            if inst:
                try:
                    out = await self._call_gpu(inst, model.id, prompt, params, timeout)
                    await cache.put(model.id, prompt, params, out)
                    return out
                except Exception as e:
                    log.error("L4.fail", err=str(e))

        # L5: degrade to any CPU model ───────────────────────────────────────
        for m in matrix.all():
            if "local_cpu" in m.providers and m.id in GGUF_MAP:
                try:
                    out = await self._call_cpu(m.id, prompt, params, timeout, system)
                    out["degraded"] = True
                    out["model"] = m.id
                    return out
                except Exception:
                    continue

        raise RuntimeError("all_layers_exhausted")

    # ── Item 2: provider-aware prompt caching ────────────────────────────────

    async def _call_free(
        self,
        prov,
        model_id: str,
        prompt: str,
        params: dict,
        timeout: float,
        system: str = "",
    ) -> dict:
        prov.record()
        if prov.cache_strategy == "anthropic":
            return await self._call_anthropic_native(prov, model_id, prompt, params, timeout, system)
        return await self._call_openai_compat(prov, model_id, prompt, params, timeout, system)

    async def _call_openai_compat(
        self,
        prov,
        model_id: str,
        prompt: str,
        params: dict,
        timeout: float,
        system: str = "",
    ) -> dict:
        """OpenAI-compatible /chat/completions.
        Google and OpenRouter auto-cache; no extra headers needed.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": prov.model_map[model_id],
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"{prov.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {prov.api_key}"},
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": model_id,
            "provider": f"free:{prov.name}",
            "choices": data["choices"],
            "usage": data.get("usage", {}),
        }

    async def _call_anthropic_native(
        self,
        prov,
        model_id: str,
        prompt: str,
        params: dict,
        timeout: float,
        system: str = "",
    ) -> dict:
        """Anthropic /v1/messages with cache_control on system prompt.
        90% cost reduction on cache reads when system prompt ≥ 1024 tokens.
        """
        upstream = prov.model_map[model_id]
        body: dict = {
            "model": upstream,
            "max_tokens": params.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
        }
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]

        if system:
            use_cache = _token_estimate(system) >= _ANTHROPIC_CACHE_MIN_TOKENS
            body["system"] = [
                {
                    "type": "text",
                    "text": system,
                    **({"cache_control": {"type": "ephemeral"}} if use_cache else {}),
                }
            ]

        headers = {
            "x-api-key": prov.api_key,
            **_ANTHROPIC_HEADERS,
        }
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"{prov.base_url}/messages",
                headers=headers,
                json=body,
            )
            r.raise_for_status()
            data = r.json()

        # Normalise to OpenAI-compatible shape
        text = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        if cache_read:
            log.info(
                "anthropic.cache_hit",
                model=model_id,
                cache_read_tokens=cache_read,
            )
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "model": model_id,
            "provider": f"free:{prov.name}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": data.get("stop_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            },
        }

    # ── Item 4: LoRA-aware CPU call ──────────────────────────────────────────

    async def _call_cpu(
        self,
        model_id: str,
        prompt: str,
        params: dict,
        timeout: float,
        system: str = "",
        adapter_name: Optional[str] = None,
    ) -> dict:
        port = await cpu.ensure(model_id, GGUF_MAP[model_id])
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": model_id,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
            "stream": False,
        }

        # LoRA hot-swap: inject lora[] selector if adapter requested
        if adapter_name:
            lora_id = cpu.adapter_id(model_id, adapter_name)
            if lora_id is not None:
                body["lora"] = [{"id": lora_id, "scale": 1.0}]
                log.debug("cpu.lora", model=model_id, adapter=adapter_name, id=lora_id)
            else:
                log.warning("cpu.lora_missing", model=model_id, adapter=adapter_name)

        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json=body
            )
            r.raise_for_status()
            data = r.json()
        data["provider"] = "cpu:ampere"
        return data

    async def _call_gpu(
        self, inst, model_id: str, prompt: str, params: dict, timeout: float
    ) -> dict:
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(
                f"{inst.endpoint}/v1/chat/completions", json=body
            )
            r.raise_for_status()
            data = r.json()
        data["provider"] = f"spot:{inst.provider}"
        return data


fallback = Fallback()
