import asyncio
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
}


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
    ) -> dict:
        cached = await cache.get(model.id, prompt, params)
        if cached:
            cached["cached"] = True
            return cached

        # L1: free API
        prov = free_pool.pick(model.id)
        if prov:
            try:
                out = await self._call_free(prov, model.id, prompt, params, timeout)
                await cache.put(model.id, prompt, params, out)
                return out
            except Exception as e:
                log.warning("L1.fail", provider=prov.name, err=str(e))
                prov.consecutive_fails += 1
                if prov.consecutive_fails >= 3:
                    prov.healthy = False

        # L2: CPU (llama.cpp on Ampere)
        if "local_cpu" in model.providers and model.id in GGUF_MAP:
            try:
                out = await self._call_cpu(model.id, prompt, params, timeout)
                await cache.put(model.id, prompt, params, out)
                return out
            except Exception as e:
                log.warning("L2.fail", model=model.id, err=str(e))

        # L4: spot GPU
        if "spot" in model.providers or "local_gpu" in model.providers:
            inst = await spot.bid(model.id, int(model.vram_q4_gb))
            if inst:
                try:
                    out = await self._call_gpu(inst, model.id, prompt, params, timeout)
                    await cache.put(model.id, prompt, params, out)
                    return out
                except Exception as e:
                    log.error("L4.fail", err=str(e))

        # L5: degrade to any CPU model
        for m in matrix.all():
            if "local_cpu" in m.providers and m.id in GGUF_MAP:
                try:
                    out = await self._call_cpu(m.id, prompt, params, timeout)
                    out["degraded"] = True
                    out["model"] = m.id
                    return out
                except Exception:
                    continue

        raise RuntimeError("all_layers_exhausted")

    async def _call_free(
        self,
        prov,
        model_id: str,
        prompt: str,
        params: dict,
        timeout: float,
    ) -> dict:
        upstream = prov.model_map[model_id]
        body = {
            "model": upstream,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
            "stream": False,
        }
        prov.record()
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

    async def _call_cpu(
        self, model_id: str, prompt: str, params: dict, timeout: float
    ) -> dict:
        port = await cpu.ensure(model_id, GGUF_MAP[model_id])
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 1024),
            "stream": False,
        }
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
