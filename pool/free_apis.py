import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model_map: dict[str, str]
    rpm: int
    weight: float = 1.0
    healthy: bool = True
    consecutive_fails: int = 0
    last_check: float = 0.0
    _rpm_window: list[float] = field(default_factory=list)

    def can_call(self) -> bool:
        now = time.time()
        self._rpm_window = [t for t in self._rpm_window if now - t < 60]
        return self.healthy and len(self._rpm_window) < self.rpm

    def record(self) -> None:
        self._rpm_window.append(time.time())


@dataclass
class FreePool:
    providers: list[Provider]

    async def _health(self, p: Provider, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get(
                f"{p.base_url}/models",
                headers={"Authorization": f"Bearer {p.api_key}"},
                timeout=5.0,
            )
            p.healthy = r.status_code < 500
            if p.healthy:
                p.consecutive_fails = 0
            else:
                p.consecutive_fails += 1
        except Exception:
            p.consecutive_fails += 1
            if p.consecutive_fails >= 3:
                p.healthy = False
        p.last_check = time.time()

    async def health_loop(self, interval: float = 15.0) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.gather(
                    *[self._health(p, client) for p in self.providers],
                    return_exceptions=True,
                )
                await asyncio.sleep(interval)

    def pick(self, model_id: str) -> Optional[Provider]:
        candidates = [
            p for p in self.providers if p.can_call() and model_id in p.model_map
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (-p.weight, p.consecutive_fails))
        return candidates[0]


def build_pool() -> FreePool:
    def mk(
        name: str,
        url: str,
        env: str,
        mmap: dict,
        rpm: int,
        weight: float = 1.0,
    ) -> Optional[Provider]:
        key = os.getenv(env, "")
        return Provider(name, url, key, mmap, rpm, weight) if key else None

    specs = [
        (
            "groq",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            {
                "llama-3.3-70b": "llama-3.3-70b-versatile",
                "gpt-oss-120b": "openai/gpt-oss-120b",
            },
            30,
            1.5,
        ),
        (
            "cerebras",
            "https://api.cerebras.ai/v1",
            "CEREBRAS_API_KEY",
            {
                "llama-3.3-70b": "llama3.3-70b",
                "gpt-oss-120b": "gpt-oss-120b",
            },
            30,
            1.5,
        ),
        (
            "sambanova",
            "https://api.sambanova.ai/v1",
            "SAMBANOVA_API_KEY",
            {"llama-3.3-70b": "Meta-Llama-3.3-70B-Instruct"},
            30,
            1.2,
        ),
        (
            "together",
            "https://api.together.xyz/v1",
            "TOGETHER_API_KEY",
            {
                "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "qwen3.6-27b": "Qwen/Qwen3.6-27B",
            },
            60,
            1.0,
        ),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY",
            {
                "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
                "gpt-oss-120b": "openai/gpt-oss-120b:free",
            },
            20,
            1.0,
        ),
        (
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "GOOGLE_API_KEY",
            {"gemini-2.5-flash": "gemini-2.5-flash"},
            60,
            1.3,
        ),
        (
            "mistral",
            "https://api.mistral.ai/v1",
            "MISTRAL_API_KEY",
            {"mistral-small": "mistral-small-latest"},
            60,
            1.0,
        ),
        (
            "cohere",
            "https://api.cohere.ai/compatibility/v1",
            "COHERE_API_KEY",
            {"command-r": "command-r-plus-08-2024"},
            20,
            0.9,
        ),
        (
            "hf",
            "https://api-inference.huggingface.co/v1",
            "HF_TOKEN",
            {"llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct"},
            30,
            0.8,
        ),
        (
            "replicate",
            "https://api.replicate.com/v1",
            "REPLICATE_API_TOKEN",
            {"llama-3.3-70b": "meta/meta-llama-3.3-70b-instruct"},
            10,
            0.7,
        ),
    ]
    providers = [p for p in (mk(*s) for s in specs) if p]
    log.info("free_pool.built", n=len(providers), names=[p.name for p in providers])
    return FreePool(providers)


free_pool = build_pool()
