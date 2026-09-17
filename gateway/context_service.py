"""
Estate context service — singleton that fetches and caches the estate knowledge
graph brief. Injected as a system prompt prefix by the chat endpoint so every
request, regardless of provider, carries estate rules and live graph state.
"""

import asyncio
import os
import subprocess
import time

import structlog

log = structlog.get_logger()

_ESTATE_GRAPH = os.path.expanduser("~/Documents/code/estate-graph")
_AGENTS_MD = os.path.expanduser("~/AGENTS.md")
_TTL = 60.0


class ContextService:
    def __init__(self) -> None:
        self._cache: str = ""
        self._expires: float = 0.0
        self._lock = asyncio.Lock()

    def _build_static(self) -> str:
        try:
            return open(_AGENTS_MD).read().strip()
        except OSError:
            return ""

    def _build_live(self) -> str:
        try:
            result = subprocess.run(
                ["growmos", "--root", _ESTATE_GRAPH, "context", "--brief", "--budget", "500"],
                capture_output=True, text=True, timeout=8.0,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    async def get(self) -> str:
        now = time.monotonic()
        if now < self._expires and self._cache:
            return self._cache
        async with self._lock:
            if now < self._expires and self._cache:
                return self._cache
            static = await asyncio.get_event_loop().run_in_executor(None, self._build_static)
            live = await asyncio.get_event_loop().run_in_executor(None, self._build_live)
            parts = [p for p in [static, live] if p]
            self._cache = "\n\n".join(parts)
            self._expires = time.monotonic() + _TTL
            log.info("context_service.refresh", chars=len(self._cache))
            return self._cache


context_service = ContextService()
