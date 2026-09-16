"""Async batch queue for low-priority free-tier requests.

Uses Anthropic Message Batches API: 50% cost reduction, up to 24h SLA.
Only used for priority='low' requests — never in the hot path.
"""
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()

_ANTHROPIC_BASE = "https://api.anthropic.com/v1"
_BATCH_WINDOW_SECS = float(os.getenv("BATCH_WINDOW_SECS", "10"))
_MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
_POLL_INTERVAL = float(os.getenv("BATCH_POLL_SECS", "30"))
_DEFAULT_MODEL = os.getenv("BATCH_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class _PendingRequest:
    custom_id: str
    model: str
    system: str
    prompt: str
    params: dict
    future: asyncio.Future


class BatchWorker:
    """Collects low-priority requests, submits to Anthropic Batches API in windows."""

    def __init__(self) -> None:
        self._api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._running = False

    def available(self) -> bool:
        return bool(self._api_key)

    async def start(self) -> None:
        if not self._api_key:
            log.info("batch_worker.disabled", reason="no ANTHROPIC_API_KEY")
            return
        self._running = True
        asyncio.create_task(self._dispatch_loop())
        log.info("batch_worker.started")

    async def submit(
        self, model: str, system: str, prompt: str, params: dict
    ) -> dict:
        """Enqueue a request. Awaiting the returned coroutine blocks until result arrives."""
        if not self._api_key:
            raise RuntimeError("batch_worker_unavailable")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        req = _PendingRequest(
            custom_id=str(uuid.uuid4()),
            model=model or _DEFAULT_MODEL,
            system=system,
            prompt=prompt,
            params=params,
            future=fut,
        )
        await self._queue.put(req)
        log.debug("batch_worker.enqueued", custom_id=req.custom_id)
        return await fut

    async def _dispatch_loop(self) -> None:
        while self._running:
            batch: list[_PendingRequest] = []
            deadline = time.monotonic() + _BATCH_WINDOW_SECS
            while time.monotonic() < deadline and len(batch) < _MAX_BATCH_SIZE:
                try:
                    remaining = max(0.1, deadline - time.monotonic())
                    req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break
            if batch:
                asyncio.create_task(self._run_batch(batch))

    async def _run_batch(self, batch: list[_PendingRequest]) -> None:
        requests_payload = []
        for req in batch:
            msg_params: dict = {
                "model": req.model,
                "max_tokens": req.params.get("max_tokens", 1024),
                "messages": [{"role": "user", "content": req.prompt}],
            }
            if req.system:
                msg_params["system"] = [
                    {
                        "type": "text",
                        "text": req.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            if req.params.get("temperature") is not None:
                msg_params["temperature"] = req.params["temperature"]
            requests_payload.append(
                {"custom_id": req.custom_id, "params": msg_params}
            )

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as c:
            try:
                r = await c.post(
                    f"{_ANTHROPIC_BASE}/messages/batches",
                    headers=headers,
                    json={"requests": requests_payload},
                )
                r.raise_for_status()
                batch_id = r.json()["id"]
                log.info("batch_worker.submitted", batch_id=batch_id, n=len(batch))
            except Exception as e:
                log.error("batch_worker.submit_fail", err=str(e))
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError(f"batch_submit_failed: {e}"))
                return

        # Poll for completion
        result_map = await self._poll_until_done(batch_id, headers)

        for req in batch:
            if not req.future.done():
                result = result_map.get(req.custom_id)
                if result:
                    req.future.set_result(result)
                else:
                    req.future.set_exception(RuntimeError("batch_result_missing"))

    async def _poll_until_done(
        self, batch_id: str, headers: dict
    ) -> dict[str, dict]:
        async with httpx.AsyncClient(timeout=30.0) as c:
            while True:
                await asyncio.sleep(_POLL_INTERVAL)
                try:
                    r = await c.get(
                        f"{_ANTHROPIC_BASE}/messages/batches/{batch_id}",
                        headers=headers,
                    )
                    r.raise_for_status()
                    status = r.json().get("processing_status")
                    log.debug("batch_worker.poll", batch_id=batch_id, status=status)
                    if status == "ended":
                        break
                except Exception as e:
                    log.warning("batch_worker.poll_err", err=str(e))

            # Stream NDJSON results
            results: dict[str, dict] = {}
            try:
                r = await c.get(
                    f"{_ANTHROPIC_BASE}/messages/batches/{batch_id}/results",
                    headers=headers,
                    timeout=60.0,
                )
                r.raise_for_status()
                for line in r.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    cid = row.get("custom_id", "")
                    if row.get("result", {}).get("type") == "succeeded":
                        msg = row["result"]["message"]
                        results[cid] = {
                            "id": msg.get("id", ""),
                            "object": "chat.completion",
                            "model": msg.get("model", ""),
                            "provider": "anthropic:batch",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": msg["content"][0]["text"]
                                        if msg.get("content")
                                        else "",
                                    },
                                    "finish_reason": msg.get("stop_reason", "stop"),
                                }
                            ],
                            "usage": msg.get("usage", {}),
                        }
                    else:
                        log.warning(
                            "batch_worker.result_failed",
                            custom_id=cid,
                            result_type=row.get("result", {}).get("type"),
                        )
            except Exception as e:
                log.error("batch_worker.results_fetch_fail", err=str(e))
            return results

    async def stop(self) -> None:
        self._running = False


batch_worker = BatchWorker()
