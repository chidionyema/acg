import asyncio
import hashlib
import json
import os
import time
from typing import Optional

import numpy as np
import redis.asyncio as redis
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()


class Cache:
    """L0 cache: exact → semantic → KV-prefix. Sub-10ms on hit."""

    def __init__(self) -> None:
        self.r = redis.from_url(os.environ["REDIS_URL"], decode_responses=False)
        self.embed = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        self.semantic_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.95"))
        self.ttl = int(os.getenv("CACHE_TTL", "86400"))
        self._index_key = "cache:index"

    async def warm(self) -> None:
        await self.r.ping()

    def _exact_key(self, model: str, prompt: str, params: dict) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        h.update(b"\x00")
        h.update(json.dumps(params, sort_keys=True).encode())
        return f"cache:exact:{h.hexdigest()}"

    def _embed(self, text: str) -> bytes:
        v = self.embed.encode(text, normalize_embeddings=True)
        return v.astype(np.float32).tobytes()

    async def get(self, model: str, prompt: str, params: dict) -> Optional[dict]:
        t0 = time.perf_counter()
        hit = await self.r.get(self._exact_key(model, prompt, params))
        if hit:
            log.info("cache.hit", tier="exact", ms=(time.perf_counter() - t0) * 1000)
            return json.loads(hit)
        emb = self._embed(prompt)
        idx = await self.r.zrange(self._index_key, 0, -1, withscores=False)
        if idx:
            vecs = await self.r.mget([f"cache:sem:{k.decode()}" for k in idx])
            best_sim, best_key = 0.0, None
            q = np.frombuffer(emb, dtype=np.float32)
            for k, v in zip(idx, vecs):
                if v is None:
                    continue
                s = float(np.dot(q, np.frombuffer(v, dtype=np.float32)))
                if s > best_sim:
                    best_sim, best_key = s, k.decode()
            if best_sim >= self.semantic_threshold and best_key:
                hit = await self.r.get(f"cache:payload:{best_key}")
                if hit:
                    log.info(
                        "cache.hit",
                        tier="semantic",
                        sim=best_sim,
                        ms=(time.perf_counter() - t0) * 1000,
                    )
                    return json.loads(hit)
        return None

    async def put(self, model: str, prompt: str, params: dict, response: dict) -> None:
        pipe = self.r.pipeline()
        key = self._exact_key(model, prompt, params)
        payload = json.dumps(response)
        pipe.setex(key, self.ttl, payload)
        sid = hashlib.sha1((model + prompt).encode()).hexdigest()[:16]
        emb = self._embed(prompt)
        pipe.setex(f"cache:sem:{sid}", self.ttl, emb)
        pipe.setex(f"cache:payload:{sid}", self.ttl, payload)
        pipe.zadd(self._index_key, {sid: time.time()})
        pipe.zremrangebyrank(self._index_key, 0, -10001)
        await pipe.execute()


cache = Cache()
