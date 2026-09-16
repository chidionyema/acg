import os
import time
from dataclasses import dataclass
from typing import Callable, Coroutine, Optional

import redis.asyncio as redis
import structlog

log = structlog.get_logger()


@dataclass
class Ceiling:
    tokens_per_hour: int
    dollars_per_hour: float
    watts_per_hour: float
    rpm: int


class Meter:
    """Redis sliding window. On breach: sever, terminate GPU, audit."""

    def __init__(self) -> None:
        self.r = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        self.sever_hook: Optional[Callable[..., Coroutine]] = None

    def _now(self) -> tuple[int, int]:
        t = int(time.time())
        return t, t - (t % 3600)

    async def check(
        self,
        tenant: str,
        ceiling: Ceiling,
        tokens: int,
        dollars: float,
        watts: float,
    ) -> Optional[str]:
        now, hour = self._now()
        keys = {
            "tok": f"m:{tenant}:tok:{hour}",
            "usd": f"m:{tenant}:usd:{hour}",
            "w": f"m:{tenant}:w:{hour}",
            "rpm": f"m:{tenant}:rpm:{now // 60}",
        }
        pipe = self.r.pipeline()
        pipe.incrby(keys["tok"], tokens)
        pipe.expire(keys["tok"], 3600)
        pipe.incrbyfloat(keys["usd"], dollars)
        pipe.expire(keys["usd"], 3600)
        pipe.incrbyfloat(keys["w"], watts)
        pipe.expire(keys["w"], 3600)
        pipe.incr(keys["rpm"])
        pipe.expire(keys["rpm"], 120)
        results = await pipe.execute()
        tok, usd, w, rpm = results[0], results[2], results[4], results[6]

        breach: Optional[str] = None
        if tok > ceiling.tokens_per_hour:
            breach = "tokens_per_hour"
        elif usd > ceiling.dollars_per_hour:
            breach = "dollars_per_hour"
        elif w > ceiling.watts_per_hour:
            breach = "watts_per_hour"
        elif rpm > ceiling.rpm:
            breach = "rpm"

        if breach:
            await self._sever(tenant, breach, {"tok": tok, "usd": usd, "w": w, "rpm": rpm})
        return breach

    async def _sever(self, tenant: str, reason: str, stats: dict) -> None:
        await self.r.set(f"state:{tenant}", "throttled", ex=3600)
        log.error("meter.breach", tenant=tenant, reason=reason, **stats)
        if self.sever_hook:
            try:
                await self.sever_hook(tenant, reason)
            except Exception as e:
                log.error("meter.sever_hook", err=str(e))

    async def is_throttled(self, tenant: str) -> bool:
        return (await self.r.get(f"state:{tenant}")) == "throttled"


meter = Meter()
