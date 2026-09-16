import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

log = structlog.get_logger()


@dataclass
class Instance:
    id: str
    provider: str
    gpu: str
    vram_gb: int
    hourly: float
    endpoint: str
    started: float
    checkpoint_key: str = ""


class SpotDriver:
    """Bids across Vast/RunPod/TensorDock, keeps cheapest, checkpoints."""

    def __init__(self) -> None:
        self.vast_key = os.getenv("VAST_API_KEY", "")
        self.runpod_key = os.getenv("RUNPOD_API_KEY", "")
        self.max_hourly = float(os.getenv("MAX_HOURLY_USD", "0.30"))
        self.instances: dict[str, Instance] = {}

    async def bid(self, model_id: str, vram_gb: int) -> Optional[Instance]:
        if not self.vast_key:
            return None
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(
                "https://console.vast.ai/api/v0/bundles/",
                params={
                    "q": (
                        f'{{"gpu_ram":{{"gte":{vram_gb * 1024}}},'
                        f'"rentable":{{"eq":true}},'
                        f'"order":[["dph_total","asc"]]}}'
                    )
                },
                headers={"Authorization": f"Bearer {self.vast_key}"},
            )
            offers = r.json().get("offers", [])[:5]
        if not offers:
            return None
        best = min(offers, key=lambda o: o["dph_total"])
        if best["dph_total"] > self.max_hourly:
            log.warning("spot.price_high", usd=best["dph_total"], max=self.max_hourly)
            return None
        instance = Instance(
            id=str(best["id"]),
            provider="vast",
            gpu=best["gpu_name"],
            vram_gb=best["gpu_ram"] // 1024,
            hourly=best["dph_total"],
            endpoint="",
            started=time.time(),
            checkpoint_key=f"ckpt:{model_id}:{int(time.time())}",
        )
        self.instances[model_id] = instance
        log.info("spot.bid", model=model_id, usd=instance.hourly, gpu=instance.gpu)
        return instance

    async def release(self, model_id: str) -> None:
        inst = self.instances.pop(model_id, None)
        if not inst:
            return
        log.info("spot.release", model=model_id, id=inst.id)

    def spend_per_hour(self) -> float:
        return sum(i.hourly for i in self.instances.values())


spot = SpotDriver()
