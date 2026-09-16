import hashlib
import os
import threading
import time
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class Model:
    id: str
    params: float
    active: float
    license: str
    ram_q4_gb: float
    vram_q4_gb: float
    cpu_tps_ampere: float
    gpu_tps_3090: float
    gpu_tps_a100: float
    intents: tuple[str, ...]
    quality: float
    beats: tuple[str, ...]
    benchmarks: dict
    providers: tuple[str, ...]
    hf: str


class Matrix:
    def __init__(self, path: str = "matrix/models.yaml"):
        self.path = Path(path)
        self._models: dict[str, Model] = {}
        self._by_intent: dict[str, list[Model]] = {}
        self._lock = threading.RLock()
        self._mtime = 0.0
        self._sha = ""
        self.reload()
        self._start_watcher()

    def reload(self) -> bool:
        st = self.path.stat()
        if st.st_mtime == self._mtime:
            return False
        raw = self.path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha == self._sha:
            return False
        data = yaml.safe_load(raw)
        models: dict[str, Model] = {}
        by_intent: dict[str, list[Model]] = {}
        for m in data["models"]:
            model = Model(
                id=m["id"],
                params=m["params"],
                active=m["active"],
                license=m["license"],
                ram_q4_gb=m["ram_q4_gb"],
                vram_q4_gb=m["vram_q4_gb"],
                cpu_tps_ampere=m["cpu_tps_ampere"],
                gpu_tps_3090=m.get("gpu_tps_3090", 0.0),
                gpu_tps_a100=m.get("gpu_tps_a100", 0.0),
                intents=tuple(m["intents"]),
                quality=m["quality"],
                beats=tuple(m.get("beats", [])),
                benchmarks=m.get("benchmarks", {}),
                providers=tuple(m["providers"]),
                hf=m["hf"],
            )
            models[model.id] = model
            for intent in model.intents:
                by_intent.setdefault(intent, []).append(model)
        for lst in by_intent.values():
            lst.sort(key=lambda x: (-x.quality, x.active))
        with self._lock:
            self._models = models
            self._by_intent = by_intent
            self._mtime = st.st_mtime
            self._sha = sha
        log.info("matrix.reload", models=len(models), sha=sha[:8])
        return True

    def _start_watcher(self) -> None:
        def watch() -> None:
            while True:
                try:
                    self.reload()
                except Exception as e:
                    log.error("matrix.watch", err=str(e))
                time.sleep(5)

        t = threading.Thread(target=watch, daemon=True)
        t.start()

    def get(self, model_id: str) -> Optional[Model]:
        with self._lock:
            return self._models.get(model_id)

    def for_intent(self, intent: str) -> list[Model]:
        with self._lock:
            return list(self._by_intent.get(intent, []))

    def all(self) -> list[Model]:
        with self._lock:
            return list(self._models.values())


matrix = Matrix(os.getenv("MATRIX_PATH", "matrix/models.yaml"))
