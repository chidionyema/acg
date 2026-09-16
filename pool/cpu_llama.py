import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()


class LlamaCPUServer:
    """Manages llama.cpp server processes on Ampere A1 (ARM64)."""

    def __init__(
        self,
        model_dir: str = "/models",
        port_base: int = 8100,
        threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.port_base = port_base
        self.threads = threads
        self.procs: dict[str, subprocess.Popen] = {}
        self.ports: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _bin(self) -> str:
        return os.getenv("LLAMA_BIN", "/usr/local/bin/llama-server")

    async def ensure(self, model_id: str, gguf: str, ctx: int = 4096) -> int:
        async with self._lock:
            if model_id in self.procs and self.procs[model_id].poll() is None:
                return self.ports[model_id]
            port = self.port_base + len(self.procs)
            path = self.model_dir / gguf
            if not path.exists():
                raise FileNotFoundError(f"GGUF missing: {path}")
            cmd = [
                self._bin(),
                "-m", str(path),
                "--host", "127.0.0.1",
                "--port", str(port),
                "-t", str(self.threads),
                "-c", str(ctx),
                "-b", "512",
                "-ub", "512",
                "--mlock",
                "--no-mmap",
                "-np", "4",
                "--cont-batching",
                "--metrics",
            ]
            log.info("llama.start", model=model_id, port=port)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.procs[model_id] = proc
            self.ports[model_id] = port
            await self._wait_ready(port, timeout=120)
            return port

    async def _wait_ready(self, port: int, timeout: float) -> None:
        t0 = time.time()
        async with httpx.AsyncClient() as c:
            while time.time() - t0 < timeout:
                try:
                    r = await c.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        raise TimeoutError(f"llama not ready on {port}")

    async def stop(self, model_id: str) -> None:
        async with self._lock:
            p = self.procs.pop(model_id, None)
            self.ports.pop(model_id, None)
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def loaded(self) -> list[str]:
        return [m for m, p in self.procs.items() if p.poll() is None]


cpu = LlamaCPUServer()
