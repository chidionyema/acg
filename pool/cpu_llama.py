import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()

# Maps adapter name → GGUF path relative to adapters_dir
# Populated at startup from LORA_ADAPTERS env: "name1:path1,name2:path2"
def _parse_adapters(env: str) -> dict[str, str]:
    val = os.getenv(env, "")
    if not val:
        return {}
    result: dict[str, str] = {}
    for pair in val.split(","):
        pair = pair.strip()
        if ":" in pair:
            name, path = pair.split(":", 1)
            result[name.strip()] = path.strip()
    return result


class LlamaCPUServer:
    """Manages llama.cpp server processes on Ampere A1 (ARM64).

    Supports per-request LoRA adapter hot-swap via llama.cpp PR #10994:
    adapters are preloaded at startup with --lora-init-without-apply,
    then selected per-request via the lora[] body field.
    """

    def __init__(
        self,
        model_dir: str = "/models",
        adapters_dir: str = "/models/adapters",
        port_base: int = 8100,
        threads: int = 4,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.adapters_dir = Path(adapters_dir)
        self.port_base = port_base
        self.threads = threads
        self.procs: dict[str, subprocess.Popen] = {}
        self.ports: dict[str, int] = {}
        # adapter_name → index in --lora list (0-based), per model_id
        self._adapter_index: dict[str, dict[str, int]] = {}
        self._lock = asyncio.Lock()
        # Global adapter registry from env
        self._adapters = _parse_adapters("LORA_ADAPTERS")

    def _bin(self) -> str:
        return os.getenv("LLAMA_BIN", "/usr/local/bin/llama-server")

    def _adapter_args(self, model_id: str) -> tuple[list[str], dict[str, int]]:
        """Build --lora flags for adapters compatible with this model.
        Returns (cmd_args, {adapter_name: index}).
        """
        args: list[str] = []
        index: dict[str, int] = {}
        i = 0
        for name, rel_path in self._adapters.items():
            # Convention: adapter path prefixed with model_id indicates compatibility
            if not rel_path.startswith(model_id) and model_id not in rel_path:
                continue
            full = self.adapters_dir / rel_path
            if full.exists():
                args += ["--lora", str(full)]
                index[name] = i
                i += 1
        return args, index

    async def ensure(
        self, model_id: str, gguf: str, ctx: int = 4096
    ) -> int:
        async with self._lock:
            if model_id in self.procs and self.procs[model_id].poll() is None:
                return self.ports[model_id]
            port = self.port_base + len(self.procs)
            path = self.model_dir / gguf
            if not path.exists():
                raise FileNotFoundError(f"GGUF missing: {path}")

            adapter_args, adapter_index = self._adapter_args(model_id)
            self._adapter_index[model_id] = adapter_index

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
            if adapter_args:
                cmd += ["--lora-init-without-apply"] + adapter_args
                log.info(
                    "llama.start",
                    model=model_id,
                    port=port,
                    adapters=list(adapter_index.keys()),
                )
            else:
                log.info("llama.start", model=model_id, port=port)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.procs[model_id] = proc
            self.ports[model_id] = port
            await self._wait_ready(port, timeout=120)
            return port

    def adapter_id(self, model_id: str, adapter_name: str) -> int | None:
        """Return the 0-based lora[] id for a named adapter, or None if not loaded."""
        return self._adapter_index.get(model_id, {}).get(adapter_name)

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
            self._adapter_index.pop(model_id, None)
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def loaded(self) -> list[str]:
        return [m for m, p in self.procs.items() if p.poll() is None]


cpu = LlamaCPUServer()
