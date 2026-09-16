"""Provider health checks and circuit breaker state."""
from pool.free_apis import free_pool
from pool.cpu_llama import cpu
from matrix.loader import matrix


def health_summary() -> dict:
    return {
        "status": "ok",
        "free_providers_healthy": sum(1 for p in free_pool.providers if p.healthy),
        "free_providers_total": len(free_pool.providers),
        "cpu_loaded": cpu.loaded(),
        "models": len(matrix.all()),
    }
