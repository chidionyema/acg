"""Drop-in OpenAI-compatible client for ACG."""
import os
from openai import OpenAI


def client(
    api_key: str = "",
    base_url: str = "",
    tier: str = "free",
    tenant: str = "default",
) -> OpenAI:
    key = api_key or os.getenv("ACG_API_KEY") or f"acg_{tier}_{tenant}_local"
    url = base_url or os.getenv("ACG_BASE_URL", "http://localhost:8000/v1")
    return OpenAI(api_key=key, base_url=url)
