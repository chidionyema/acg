import re
from typing import Optional

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from matrix.loader import matrix, Model

log = structlog.get_logger()

INTENT_PROTOS = {
    "code": "write, debug, refactor, explain code, implement function, fix bug, unit test",
    "math": "solve math, prove, derive, compute, equation, integral, algebra, geometry",
    "reasoning": "reason step by step, analyze, deduce, logic puzzle, plan, strategy",
    "chat": "casual conversation, greeting, small talk, opinion, chit chat",
    "general": "summarize, translate, rewrite, extract, classify, general question",
}


class SemanticRouter:
    def __init__(self) -> None:
        self.embed = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        self._proto_emb: dict[str, np.ndarray] = {}
        for k, v in INTENT_PROTOS.items():
            self._proto_emb[k] = self.embed.encode(v, normalize_embeddings=True)

    def classify(self, prompt: str) -> tuple[str, float]:
        v = self.embed.encode(prompt[:512], normalize_embeddings=True)
        best, best_s = "general", 0.0
        for k, p in self._proto_emb.items():
            s = float(np.dot(v, p))
            if s > best_s:
                best_s, best = s, k
        if re.search(r"```|\bdef \b|\bclass \b|;\s*$|\bimport \b", prompt):
            best = "code"
        if re.search(r"\b(solve|prove|derive|integral|equation)\b", prompt, re.I):
            best = "math"
        return best, best_s

    def route(self, prompt: str, tenant_tier: str = "free") -> Model:
        intent, _ = self.classify(prompt)
        candidates = matrix.for_intent(intent) or matrix.for_intent("general")
        if tenant_tier == "free":
            candidates = [
                m
                for m in candidates
                if "local_cpu" in m.providers or "free_api" in m.providers
            ]
        if not candidates:
            candidates = matrix.for_intent("general")
        candidates.sort(key=lambda m: (-m.quality, m.active))
        return candidates[0]


router = SemanticRouter()
