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

# RouteLLM-inspired: complexity floor per difficulty tier → cascade to stronger model
_QUALITY_FLOOR = {"easy": 0.70, "medium": 0.76, "hard": 0.85}

_HARD_RE = re.compile(
    r"step.by.step|prove\b|derive\b|architect\b|tradeoff|trade.off"
    r"|refactor.*production|optimi[sz]e.*system|design.*scalab",
    re.I,
)
_MEDIUM_RE = re.compile(
    r"\bexplain\b|\bcompare\b|\bdifference\b|\bwhy\b|\bhow\b.{0,40}\bwork",
    re.I,
)


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

    def difficulty(self, prompt: str) -> str:
        """Estimate query complexity for cascade routing (RouteLLM-inspired)."""
        tokens = len(prompt.split())
        multi_code_block = len(re.findall(r"```", prompt)) >= 2
        if tokens > 300 or multi_code_block or _HARD_RE.search(prompt):
            return "hard"
        if tokens > 80 or _MEDIUM_RE.search(prompt):
            return "medium"
        return "easy"

    def route(self, prompt: str, tenant_tier: str = "free") -> Model:
        intent, _ = self.classify(prompt)
        diff = self.difficulty(prompt)
        quality_floor = _QUALITY_FLOOR[diff]

        all_candidates = matrix.for_intent(intent) or matrix.for_intent("general")

        if tenant_tier == "free":
            all_candidates = [
                m for m in all_candidates
                if "local_cpu" in m.providers or "free_api" in m.providers
            ]
        if not all_candidates:
            all_candidates = matrix.for_intent("general")

        # cascade: prefer models clearing the difficulty quality floor
        candidates = [m for m in all_candidates if m.quality >= quality_floor]
        if not candidates:
            candidates = all_candidates  # fall back — never leave empty

        candidates.sort(key=lambda m: (-m.quality, m.active))
        chosen = candidates[0]
        log.debug(
            "router.route",
            intent=intent,
            difficulty=diff,
            quality_floor=quality_floor,
            model=chosen.id,
        )
        return chosen


router = SemanticRouter()
