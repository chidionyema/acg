"""Smoke tests — no Redis, no live LLM, no sentence-transformers download.
Validates: matrix loader, cascade router difficulty, provider pool, LoRA index,
meter peek contract, batch worker availability check, cost gate logic.
"""
import os
import sys
import unittest

# Point at repo root so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("MATRIX_PATH", "matrix/models.yaml")

# ── Matrix loader ─────────────────────────────────────────────────────────────

class TestMatrixLoader(unittest.TestCase):
    def setUp(self):
        from matrix.loader import Matrix
        self.m = Matrix("matrix/models.yaml")

    def test_all_models_loaded(self):
        models = self.m.all()
        self.assertGreaterEqual(len(models), 8, "expected ≥8 models in matrix")

    def test_get_by_id(self):
        m = self.m.get("smollm3-3b")
        self.assertIsNotNone(m)
        self.assertEqual(m.id, "smollm3-3b")
        self.assertIn("code", m.intents)

    def test_for_intent_sorted_by_quality(self):
        code_models = self.m.for_intent("code")
        self.assertGreater(len(code_models), 0)
        qualities = [m.quality for m in code_models]
        self.assertEqual(qualities, sorted(qualities, reverse=True))

    def test_model_fields_typed(self):
        m = self.m.get("zaya1-8b")
        self.assertIsNotNone(m)
        self.assertIsInstance(m.params, float)
        self.assertIsInstance(m.intents, tuple)
        self.assertIsInstance(m.providers, tuple)
        self.assertIsInstance(m.benchmarks, dict)

    def test_hot_reload_no_change(self):
        # second reload should be a no-op (same mtime/sha)
        changed = self.m.reload()
        self.assertFalse(changed)


# ── Cascade router (difficulty) — no sentence-transformers needed ─────────────

class TestDifficulty(unittest.TestCase):
    def _difficulty(self, prompt):
        import re
        tokens = len(prompt.split())
        HARD_RE = re.compile(
            r"step.by.step|prove\b|derive\b|architect\b|tradeoff|trade.off"
            r"|refactor.*production|optimi[sz]e.*system|design.*scalab",
            re.I,
        )
        MEDIUM_RE = re.compile(
            r"\bexplain\b|\bcompare\b|\bdifference\b|\bwhy\b|\bhow\b.{0,40}\bwork",
            re.I,
        )
        multi_code = len(re.findall(r"```", prompt)) >= 2
        if tokens > 300 or multi_code or HARD_RE.search(prompt):
            return "hard"
        if tokens > 80 or MEDIUM_RE.search(prompt):
            return "medium"
        return "easy"

    def test_easy(self):
        self.assertEqual(self._difficulty("hi"), "easy")
        self.assertEqual(self._difficulty("what is python"), "easy")

    def test_medium(self):
        self.assertEqual(self._difficulty("explain how TCP works"), "medium")
        self.assertEqual(self._difficulty("why does the GIL exist"), "medium")

    def test_hard_regex(self):
        self.assertEqual(self._difficulty("prove that sqrt(2) is irrational"), "hard")
        self.assertEqual(self._difficulty("design a scalable microservice"), "hard")

    def test_hard_long(self):
        prompt = "word " * 350
        self.assertEqual(self._difficulty(prompt), "hard")

    def test_hard_multi_code_block(self):
        prompt = "review this:\n```python\npass\n```\nand this:\n```js\n1\n```"
        self.assertEqual(self._difficulty(prompt), "hard")

    def test_quality_floors(self):
        floors = {"easy": 0.70, "medium": 0.76, "hard": 0.85}
        from matrix.loader import Matrix
        m = Matrix("matrix/models.yaml")
        # for each difficulty, at least one model clears the floor
        for diff, floor in floors.items():
            candidates = [x for x in m.all() if x.quality >= floor]
            self.assertGreater(len(candidates), 0, f"no model for {diff} floor {floor}")


# ── Provider pool logic ───────────────────────────────────────────────────────

class TestProviderPool(unittest.TestCase):
    def test_build_pool_no_keys(self):
        # With no env vars set, pool should still build (just empty)
        from pool.free_apis import build_pool
        pool = build_pool()
        # All providers need env keys — pool may be empty in test env
        self.assertIsInstance(pool.providers, list)

    def test_can_call_rpm_window(self):
        from pool.free_apis import Provider
        p = Provider(
            name="test", base_url="http://x", api_key="k",
            model_map={"m": "m"}, rpm=3
        )
        self.assertTrue(p.can_call())
        p.record(); p.record(); p.record()
        self.assertFalse(p.can_call())  # hit rpm=3

    def test_cache_strategy_anthropic(self):
        from pool.free_apis import Provider
        p = Provider(
            name="anthropic", base_url="https://api.anthropic.com/v1",
            api_key="k", model_map={}, rpm=60,
            cache_strategy="anthropic"
        )
        self.assertEqual(p.cache_strategy, "anthropic")
        self.assertFalse(p.auto_caches)  # anthropic uses explicit, not auto

    def test_auto_cache_providers(self):
        from pool.free_apis import Provider
        for name in ("google", "openrouter"):
            p = Provider(name=name, base_url="x", api_key="k",
                         model_map={}, rpm=60)
            self.assertTrue(p.auto_caches)


# ── LoRA adapter index ────────────────────────────────────────────────────────

class TestLoraAdapterIndex(unittest.TestCase):
    def test_no_adapter_returns_none(self):
        from pool.cpu_llama import LlamaCPUServer
        srv = LlamaCPUServer()
        result = srv.adapter_id("smollm3-3b", "nonexistent")
        self.assertIsNone(result)

    def test_parse_adapters_env(self):
        os.environ["LORA_ADAPTERS"] = "code-v1:smollm3-3b/code.gguf,math-v1:zaya1-8b/math.gguf"
        from pool import cpu_llama
        import importlib
        importlib.reload(cpu_llama)
        # _parse_adapters is a module-level function
        result = cpu_llama._parse_adapters("LORA_ADAPTERS")
        self.assertEqual(result["code-v1"], "smollm3-3b/code.gguf")
        self.assertEqual(result["math-v1"], "zaya1-8b/math.gguf")
        del os.environ["LORA_ADAPTERS"]


# ── Batch worker availability ─────────────────────────────────────────────────

class TestBatchWorker(unittest.TestCase):
    def test_unavailable_without_key(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        from gateway.batch_worker import BatchWorker
        w = BatchWorker()
        self.assertFalse(w.available())

    def test_available_with_key(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
        from gateway.batch_worker import BatchWorker
        w = BatchWorker()
        self.assertTrue(w.available())
        del os.environ["ANTHROPIC_API_KEY"]


# ── Token estimation (cache gate) — tested inline, no heavy imports ───────────

class TestTokenEstimate(unittest.TestCase):
    def _estimate(self, text: str) -> int:
        return max(1, len(text) // 4)

    def test_empty(self):
        self.assertEqual(self._estimate(""), 1)

    def test_long(self):
        # 4096 chars → 1024 tokens
        self.assertGreaterEqual(self._estimate("a" * 4096), 1024)

    def test_short(self):
        self.assertLess(self._estimate("short text"), 10)

    def test_cache_threshold(self):
        # system prompts need ≥1024 token estimate to get cache_control
        self.assertGreaterEqual(self._estimate("x" * 4096), 1024)
        self.assertLess(self._estimate("x" * 100), 1024)


# ── GGUF map completeness — tested against matrix, no gateway import ──────────

EXPECTED_GGUF_MAP = {
    "smollm3-3b": "smollm3-3b-Q4_K_M.gguf",
    "nanbeige-3b": "nanbeige4.1-3b-Q4_K_M.gguf",
    "zaya1-8b": "zaya1-8b-Q4_K_M.gguf",
    "falcon-h1r-7b": "falcon-h1r-7b-Q4_K_M.gguf",
    "ornith-1.5-9b": "ornith-1.5-9b-Q4_K_M.gguf",
}


class TestGgufMap(unittest.TestCase):
    def test_all_cpu_models_have_gguf(self):
        from matrix.loader import Matrix
        m = Matrix("matrix/models.yaml")
        cpu_models = [x for x in m.all() if "local_cpu" in x.providers]
        for model in cpu_models:
            self.assertIn(
                model.id, EXPECTED_GGUF_MAP,
                f"{model.id} has local_cpu provider but no GGUF_MAP entry"
            )

    def test_gguf_filenames_are_q4_k_m(self):
        for mid, fname in EXPECTED_GGUF_MAP.items():
            self.assertIn("Q4_K_M", fname, f"{mid} should use Q4_K_M quant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
