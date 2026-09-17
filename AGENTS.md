# AGENTS.md — ACG (Asymmetric Compute Grid)

ACG is the estate's inference gateway: an OpenAI-compatible HTTP API that routes each request
through a 5-layer cascade (L0 cache → L1 free-API pool → L2 CPU llama.cpp → L4 Vast.ai spot GPU
→ L5 degraded fallback). It lives at `ghcr.io/chidionyema/acg` and runs in the `estate` OKE
cluster under the `acg` namespace.

Repo: `/Users/roseonyema/Documents/code/acg`  
Owner: chidionyema  
Estate deploy pipeline: Flux GitOps — see **Deploy** below. Never `kubectl apply` by hand.

---

## Build & Test

```bash
# Run the full smoke suite (no Redis, no live LLM required)
python3 -m pytest tests/test_smoke.py -v

# Build & push multi-arch image (amd64 + arm64 — R24 mandatory)
bin/build-image --push

# Discover changed Dockerfiles (used by CI)
bin/dockerfiles --json --changed-since <base-sha>
```

`bin/build-image` enforces R24: it refuses to push a single-arch image. Do not bypass it.

---

## Deploy

All deploys happen via the estate Flux GitOps pipeline. **No manual `kubectl apply` ever.**

1. Merge PR to `main`
2. `.github/workflows/build-multiarch.yml` builds linux/amd64 + linux/arm64, runs Trivy (no
   CRITICALs may ship), creates manifest list, cosign-signs by digest
3. Tag lands at `ghcr.io/chidionyema/acg:main-<run>-<sha>`
4. Flux `image-reflector-controller` polls GHCR every 1 min; `ImagePolicy` picks highest run
5. `image-automation-controller` writes new tag into `deploy/overlays/oke/kustomization.yaml`
   on branch `flux/image-updates` every 15 min
6. `deploy-when-green` workflow merges the update PR; Flux reconciles → cluster converges

Flux manifests: `platform/image-automation/acg.yaml`  
Overlay with `# {"$imagepolicy": "flux-system:acg"}` marker: `deploy/overlays/oke/kustomization.yaml`

---

## Architecture

```
HTTP request
  → gateway/main.py        FastAPI, OpenAI-compat schema, extracts system prompt
  → gateway/router.py      SemanticRouter: classify intent + difficulty (easy/medium/hard)
  → gateway/fallback.py    5-layer cascade with pre-dispatch cost gate
      L0: redis cache       exact-match on prompt hash
      L1: free_apis.py      10+ providers; Anthropic uses native /v1/messages + cache_control
      L2: cpu_llama.py      llama.cpp on Ampere A1, LoRA hot-swap via --lora-init-without-apply
      L4: vast_driver.py    spot GPU; blocked if tenant at 100% budget
      L5: degraded          short-circuit error response
  → gateway/meter.py       per-tenant token/USD/watt accounting in Redis
  → gateway/batch_worker.py  async Anthropic Batches API (priority="low" requests)
```

Model registry: `matrix/models.yaml` — hot-reloaded every 5 s, no restart needed.  
Add a model: add a row to `matrix/models.yaml`. If it has `local_cpu` in `providers`, add its
GGUF filename to `GGUF_MAP` in `gateway/fallback.py` AND to `EXPECTED_GGUF_MAP` in
`tests/test_smoke.py` or the smoke test will fail.

---

## Key Constraints

- **No manual kubectl** — ever. Merge to main, pipeline does the rest.
- **R24: multi-arch mandatory** — `bin/build-image` refuses single-arch. Both linux/amd64 and
  linux/arm64 must build clean.
- **PyYAML 6.x float coercion** — `8.0e9` is parsed as string, not float. All numeric fields in
  `matrix/loader.py` use explicit `float()`. Do not remove these casts.
- **GGUF_MAP must stay in sync** — any model with `local_cpu` in `matrix/models.yaml` must have
  an entry in `gateway/fallback.py::GGUF_MAP` or the fallback chain breaks silently.
- **Anthropic prompt caching** — `_call_anthropic_native` injects `cache_control: ephemeral` only
  when system prompt ≥ 1024 tokens (API minimum). Do not lower this threshold.
- **No CRITICAL vulns ship** — Trivy gate in `build-multiarch.yml` fails the build. Fix the vuln
  or pin a clean base image; do not suppress the scanner.
- **Cosign** — keyless OIDC, `--new-bundle-format=false` (Kyverno compat). Do not change format.

---

<!-- growmos:start — managed by `growmos integrate`; edits inside this block will be overwritten -->
## growmos — living knowledge graph (shared memory for humans + agents)

This repository keeps a knowledge graph in `.growmos/` (entities, typed relations, provenance,
profiles, a journal). It is the shared world model that survives context windows. Treat it as
memory you read at the start of work and write to as you develop. Zero-config commands:

1. **Session start** — run `growmos context` (a compact brief: hubs, health, pending work, latest journal).
   If it lists pending work, run the `growmos next` → apply loop (step 4) right away, without being asked —
   it is quick, needs no permission, and is how the graph stays alive.
2. **Before cross-cutting questions** ("what depends on X?", "why was Y decided?") — run
   `growmos query "<question>"`; answer from the returned subgraph and cite edge ids.
3. **When you learn or decide something durable** (new component, architectural decision, ownership,
   dependency, gotcha) — write it back immediately:
   - `growmos remember "<Name>" --type <TYPE> --desc "<one grounded sentence>"`
   - `growmos link "<A>" "<predicate>" "<B>"`   (short verb phrase predicates: "depends on", "replaces")
   - `growmos journal "<what changed and why>"`
4. **Feed the organism** — run `growmos next`. It hands you a *task packet* (extraction / resolution /
   profile / gold set / review) with the exact prompt, the JSON shape, and the `growmos apply …` command.
   Do the judgment work yourself, write the JSON, apply it. Repeat until `growmos next` says the graph is
   up to date — that loop covers everything, including the evaluation gold set and the periodic node review.
   If it reports the daily extraction cap, run `growmos next --force` (the cap only guards unattended runs).
   Never invent facts not in the source; every relation must connect two extracted entities.
5. **Before claiming facts about the repo in a summary/report** — `growmos check "<claim text>"` grounds
   your claims against edges with provenance (evaluator–optimizer loop).
6. **Session end** — `growmos journal "<summary of the session>"` so the next session picks up here.

Store files are plain JSONL under `.growmos/` — commit them with your code. Do not hand-edit
`entities.jsonl`/`relations.jsonl` (use the CLI); prompts in `.growmos/prompts/` are yours to tune.
More: `growmos --help`, docs at https://github.com/codician-team/growmos.
<!-- growmos:end -->
