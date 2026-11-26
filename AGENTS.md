# AGENTS.md — `robo_vlm_qc` Engineering & Contributor Rules

> **Version**: `2.0` | **Last Synced**: `2026-11-26`  
> Authoritative architecture specification: [`docs/architecture.md`](docs/architecture.md) · Operations & API reference: [`docs/operations.md`](docs/operations.md)

This document defines the repository structure, development toolchain, data-safety redlines, and Living Documentation rules for contributors and AI coding agents working on `robo_vlm_qc`.

---

## 1. Repository Architecture & Core Module Map

`robo_vlm_qc` is a standalone, VLM-native embodied trajectory QC, dual-role human review, and multi-format robotics data platform:

- **Configuration Center ([`configs/`](configs/))** — bundled into wheels as `robo_vlm_qc._bundled_configs` via [`pyproject.toml`](pyproject.toml):
  - [`configs/default.yaml`](configs/default.yaml): Global OpenAI-compatible VLM defaults (`https://api.openai.com/v1`, `OPENAI_API_KEY`, `.env`, `gpt-4o`, `thinking: false`, `thinking_mode: auto`), resilience guards (`circuit_breaker`, Gemini `mediaResolution` & blind-node token-density guard, `pricing`), stage model routing (`stage_models`), dynamic duration token budget (`5000 tokens/s`), S1–S5 sampling & BAFT tracking params (`batch_size: 4`, `track_width: 480`, `boundary_idle_kin_skip_s4b: true`, `start_end_compare: true`), I/O defaults, guardrail thresholds (`splice_gate`, `hallucination`, `visual_gate`), review lease/coverage rules, and export settings (`export.robot_type: "bimanual_robot"`).
  - [`configs/presets/`](configs/presets/): Ready-to-use provider overrides ([`openai.yaml`](configs/presets/openai.yaml), [`dashscope.yaml`](configs/presets/dashscope.yaml), [`gemini.yaml`](configs/presets/gemini.yaml), [`vllm.yaml`](configs/presets/vllm.yaml)).
  - [`configs/labels.yaml`](configs/labels.yaml): 11 pure-visual core labels (`version: 1`, 7 categories) using language-neutral `name` fields and decoupled from rule operators.
  - [`configs/prompts/`](configs/prompts/): Embodiment-agnostic (robot arms/grippers & human hands) stage-isolated YAML prompt templates (`s1_grounding.yaml`, `s2_macro.yaml`, `s3_discover.yaml`, `s4_verify.yaml`, `s5_success.yaml`).
  - [`configs/tasks/`](configs/tasks/): Generic `TaskSpec` definitions ([`_default.yaml`](configs/tasks/_default.yaml), [`pick_and_place.yaml`](configs/tasks/pick_and_place.yaml), [`bimanual_handover.yaml`](configs/tasks/bimanual_handover.yaml)).
- **Python Backend ([`robo_vlm_qc/`](robo_vlm_qc/))**
  - [`config.py`](robo_vlm_qc/config.py) & [`cli.py`](robo_vlm_qc/cli.py): Layered YAML loader (`ROBO_VLM_QC_CONFIG_DIR` → repo `configs/` → bundled `_bundled_configs` → `cwd/configs`, merged with `<workspace>/configs_override/`), provider-isolated SHA256 cache bases, and Click CLI (`ingest` w/ `--task`, `run-vlm`, `render-mp4`, `export` w/ `--state-schema-mode` & `--split-by-subtask`, `serve`, `eval`, `config-vlm`).
  - `models/`: Authoritative Pydantic v2 domain models (`episode.py`, `annotation.py`, `review.py`, `taxonomy.py`, `export.py`) with `extra="forbid"`, clean English fields (`name`, `title`, `label`, `instruction`) plus backward-compatible legacy alias normalization on input, and explicit rejection of abolished rule fields.
  - `io/`: Read-only 3-In `InputAdapter` registry (`h5_raw_v3.py`, `lerobot.py`, `video_dir.py`), `base.py` (`MediaSource`, `KinematicsTrack`), and `media.py` (adaptive 2×2 / 1×2 tiler, widened `DEFAULT_MAIN_ZOOM`, `compose_start_end_compare`, and `PreviewCache`).
  - `vlm/`: `VlmClient` (`client.py` — OpenAI-compat + native Gemini `/v1beta`, blind-node density guard, `VlmCircuitBreaker`, `redact_secrets`, cooperative `cancel()`), content-addressed `StageCache` (`cache.py`), `PromptBuilder` (`prompt_builder.py` — 30 Hz kinematics summaries, grounding & QC summaries), `tracker.py` (offline BAFT tracker w/ direct ORB+RANSAC camera-motion detection, hallucinated-anchor rejection, single-switchpoint DP, anti-jitter adaptive anchor weighting), stages (`stage1_grounding.py`, `stage2_macro.py`, `stage3_discover.py`, `stage4_verify.py`, `stage5_success.py`), and 3-level concurrent `VlmDag` (`dag.py`).
  - `guardrail/`: Light post-processing gates (`splice_gate.py` w/ edge-exact `boundary_idle` snapping, arm-aware `hallucination.py`, `visual_gate.py`) — zero kinematic rule detectors.
  - `store/`: SQLite WAL connection pool & 6-table schema (`db.py`), atomic JSON + `.history/` snapshots (`artifacts.py`), and domain `Repository` (`repository.py` — claim leases, non-destructive `task_id` reassignment, cancel rollback).
  - `lab/`: Pure In-Context `VLM Lab` engine (`diagnosis.py`, `few_shot.py`, `eval_runner.py`, `prompt_assist.py`, `task_assist.py` multimodal advisory `TaskSpec` drafter).
  - `export/`: Transactional `_staging → rename` base & `OutputExporter` registry (`base.py`), unified `PieceSlicer` (`slicer.py` w/ optional `split_by_subtask`), and 4 exporters (`lerobot_v3.py` w/ adaptive `state_schema_mode`, `h5_clean.py`, `zero_copy_index.py`, `viz_mp4.py`).
  - `api/`: FastAPI application (`app.py`), Reviewer/Lead RBAC (`auth.py`), shared `AppState` & SSE broadcaster (`state.py`), and 7 routers (`episodes.py`, `reviews.py`, `jobs.py`, `exports.py`, `lab.py`, `settings.py`, `ingest.py`).
- **Unified Dual-Role Bilingual Frontend ([`web/src/`](web/src/))**
  - Pages: `QueuePage.tsx`, `ReviewStudioPage.tsx`, `ExportPage.tsx`, `VlmLabPage.tsx` (4 tabs: `diagnosis`, `few_shots`, `taxonomy`, `eval`).
  - Shared Studio Components: `StudioPlayer.tsx` (30fps rVFC 2D Grounding canvas + `utils/groundingProjection.ts`), `FourTrackTimeline.tsx`, `SegmentInspector.tsx`, `SuccessPanel.tsx`, `DiffOverlay.tsx`, `IngestModal.tsx`, `TaskDraftWizardModal.tsx`, `VlmSettingsModal.tsx`, `JobCenterDrawer.tsx`, `LabHistoryDrawer.tsx`, `ShortcutsModal.tsx`.
  - Typed API, Bilingual i18n & State: `api/{client,types,sse}.ts`, `i18n/{common,studio,lab}.ts`, `stores/{authStore,i18nStore,jobStore,reviewHistoryStore}.tsx`, `constants/labels.ts`.
- **Scripts & Tests ([`scripts/`](scripts/), [`tests/`](tests/))**
  - [`scripts/generate_demo_dataset.py`](scripts/generate_demo_dataset.py): Standalone synthetic `video_dir` + `lerobot` v3 demo data generator.
  - Semantic test suites organized under `tests/unit/`, `tests/api/`, and `tests/e2e/`.

---

## 2. Dev, Test & Build Toolchain

- **Python / Pytest** (Python `>= 3.12`, `ffmpeg` on `PATH`):
  ```bash
  # Install editable package with dev dependencies
  pip install -e ".[dev]"

  # Run full repository test suite
  pytest tests -q

  # Run semantic suites individually
  pytest tests/unit -q
  pytest tests/api -q
  pytest tests/e2e -q
  ```
- **Frontend Test & Build** (Node.js `>= 22`): Any modification under [`web/src/`](web/src/) must pass typecheck, Vitest unit/integration tests, and emit `web/dist/` cleanly:
  ```bash
  cd web
  npm ci
  CI=true npm test && npm run build
  ```
- **Wheel Packaging & Demo Smoke Check**:
  ```bash
  python scripts/generate_demo_dataset.py --out /tmp/demo_data
  robo-vlm-qc -w /tmp/demo_ws ingest --data /tmp/demo_data --json
  ```

---

## 3. Data Safety & Architectural Redlines

1. **Strict Source Isolation (Read-Only)**: Never modify, rename, or delete anything inside `--data` source datasets.
2. **Credential Secrecy**: Never print, log, commit, or embed API keys (`OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `GEMINI_API_KEY`, `.env` contents) or auth tokens in code, tests, error messages, logs, or documentation.
3. **Server Security Defaults**: `robo-vlm-qc serve` must bind `127.0.0.1` by default, require explicit tokens (`--reviewer-token` / `--lead-token` or `ROBO_VLM_QC_*_TOKEN`) unless `--no-auth` is passed, enforce configurable CORS origins (`--cors-origin` / `ROBO_VLM_QC_CORS_ORIGINS`), and enforce `--allowed-data-roots` containment when configured.
4. **Atomic Persistence & `.history/` Protection**: All workspace JSON/YAML overrides and artifacts must go through `robo_vlm_qc/store/artifacts.py` (`atomic_write`, `archive_before` → `<workspace>/.history/`), and all dataset exports must go through `staged_export_dir` (`_staging → os.replace`).
5. **GT Freeze & Lease Protection**: Batch VLM runs and review writes must respect `gt_frozen` immutability and SQLite claim lease ownership (`store/repository.py`). All tests must run offline using `tmp_path` workspaces and mock VLM servers.
6. **No Kinematic Rule Detectors or Abolished Fields**: Never reintroduce `long_pause`, `back_and_forth`, 30 Hz kinematic rule detectors, or fragmented multi-file truth stores. `models/annotation.py` actively rejects abolished legacy fields.
7. **Read-Only VLM Visual Layers**: Human reviews (`ReviewWritePayload`) may edit only the 3 editable layers (`qc_segments`, `success`, `macro_stages`). `grounding`, `arm_timelines`, and `raw_vlm_segments` must remain read-only.
8. **Zero VLM SFT/DPO/LoRA Export**: `VLM Lab` (`robo_vlm_qc/lab/`, `/api/lab/*`, `VlmLabPage.tsx`) is **strictly a pure In-Context iteration loop** (error diagnosis + 2×2 visual/text Few-Shot mistake book + versioned taxonomy/prompts/tasks + GT regression eval sandbox). Never implement VLM SFT, DPO, LoRA, PEFT, or RLHF fine-tuning export in this repository.

---

## 4. Living Documentation Rule

Code and authoritative documentation must stay synchronized within the same task or pull request. Whenever any scope below changes, update the mapped documentation files before concluding work:

| Code / Config Scope Changed | Required Documentation Updates |
| :--- | :--- |
| `robo_vlm_qc/models/`, `store/db.py`, `vlm/dag.py`, `vlm/cache.py`, `guardrail/` | Sync [`docs/architecture.md`](docs/architecture.md) (schemas, DAG, cache keys, guardrail thresholds). |
| `robo_vlm_qc/io/`, `export/`, `configs/default.yaml`, `configs/labels.yaml`, `configs/presets/` | Sync [`README.md`](README.md), [`README_zh.md`](README_zh.md), [`docs/architecture.md`](docs/architecture.md), and [`docs/operations.md`](docs/operations.md). |
| `robo_vlm_qc/cli.py`, `api/`, `lab/`, `web/src/`, `configs/{prompts,tasks}/` | Sync [`README.md`](README.md), [`README_zh.md`](README_zh.md), and [`docs/operations.md`](docs/operations.md) (CLI flags, `/api/*` contracts, Studio & `VLM Lab` SOP). |
| New top-level packages, packaging, CI/Docker, or architectural invariants | Sync [`AGENTS.md`](AGENTS.md) (§1–§4) and bump the `Last Synced` header date. |

### Pre-Completion Documentation Checklist

1. Verify all relative Markdown links across [`README.md`](README.md), [`README_zh.md`](README_zh.md), [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/architecture.md`](docs/architecture.md), and [`docs/operations.md`](docs/operations.md) resolve to existing repository files.
2. Verify all Mermaid diagram node labels use double quotes (`NodeId["Label"]`) and accurately reflect implemented code.
3. Verify zero claims of VLM SFT/DPO/LoRA export capability exist anywhere in code or docs.
4. Run `pytest tests -q` and `cd web && CI=true npm test && npm run build` and confirm 100% pass.
