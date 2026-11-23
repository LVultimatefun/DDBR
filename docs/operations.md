# `robo_vlm_qc` Operations Guide — CLI, Config, API, Studio & `VLM Lab`

> **Version**: `2.0` | **Last Synced**: `2026-11-23`  
> System design & contracts: [`docs/architecture.md`](architecture.md) · Engineering rules: [`AGENTS.md`](../AGENTS.md)

---

## 1. Complete CLI Reference (`robo-vlm-qc` / `python -m robo_vlm_qc`)

Global options (placed before the subcommand):

- `--workspace`, `-w <PATH>`: Workspace directory for SQLite DB, `stage_cache/`, `artifacts/`, `configs_override/`, `exports/`, `.history/` (default: `./workspace`).
- `--config-dir`, `-c <PATH>`: Optional override path for the base `configs/` directory (also configurable via `ROBO_VLM_QC_CONFIG_DIR`).

### 1.1 `ingest` — Register Read-Only Datasets into SQLite

```bash
robo-vlm-qc -w ./workspace ingest \
  --data /path/to/dataset_or_file \
  [--data /path/to/second_dataset] \
  [--adapter auto|h5_raw_v3|lerobot|video_dir] \
  [--dataset my_dataset_name] \
  [--dry-run] \
  [--json]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--data` | *(required, repeatable)* | Read-only file or directory to scan. |
| `--adapter` | `auto` | Force specific `InputAdapter` or auto-detect by priority (`h5_raw_v3` → `lerobot` → `video_dir`). |
| `--dataset` | `None` | Override inferred `dataset` identifier (`episode_id = "<dataset>/<stem>"`). |
| `--dry-run` | `false` | Discover and validate metadata without writing to SQLite. |
| `--json` | `false` | Emit structured JSON summary instead of Rich table. |

### 1.2 `run-vlm` — Execute Causal VLM DAG & Light Guardrails

```bash
robo-vlm-qc -w ./workspace run-vlm \
  [--episode ds/ep_0001] \
  [--dataset ds] \
  [--stage all|s1|s2|s3|s4|s4a|s4b|s5] \
  [--fresh] \
  [--workers 4] \
  [--json]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--episode` | `()` (repeatable) | Target specific `episode_id`(s). |
| `--dataset` | `None` | Filter ingested episodes by dataset name. |
| `--stage` | `all` | Re-run a specific stage and its downstream dependents (`s1`..`s5`, `s1_grounding`, `s2_macro`, `s3_discover`, `s4a`, `s4b`, `s4_verify`, `s5_success`). |
| `--fresh` | `false` | Bypass `StageCache` and force fresh VLM calls. |
| `--workers` | `4` | Concurrent episode worker count. |
| `--json` | `false` | Emit structured JSON report. |

### 1.3 `render-mp4` — Render 1080p Studio HUD MP4

```bash
robo-vlm-qc -w ./workspace render-mp4 \
  [--episode ds/ep_0001 | --all-ready] \
  [--out /path/to/file_or_dir] \
  [--json]
```

Requires either `--episode` or `--all-ready`. When `--out` is omitted, renders to `<workspace>/exports/viz_mp4/<dataset>__<stem>.mp4`.

### 1.4 `export` — Slice & Export Curated Datasets

```bash
robo-vlm-qc -w ./workspace export \
  --format lerobot_v3|h5_clean|zero_copy_index|viz_mp4 \
  --recipe bc_clean|recovery_plus|badcase_pairs \
  --out ./exports/out_bundle \
  [--dataset ds] [--task task_id] [--episode ds/ep_0001] \
  [--status approved --status gt_frozen] \
  [--chunk-size 100] [--tail-frames 15] [--smooth-tail on_jump|always|never] \
  [--min-piece-s 0.1] [--context-window-s 2.0] \
  [--dry-run] [--json]
```

Default `--status` includes `approved` and `gt_frozen` (`EpisodeStatus` choices: `ingested`, `vlm_queued`, `vlm_running`, `vlm_ready`, `claimed`, `reviewed`, `flagged`, `approved`, `gt_frozen`).

### 1.5 `serve` — Start Unified FastAPI + React Dual-Role Studio

```bash
robo-vlm-qc -w ./workspace serve \
  [--host 127.0.0.1] [--port 8000] \
  [--reviewer-token <tok> --lead-token <tok> | --no-auth] \
  [--cors-origin http://localhost:5173] \
  [--allowed-data-roots /data/datasets_a --allowed-data-roots /data/datasets_b]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind host address. Use `--host 0.0.0.0` explicitly for Docker or trusted LAN deployments. |
| `--port` | `8000` | Bind TCP port (`1..65535`). |
| `--reviewer-token` | `None` | Reviewer role token (or `ROBO_VLM_QC_REVIEWER_TOKEN`). |
| `--lead-token` | `None` | Lead role token (or `ROBO_VLM_QC_LEAD_TOKEN`). |
| `--no-auth` | `false` | Disable token authentication for local loopback development. |
| `--cors-origin` | `()` (repeatable) | Explicit allowed CORS origin (or comma-separated `ROBO_VLM_QC_CORS_ORIGINS`; defaults to localhost/127.0.0.1 dev ports `5173` and `8000`). |
| `--allowed-data-roots` | `()` (repeatable) | Optional read-only directory allowlist enforced by `GET /api/ingest/browse` and async `POST /api/jobs (job_type="ingest")`. Paths outside configured roots (including `..`, prefix collisions, and escaping symlinks) are rejected with `403`. |

Authentication fails closed: both `robo-vlm-qc serve` and `create_app()` require `--reviewer-token` / `--lead-token` (or `ROBO_VLM_QC_*_TOKEN` env vars) unless `--no-auth` / `no_auth=True` is explicitly provided.

### 1.6 `eval` — Run `VLM Lab` GT Regression Sandbox

```bash
robo-vlm-qc -w ./workspace eval \
  [--dataset ds] [--task task_id] [--episode ds/ep_0001] \
  [--prompt-ver v2] [--label-ver 2] \
  [--few-shot | --no-few-shot] \
  [--workers 4] [--json]
```

Evaluates all matching `gt_frozen` episodes in sandbox mode (`is_eval_sandbox=True`) without mutating episode status or `active_run_id`, reusing S1/S2/S5 `StageCache`. Exits with code `2` if the workspace DB is missing or no `gt_frozen` episodes exist.

---

## 2. Configuration, Provider Presets & 11 Pure-Visual Labels

### 2.1 Layered Overrides & `.history/` Snapshots

[`ConfigManager`](../robo_vlm_qc/config.py) resolves the base config directory in order (`--config-dir` → `ROBO_VLM_QC_CONFIG_DIR` → repo `configs/` → wheel-bundled `robo_vlm_qc._bundled_configs` → `cwd/configs`) and overlays `<workspace>/configs_override/`:
- `default.yaml` and `tasks/<task_id>.yaml`: deep-merged over base YAML.
- `labels.yaml` and `prompts/<stage>.yaml`: replaced when overridden.
- All programmatic override writes use `archive_and_write_override` / `ArtifactStore.write_config_override`, saving a timestamped JSON snapshot of the prior state under `<workspace>/.history/` before atomic replacement.

### 2.2 Key Settings in [`configs/default.yaml`](../configs/default.yaml) & Provider Presets

- **Default VLM Client**: `provider: openai_compat`, `base_url: https://api.openai.com/v1`, `api_key_env: OPENAI_API_KEY`, `key_file: .env`, `thinking_mode: auto`, `concurrency: 32`, `episode_concurrency: 4`, `decode_concurrency: 6`.
- **Stage Model Routing (`stage_models`)**: `s1_grounding`, `s2_macro`, `s3_discover`, `s4_verify`, `s5_success` (default `gpt-4o`, `thinking: false`; customizable per stage).
- **Provider Presets ([`configs/presets/`](../configs/presets/))**: Copy any preset to `<workspace>/configs_override/default.yaml`:
  - [`openai.yaml`](../configs/presets/openai.yaml) (`gpt-4o`, `OPENAI_API_KEY`, `thinking_mode: disabled`)
  - [`dashscope.yaml`](../configs/presets/dashscope.yaml) (`qwen-vl-max`, `DASHSCOPE_API_KEY`, `thinking_mode: dashscope`, `thinking: true`)
  - [`gemini.yaml`](../configs/presets/gemini.yaml) (`provider: gemini`, `gemini-2.5-pro`, `GEMINI_API_KEY`, `thinking: true`)
  - [`vllm.yaml`](../configs/presets/vllm.yaml) (`http://127.0.0.1:8001/v1`, `Qwen/Qwen2.5-VL-72B-Instruct`, `VLLM_API_KEY`, `thinking_mode: disabled`)
- **Example TaskSpecs ([`configs/tasks/`](../configs/tasks/))**: [`_default.yaml`](../configs/tasks/_default.yaml), [`pick_and_place.yaml`](../configs/tasks/pick_and_place.yaml), [`bimanual_handover.yaml`](../configs/tasks/bimanual_handover.yaml).
- **Dynamic Budget**: `tokens_per_second: 5000`, `min_episode_tokens: 200000`, `max_tokens_per_episode: null`, `unlimited: false`.
- **Sampling & Few-Shot**: `s3_discover` (`window_s: 16.0`, `window_stride_s: 12.0`, `window_step_s: 0.8`, `few_shot_top_k: 4`, `few_shot_max_tokens: 4000`), `s4_verify` (`refine_margin_s: 1.5`, `refine_step_s: 0.5`), `s5_success` (`tail_s: 10.0`, `control_ratio: 0.30`).
- **Review & Export Policy**: `claim_ttl_s: 120.0`, `min_watch_coverage: 0.95`, `require_confirmed_success: true`, `export.robot_type: "bimanual_robot"`.

### 2.3 Core Pure-Visual Taxonomy ([`configs/labels.yaml`](../configs/labels.yaml), `version: 1`)

| Label ID | Name | Category | Default Training Role | Default Splice Policy |
| :--- | :--- | :--- | :--- | :--- |
| `boundary_idle` | Boundary Idle | `idle` | `badcase` | `cut_clean` |
| `mid_idle_freeze` | Mid-Episode Freeze | `idle` | `badcase` | `cut_clean` *(guardrail verified)* |
| `missed_grasp_retry` | Missed Grasp Retry | `failure_recovery` | `recovery` | `mask_loss` |
| `object_slip_recovery` | Object Slip Recovery | `failure_recovery` | `recovery` | `mask_loss` |
| `placement_redo` | Placement Redo | `failure_recovery` | `recovery` | `mask_loss` |
| `hesitation_oscillation` | Hesitation Oscillation | `inefficiency` | `badcase` | `mask_loss` |
| `collision_or_knock` | Collision or Knock | `failure` | `badcase` | `mask_loss` |
| `precision_fine_adjust` | Precision Fine Adjust | `necessary_skill` | `necessary` | `keep_positive` |
| `bimanual_assist_hold` | Bimanual Assist Hold | `necessary_skill` | `necessary` | `keep_positive` |
| `terminal_failure` | Terminal Failure | `terminal` | `terminal_fail` | `mask_loss` |
| `unknown_anomaly` | Unknown Anomaly | `unknown` | `badcase` | `mask_loss` |

---

## 3. Canonical FastAPI REST / SSE & RBAC Reference

- **Authentication**: Supply `X-Auth-Token: <token>` header or `?token=<token>` query parameter (used by `<video>` / `<img>` tags, direct artifact downloads, and SSE `EventSource`), plus optional `X-User: <username>` header (defaults to role name). Missing/invalid tokens return `401`.
- **RBAC**: `Reviewer` receives `403` on `/api/lab/*`, `/api/exports/*`, `/api/ingest/*`, `PUT /api/settings`, `GET /api/episodes/{ep_id}/runs`, `GET /api/episodes/{ep_id}?run_id=...`, `/api/jobs` mutations/listing/artifacts (`POST /api/jobs`, `GET /api/jobs`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/artifact`), and review actions `approve` / `freeze_gt`.

### 3.1 Core Studio, Jobs, Settings & Ingest Endpoints

| Method & Route | Role | Purpose & Key Contract Details |
| :--- | :--- | :--- |
| `GET /api/health` | Public | Returns `{"status": "ok"}`. |
| `GET /api/auth/me` | Auth | Returns `{"role": "reviewer"\|"lead", "user": "..."}`. |
| `GET /api/labels` | Auth | Returns active `LabelRegistry` JSON. |
| `GET /api/settings` | Auth | Returns workspace Studio settings: `{"min_watch_coverage": 0.95, "batch_vlm_concurrency": 4}`. |
| `PUT /api/settings` | Lead | Body (`extra="forbid"`, at least one field): `{"min_watch_coverage"?: float [0.0..1.0], "batch_vlm_concurrency"?: int [1..32]}`. Atomically merges into `configs_override/default.yaml` (preserving unrelated keys, archiving prior state to `.history/`), hot-reloads `ConfigManager`, and publishes SSE `settings_updated`. |
| `GET /api/episodes` | Auth | Query params: `status` (single or comma-separated `EpisodeStatus`, e.g. `reviewed,flagged`), `dataset`, `task_id`, `adapter_type`, `passed`, `guardrail_downgraded`, `has_unknown`, `claimed_by`, `search` (escaped SQL `LIKE` substring on `episode_id`), `sort_by` (`episode_id`\|`updated_at`\|`duration_s`\|`cut_ratio`\|`mask_ratio`\|`badcase_count`, NULLs last), `sort_dir` (`asc`\|`desc`), `limit` (`1..10000`), `offset`. Returns `{items, total, my_claimed_count, status_counts, facets}` where `status_counts` computes SQL `GROUP BY status` across all 9 statuses unaffected by `status`/`claimed_by`, and `facets` returns workspace-global `{datasets, task_ids, adapter_types}`. |
| `POST /api/episodes/{ep_id}/check-splice` | Auth | Body (`extra="forbid"`): `{t_start, t_end, label_id, splice_policy, affected_arm}`. Uses thread-safe in-memory LRU `KinematicsTrack` cache (`<20ms` warm recalc) to run `hallucination` + `splice_gate` (or `visual_gate` SSIM/L1 fallback when `has_kinematics=false`), returning `{passed, downgraded, downgrade_reason, effective_splice_policy, opt_t_start, opt_t_end, metrics, compute_ms, mode}` with zero DB mutation. |
| `GET /api/episodes/{ep_id}/runs` | Lead | Lists all historical `VlmRunArtifact` metadata for an episode (newest first): `{episode_id, active_run_id, runs: [{run_id, created_at, prompt_version, label_version, few_shot_pack_id, tokens_total, is_eval_sandbox}]}`. |
| `GET /api/episodes/{ep_id}` | Auth | Returns `ResolvedEpisode` + `claimed_by`, `claim_expires_at`, `guardrail_downgraded`, `selected_run_id`, `is_historical_run_view`. Optional `?run_id=...` (Lead-only; `403` for Reviewer, `404` on missing/cross-episode run) loads a read-only historical run snapshot (`active_rev_id=null`, `is_human_reviewed=false`, `is_historical_run_view=true`) without mutating DB state. |
| `GET /api/episodes/{ep_id}/frame` | Auth | Query: `camera?`, required float `t_s` (returns `image/jpeg`). |
| `GET /api/episodes/{ep_id}/preview` | Auth | Query: `camera?`, `hd: bool` (streams transcoded CFR H.264 MP4). |
| `POST /api/episodes/{ep_id}/claim`, `/renew`, `/release` | Auth | Body: `{"user": "<username>"}` (or uses `X-User`). `409` on active claim conflict or `gt_frozen`. |
| `GET /api/episodes/{ep_id}/reviews` | Auth | Lists immutable `HumanReviewArtifact` history (`items`, `total`). |
| `POST /api/episodes/{ep_id}/reviews` | Auth | Body (`ReviewWritePayload`, `extra="forbid"`): `{base_run_id, action, qc_segments, success, macro_stages, watch_coverage, notes?, flagged_reason?}`. Rejects `author` and read-only keys (`grounding`, `arm_timelines`, `raw_vlm_segments`) with `422`; `X-User` must match claimer (`409` otherwise); dynamically enforces workspace `min_watch_coverage` (`422`); `Reviewer` `approve`/`freeze_gt` → `403`. |
| `POST /api/jobs` | Lead | Unified async job engine (`201 Created`). Body (`extra="forbid"`): `{job_type: "batch_vlm"\|"render_mp4"\|"export"\|"eval_gt"\|"ingest", episode_ids?, stage?, fresh?, concurrency?, is_eval_sandbox?, export_params?, eval_params?, ingest_params?}`. `batch_vlm` runs multi-episode concurrently (`ThreadPoolExecutor`, default `batch_vlm_concurrency=4`) and supports partial rerun stages (`s3_s4`/`S3/S4` reusing S1/S2/S5 cache; `s1_s5`/`S1-S5` full fresh rerun; `all` respecting `fresh`; `s1..s5`). Publishes incremental SSE `job_update` events and registers downloadable `summary.artifacts`. |
| `GET /api/jobs`, `POST /api/jobs/{job_id}/cancel` | Lead | List recent jobs (`status?`, `job_type?`, `limit`, `offset`) or cancel an active `queued`/`running` job. |
| `GET /api/jobs/{job_id}/artifact` | Lead | Query: `index=0` (supports `X-Auth-Token` or `?token=`). Enforces strict workspace path-containment (`403` on escape) and streams rendered MP4 (`video/mp4`) or export bundle (`application/zip`). |
| `GET /api/jobs/{job_id}`, `/api/jobs/{job_id}/events`, `GET /api/events` | Auth | Job polling and native `StreamingResponse` `text/event-stream` SSE streams (`job_update`, `settings_updated`, `config_updated`, `episode_claimed`, `episode_released`, `review_saved`, `export_completed`). |
| `GET /api/ingest/browse` | Lead | Query: `path?`. Read-only directory browser constrained by `--allowed-data-roots` (`403` on escape/symlink traversal, `404` missing, `400` non-dir). Returns `{current_path, parent_path, allowed_roots, entries, detected_adapters}` with bounded 3-adapter episode count preflight (`h5_raw_v3`, `lerobot`, `video_dir`). |
| `POST /api/exports/dry-run`, `POST /api/exports` | Lead | Synchronous dry-run preview & transactional export (Web `ExportPage` submits async `POST /api/jobs {job_type: "export"}`). Body: `{format, recipe, chunk_size?, min_piece_s?, tail_frames?, smooth_tail?, include_statuses?, context_window_s?, dataset_filter?, task_filter?, episode_ids?}`. |
| `GET /api/exports`, `GET /api/exports/{id}`, `GET /api/exports/{id}/download` | Lead | Lists manifests, inspects manifest, or streams ZIP bundle. |

### 3.2 Lead-Only `VLM Lab` Endpoints (`/api/lab/*`)

| Method & Route | Purpose & Key Contract Details |
| :--- | :--- |
| `GET /api/lab/labels`, `PUT /api/lab/labels` | Read or replace `LabelRegistry` (rejects duplicate IDs with `422`; snapshots prior file to `.history/`). |
| `GET /api/lab/prompts`, `PUT /api/lab/prompts` | Read or update stage `PromptTemplate`s (`s1_grounding`..`s5_success`) with `.history/` backup. |
| `GET /api/lab/tasks`, `PUT /api/lab/tasks` | Read or update `TaskSpec` YAML overrides (`tasks/<task_id>.yaml`) with `.history/` backup. |
| `GET /api/lab/history` | Query: `target?` (`labels`\|`prompts`\|`tasks`\|`default`), `name?`. Lists `.history/` archived config snapshots newest-first: `{items: [{snapshot_id, target, name, relative_path, created_at, reason, size_bytes}], total}`. |
| `GET /api/lab/history/{snapshot_id}/diff` | Computes unified diff + structured line pairs (`hunk`\|`context`\|`add`\|`remove`) between snapshot and active config (`400` bad ID, `403` symlink escape, `404` missing). |
| `POST /api/lab/history/{snapshot_id}/rollback` | Validates snapshot YAML & Pydantic schema (`422` on invalid without modifying active config), archives current active config into `.history/`, atomically restores snapshot, hot-reloads `ConfigManager`, publishes SSE `config_updated`, and returns `{ok, relative_path, config_sha256}`. |
| `GET /api/lab/diagnosis` | Query: `episode_ids?`, `top_n=20`. Returns `DiagnosisReport` across all 6 buckets (`fp_overflag`, `fn_missed`, `policy_flip`, `boundary`, `success_flip`, `stage`). |
| `GET /api/lab/unknowns` | Query: `dataset?`, `limit=200`, `include_triaged=false`. Lists untriaged `unknown_anomaly` segments. |
| `POST /api/lab/unknowns/triage` | Body: `{episode_id, seg_id, action: "merge"\|"promote", target_label_id?, rationale?, new_label?}`. `merge` creates a Few-Shot counter-example with 2×2 tile + records `artifacts/unknown_triage.json`; `promote` adds new label (`409` if ID exists), increments `LabelRegistry.version`, writes `configs_override/labels.yaml` with `.history/`. |
| `GET /api/lab/few-shots`, `POST /api/lab/few-shots` | List/filter (`task_id`, `label_id`, `scope`, `enabled_only`) or create Few-Shot example (`t_end > t_start`, optional `capture_tile=true` renders `artifacts/few_shots/<example_id>.jpg`). |
| `PATCH /api/lab/few-shots/{example_id}` | Toggle `{"enabled": true\|false}`. |
| `GET /api/lab/few-shots/{example_id}/tile` | Serves rendered 2×2 JPEG comparison tile (`image/jpeg`). |
| `POST /api/lab/eval` | Synchronous GT sandbox eval (Web `VlmLabPage` submits async `POST /api/jobs {job_type: "eval_gt"}`). Body: `{dataset?, task_id?, episode_ids?, few_shot=true, top_k?, max_tokens?, workers=4, candidate_prompts?, candidate_labels?}`. Reuses S1/S2/S5 cache and writes `artifacts/eval_runs/<eval_id>.json`. |
| `GET /api/lab/eval/compare` | Query: `baseline?`, `candidate?`. Returns eval history + side-by-side metric deltas. |
| `POST /api/lab/promote` | Body: `{eval_id?, dataset?, candidate_prompts?, candidate_labels?, candidate_tasks?, reason?}`. Archives base configs to `.history/` and promotes overrides to `configs_override/` (`404` if `eval_id` not found). |
| `POST /api/lab/prompt-assist` | Body: `{stage="s3_discover", task_id?, episode_ids?, top_n=10}`. Advisory LLM prompt diff suggestions; never auto-writes configs. |

---

## 4. Dual-Role Studio & `VLM Lab` In-Context SOP

### 4.1 Unified Dual-Role Studio ([`web/src/`](../web/src/))

Single React 19 SPA with 4 pages (`/` `QueuePage`, `/review/:dataset/:stem` `ReviewStudioPage`, `/export` `ExportPage`, `/lab` `VlmLabPage`), light/dark Studio theme toggle, floating Toast notifications, and a Lead-only topbar `Live Job Pill` + slide-over Job Center Drawer (hydrating from `GET /api/jobs` + `/api/events` SSE with 1-click artifact download):
- **`Reviewer`**: Claims episodes from `QueuePage`, watches at least workspace `min_watch_coverage` (loaded dynamically from `GET /api/settings`, with a watched/unwatched ruler heatmap bar and `0.5x–3x` playback), switches between single-camera / PiP / `2×2` multi-camera synchronized views, uses NLE timeline manipulation (edge-handle drag with live frame preview, drag-to-create QC segments, `S` macro-stage split with magnetic snapping, `K` cut-clean jump-skip preview, start/end frame thumbnail comparison, live `/check-splice` feedback, keyboard shortcuts + `?` cheat sheet, unsaved-changes guard, and optional auto-advance on submit), edits only the 3 allowed layers (`qc_segments`, `success`, `macro_stages`), and submits via `submit_review` or `flag`.
- **`Lead`**: Shares the exact same NLE `StudioPlayer` + `FourTrackTimeline`, toggles `DiffOverlay` (`VLM Raw vs Human`), inspects historical VLM runs (`?run_id=...` read-only compare mode), triggers single-episode partial S3/S4 or full S1–S5 reruns, manages global queue filters (`status_counts`, searchable `facets`, `search`, `sort_by`/`sort_dir`), configures workspace settings (`PUT /api/settings`), ingests datasets safely via `IngestModal` (`GET /api/ingest/browse` + async `ingest` job), and triggers async `export` / `render_mp4` / `batch_vlm` jobs.

### 4.2 `VLM Lab` Pure In-Context Iteration SOP (`VlmLabPage.tsx`)

`VlmLabPage` has 4 tabs (`diagnosis`, `few_shots`, `taxonomy` [`labels` / `prompts` / `tasks`], `eval`) and enforces a Reviewer `403` guard (`data-testid="lab-forbidden"`):

1. **Freeze GT Benchmark**: Lead freezes verified episodes (`freeze_gt` → `gt_frozen`).
2. **Error Diagnosis (`diagnosis` tab)**: Inspect the 6-bucket KPI strip (`fp_overflag`, `fn_missed`, `policy_flip`, `boundary`, `success_flip`, `stage`), compact human-readable diffs (with raw JSON folded in `<details>`), and 1-click deep links (`/review/{ds}/{stem}?t={t_start}&seg={seg_id}&tab=diff`) straight to the exact frame and segment in `ReviewStudioPage`.
3. **Unknown Anomaly Triage (`diagnosis` tab)**: `merge` ambiguous `unknown_anomaly` segments into an existing label (generating a counter-example Few-Shot) or `promote` recurring patterns into a new visual label (`LabelRegistry.version` incremented, `.history/` snapshot saved).
4. **2×2 Visual Few-Shot Mistake Book (`few_shots` tab or inline in `diagnosis`)**: Capture 2×2 start-vs-end comparison tiles (`artifacts/few_shots/<id>.jpg`) with `vlm_wrong_json`, `human_right_json`, `rationale`, and scope (`global` / `task:<id>` / `label:<id>`). Toggle `enabled` status; S3/S4 automatically retrieve Top-K within token cap (`4000`).
5. **Versioned Taxonomy, Prompts & TaskSpecs + `.history/` Rollback (`taxonomy` tab)**: Update labels, stage prompts (`s1_grounding`..`s5_success`), or `TaskSpec`s (`labels` / `prompts` / `tasks` sub-tabs), browse `.history/` snapshots with line-by-line unified diff and 1-click atomic rollback (`LabHistoryDrawer`), and optionally request advisory AI prompt diff suggestions (`POST /api/lab/prompt-assist`).
6. **Async GT Regression Sandbox & Promote (`eval` tab & `robo-vlm-qc eval`)**: Run async sandbox evaluation (`POST /api/jobs {job_type: "eval_gt"}`) against `gt_frozen` episodes (`S1`/`S2`/`S5` cache hit 100%; `0` mutations to episode status or `active_run_id`), monitor live SSE progress and cache hits in the Job Drawer, compare baseline vs candidate (`f1`, `precision`, `recall`, `mean_iou`, `overcut_rate`, `guardrail_downgrade_rate`, `success_accuracy`, `total_tokens`, `cache_hit_rate`), and click `Promote` to apply overrides with `.history/` backup.

> **Architectural Boundary**: `VLM Lab` operates **strictly In-Context**. There is zero VLM SFT, DPO, LoRA, or PEFT fine-tuning dataset export.

---

## 5. Synthetic Demo Dataset & Docker Operations

### 5.1 Generating Local Demo Data

[`scripts/generate_demo_dataset.py`](../scripts/generate_demo_dataset.py) generates a self-contained synthetic multi-format demo pack without external downloads:

```bash
python scripts/generate_demo_dataset.py --out ./demo_data
robo-vlm-qc -w ./workspace ingest --data ./demo_data --json
```

It creates:
- `demo_pick_and_place/`: 2 multi-camera `video_dir` episodes (`ep_0001`, `ep_0002`, 3s @ 30 FPS, `main` + `left_wrist` + `right_wrist` MP4s + `episode.json`).
- `demo_lerobot_handover/`: 1 `lerobot` v3 episode (`episode_000000`, 3s @ 30 FPS, canonical 81-D kinematics Parquet + 3 camera MP4s + v3 metadata).

### 5.2 Containerized Deployment

Use [`Dockerfile`](../Dockerfile) and [`docker-compose.yml`](../docker-compose.yml) for reproducible production or team deployments:

```bash
cp .env.example .env
# Populate ROBO_VLM_QC_REVIEWER_TOKEN, ROBO_VLM_QC_LEAD_TOKEN, and VLM API keys in .env
docker compose up --build -d
```
