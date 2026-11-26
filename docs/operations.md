# `robo_vlm_qc` Operations Guide — CLI, Config, API, Studio & `VLM Lab`

> **Version**: `3.0` | **Last Synced**: `2026-11-26`  
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
  [--task auto|<task_id>] \
  [--dry-run] \
  [--json]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--data` | *(required, repeatable)* | Read-only file or directory to scan. |
| `--adapter` | `auto` | Force specific `InputAdapter` or auto-detect by priority (`h5_raw_v3` → `lerobot` → `video_dir`). |
| `--dataset` | `None` | Override inferred `dataset` identifier (`episode_id = "<dataset>/<stem>"`). |
| `--task`, `--task-id` | `None` (`auto`) | Optional explicit `task_id` override for all ingested episodes (must exist in configured `tasks/`; exits non-zero if unknown). Useful when raw dataset metadata contains generic device or serial names that would otherwise fall back to `_default`. |
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
| `--stage` | `all` | Re-run a specific stage (`s1`..`s5`, `s1_grounding`, `s2_macro`, `s3_discover`, `s4a`, `s4b`, `s4_verify`, `s5_success`). `s1`/`s1_grounding` runs standalone preserving cached `s2`–`s5`; `s2` cascades to `s3`/`s4`; `s3` cascades to `s4`. |
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
  [--state-schema-mode auto|preserve_native|canonical_81d] \
  [--split-by-subtask] \
  [--dry-run] [--json]
```

Default `--status` includes `approved` and `gt_frozen` (`EpisodeStatus` choices: `ingested`, `vlm_queued`, `vlm_running`, `vlm_ready`, `claimed`, `reviewed`, `flagged`, `approved`, `gt_frozen`). For `lerobot_v3`, `--state-schema-mode auto` preserves native dims/names for uniform LeRobot sources, skips dummy zeros for pure-video `video_dir`, and unifies H5/mixed sets to canonical 81-D; `--split-by-subtask` exports each clean subtask span as an independent episode with `task` set to `MacroStage.instruction`.

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

### 1.7 `config-vlm` — One-Line VLM API / Model Configuration & Probe

```bash
robo-vlm-qc -w ./workspace config-vlm \
  [--preset openai|dashscope|gemini|vllm|custom] \
  [--provider openai_compat|gemini] \
  [--base-url https://api.openai.com/v1] \
  [--api-key "$API_KEY"] \
  [--api-key-env OPENAI_API_KEY] \
  [--model gpt-4o] \
  [--thinking-mode auto|dashscope|disabled] \
  [--test] \
  [--json]
```

Updates `<workspace>/configs_override/default.yaml` (with `.history/` backup) while saving `--api-key` exclusively to `<workspace>/.env` (`0600` permissions, never written to YAML or echoed in plaintext). Pass `--test` to run a live connectivity & latency probe against the configured endpoint.

---

## 2. Configuration, Provider Presets & 11 Pure-Visual Labels

### 2.1 Layered Overrides & `.history/` Snapshots

[`ConfigManager`](../robo_vlm_qc/config.py) resolves the base config directory in order (`--config-dir` → `ROBO_VLM_QC_CONFIG_DIR` → repo `configs/` → wheel-bundled `robo_vlm_qc._bundled_configs` → `cwd/configs`) and overlays `<workspace>/configs_override/`:
- `default.yaml` and `tasks/<task_id>.yaml`: deep-merged over base YAML.
- `labels.yaml` and `prompts/<stage>.yaml`: replaced when overridden.
- All programmatic override writes use `archive_and_write_override` / `ArtifactStore.write_config_override`, saving a timestamped JSON snapshot of the prior state under `<workspace>/.history/` before atomic replacement.

### 2.2 Key Settings in [`configs/default.yaml`](../configs/default.yaml) & Provider Presets

- **Default VLM Client & Resilience**: `provider: openai_compat`, `base_url: https://api.openai.com/v1`, `api_key_env: OPENAI_API_KEY`, `key_file: .env`, `thinking_mode: auto`, `concurrency: 32`, `episode_concurrency: 4`, `decode_concurrency: 6`, `gemini_media_resolution: "MEDIA_RESOLUTION_HIGH"`, `vision_min_tokens_per_image: 550`, `vision_blind_node_retries: 5`, `vision_blind_node_guard: "auto"`, `vision_blind_node_on_exhaust: "raise"`, `fallback_model: null`, `fallback_provider: null`, `circuit_breaker: {enabled: true, failure_threshold: 3, trip_on_quota_403: true}`, and `pricing` (`default`, `gpt-4o`, `gpt-4o-mini`, `gemini-2.5-pro`, `gemini-2.5-flash`, `qwen-vl-max`).
- **Stage Model Routing (`stage_models`)**: `s1_grounding`, `s2_macro`, `s3_discover`, `s4_verify`, `s5_success` (default `gpt-4o`, `thinking: false`; customizable per stage via YAML, `config-vlm`, or Studio `Model / API` modal).
- **Provider Presets ([`configs/presets/`](../configs/presets/))**: Apply via `config-vlm --preset <name>`, the Studio `Model / API` modal, or copy to `<workspace>/configs_override/default.yaml`:
  - [`openai.yaml`](../configs/presets/openai.yaml) (`gpt-4o`, `OPENAI_API_KEY`, `thinking_mode: disabled`)
  - [`dashscope.yaml`](../configs/presets/dashscope.yaml) (`qwen-vl-max`, `DASHSCOPE_API_KEY`, `thinking_mode: dashscope`, `thinking: true`)
  - [`gemini.yaml`](../configs/presets/gemini.yaml) (`provider: gemini`, `gemini-2.5-pro`, `GEMINI_API_KEY`, `thinking: true`)
  - [`vllm.yaml`](../configs/presets/vllm.yaml) (`http://127.0.0.1:8001/v1`, `Qwen/Qwen2.5-VL-72B-Instruct`, `VLLM_API_KEY`, `thinking_mode: disabled`)
- **Example TaskSpecs ([`configs/tasks/`](../configs/tasks/))**: [`_default.yaml`](../configs/tasks/_default.yaml) (embodiment-agnostic: robot arms/grippers & human hands), [`pick_and_place.yaml`](../configs/tasks/pick_and_place.yaml), [`bimanual_handover.yaml`](../configs/tasks/bimanual_handover.yaml).
- **Dynamic Budget**: `tokens_per_second: 5000`, `min_episode_tokens: 200000`, `max_tokens_per_episode: null`, `unlimited: false`.
- **Stage 1 Grounding & Anti-Jitter BAFT Tracking (`s1_grounding`)**: Temporal batching & spatial anchoring (`batch_size: 4`, `batch_concurrency: 3`, `reference_anchor_enabled: true`, `interval_s: 2.5`, `max_keyframes: 32`, `hard_max_keyframes: 72`, `ego_height: 540`, `max_tokens: 2400`, `static_container_min_frames: 5`, `static_jitter_tol_px_1280: 85.0`) paired with local BAFT frame-level 2D tracking (`tracking.enabled: true`, `tracking.track_fps: 15.0`, `tracking.short_native_fps_max_s: 15.0`, `tracking.track_width: 480`, `tracking.flow_preset: "ultrafast"`, `tracking.ncc_stride: 2`, `tracking.ncc_min_score: 0.55`, `tracking.fb_max_err_px: 2.5`, `tracking.occlusion_conf_thresh: 0.25`, `tracking.occlusion_min_frames: 4`, `tracking.smooth_window: 5`, `tracking.camera_motion_threshold_norm: 25.0`, `tracking.outlier_jump_norm: 250.0`, `tracking.moderate_outlier_jump_norm: 75.0`, `tracking.anchor_scale_jitter_ratio: 0.22`, `tracking.anchor_min_blend_weight: 0.15`, `tracking.anchor_max_blend_weight: 0.80`, `tracking.adaptive_reanchor_max_calls: 0`, `tracking.adaptive_disagree_ratio: 0.65`).
- **Sampling, Short-Circuits & Few-Shot**: `s3_discover` (`window_s: 16.0`, `window_stride_s: 12.0`, `window_step_s: 0.8`, `cell_w: 420`, `cell_h: 236`, `quality: 76`, `few_shot_top_k: 4`, `few_shot_max_tokens: 4000`), `s4_verify` (`refine_margin_s: 1.5`, `refine_step_s: 0.5`, `boundary_idle_kin_skip_s4b: true`), `s5_success` (`tail_s: 12.0`, `tail_ratio: 0.22`, `max_tail_s: 25.0`, `body_frames: 6`, `tail_frames: 10`, `control_ratio: 0.30`, `control_frames: 8`, `cell_w: 420`, `cell_h: 236`, `quality: 78`, `start_end_compare: true`, `compare_panel_w: 640`, `compare_panel_h: 360`).
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
- **RBAC**: `Reviewer` receives `403` on `/api/lab/*`, `/api/exports/*`, `/api/ingest/*`, `PUT /api/settings`, `/api/settings/vlm*`, `POST /api/episodes/batch-task`, `GET /api/episodes/{ep_id}/runs`, `GET /api/episodes/{ep_id}?run_id=...`, `/api/jobs` mutations/listing/artifacts (`POST /api/jobs`, `GET /api/jobs`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/artifact`), and review actions `approve` / `freeze_gt`. On `PATCH /api/episodes/{ep_id}/task`, `Reviewer` may only reassign non-frozen `_default` episodes to an existing non-`_default` task (`403` otherwise).

### 3.1 Core Studio, Jobs, Settings & Ingest Endpoints

| Method & Route | Role | Purpose & Key Contract Details |
| :--- | :--- | :--- |
| `GET /api/health` | Public | Returns `{"status": "ok"}`. |
| `GET /api/auth/me` | Auth | Returns `{"role": "reviewer"\|"lead", "user": "...", "no_auth": bool}` (lets Web Studio detect `--no-auth` loopback mode automatically). |
| `GET /api/labels` | Auth | Returns active `LabelRegistry` JSON. |
| `GET /api/settings` | Auth | Returns workspace Studio settings: `{"min_watch_coverage": 0.95, "batch_vlm_concurrency": 4}`. |
| `PUT /api/settings` | Lead | Body (`extra="forbid"`, at least one field): `{"min_watch_coverage"?: float [0.0..1.0], "batch_vlm_concurrency"?: int [1..32]}`. Atomically merges into `configs_override/default.yaml` (preserving unrelated keys, archiving prior state to `.history/`), hot-reloads `ConfigManager`, and publishes SSE `settings_updated`. |
| `GET /api/settings/vlm`, `PUT /api/settings/vlm`, `POST /api/settings/vlm/test` | Lead | Inspect/update workspace VLM provider, preset (`openai`\|`dashscope`\|`gemini`\|`vllm`\|`custom`), `base_url`, `api_key_env`, unified or per-stage `stage_models` (`s1_grounding`..`s5_success`), `thinking_mode`, and optional `api_key` (written strictly to `<workspace>/.env` with `0600` permissions, masked on read, never persisted in YAML). `POST /api/settings/vlm/test` executes a live connectivity & latency probe. |
| `GET /api/episodes` | Auth | Query params: `status` (single or comma-separated `EpisodeStatus`, e.g. `reviewed,flagged`), `dataset`, `task_id`, `adapter_type`, `passed`, `guardrail_downgraded`, `has_unknown`, `claimed_by`, `search` (escaped SQL `LIKE` substring on `episode_id`), `sort_by` (`episode_id`\|`updated_at`\|`duration_s`\|`cut_ratio`\|`mask_ratio`\|`badcase_count`, NULLs last), `sort_dir` (`asc`\|`desc`), `limit` (`1..10000`), `offset`. Returns `{items, total, my_claimed_count, status_counts, facets}` where `status_counts` computes SQL `GROUP BY status` across all 9 statuses unaffected by `status`/`claimed_by`, and `facets` returns workspace-global `{datasets, task_ids, adapter_types}`. |
| `GET /api/episodes/tasks` | Auth | Returns configured workspace tasks for Queue, Ingest, and Review Studio dropdowns: `{tasks: [{task_id, title, description, goal, match, aliases, grounding_objects_count}], total}` (non-`_default` alphabetical, `_default` last). |
| `PATCH /api/episodes/{ep_id}/task` | Auth (RBAC) | Body (`extra="forbid"`): `{task_id, allow_frozen?: false}`. Validates `task_id` exists (`422`), rejects active `vlm_queued`/`vlm_running` episodes (`409`), enforces `gt_frozen` `allow_frozen=true` confirmation for Lead (`409`), and restricts Reviewer to non-frozen `_default` → non-`_default` reassignment (`403`). Updates only `episodes.task_id` & `updated_at` (preserving `active_run_id`/`active_rev_id`/`status`), publishes SSE `episode_updated`, and returns `{episode_id, previous_task_id, task_id, status, has_active_run, re_run_recommended}`. |
| `POST /api/episodes/batch-task` | Lead | Body (`extra="forbid"`): `{episode_ids, task_id, allow_frozen?: false}`. Atomically validates and reassigns multiple episodes (`404`/`409` with zero partial writes on missing, active VLM, or unconfirmed `gt_frozen`), emits `episode_updated` SSE, and returns `{task_id, updated, episode_ids, has_active_runs, re_run_recommended_ids}`. |
| `POST /api/episodes/{ep_id}/infer-subtasks` | Auth | Re-infers aligned `MacroStage` subtask spans & VLA `instruction` strings directly from the episode's human-verified QC segments without mutating saved DB state (returns draft stages for Undo/Redo-able Studio review). |
| `POST /api/episodes/{ep_id}/check-splice` | Auth | Body (`extra="forbid"`): `{t_start, t_end, label_id, splice_policy, affected_arm}`. Uses thread-safe in-memory LRU `KinematicsTrack` cache (`<20ms` warm recalc) to run `hallucination` + `splice_gate` (or `visual_gate` SSIM/L1 fallback when `has_kinematics=false`), returning `{passed, downgraded, downgrade_reason, effective_splice_policy, opt_t_start, opt_t_end, metrics, compute_ms, mode}` with zero DB mutation. |
| `GET /api/episodes/{ep_id}/runs` | Lead | Lists all historical `VlmRunArtifact` metadata for an episode (newest first): `{episode_id, active_run_id, runs: [{run_id, created_at, prompt_version, label_version, few_shot_pack_id, tokens_total, is_eval_sandbox}]}`. |
| `GET /api/episodes/{ep_id}` | Auth | Returns `ResolvedEpisode` + `claimed_by`, `claim_expires_at`, `guardrail_downgraded`, `selected_run_id`, `is_historical_run_view`. Optional `?run_id=...` (Lead-only; `403` for Reviewer, `404` on missing/cross-episode run) loads a read-only historical run snapshot (`active_rev_id=null`, `is_human_reviewed=false`, `is_historical_run_view=true`) without mutating DB state. |
| `GET /api/episodes/{ep_id}/frame` | Auth | Query: `camera?`, required float `t_s` (returns `image/jpeg`). |
| `GET /api/episodes/{ep_id}/preview` | Auth | Query: `camera?`, `hd: bool` (streams transcoded CFR H.264 MP4). |
| `POST /api/episodes/{ep_id}/claim`, `/renew`, `/release` | Auth | Body: `{"user": "<username>"}` (or uses `X-User`). `409` on active claim conflict or `gt_frozen`. |
| `GET /api/episodes/{ep_id}/reviews` | Auth | Lists immutable `HumanReviewArtifact` history (`items`, `total`). |
| `POST /api/episodes/{ep_id}/reviews` | Auth | Body (`ReviewWritePayload`, `extra="forbid"`): `{base_run_id, action, qc_segments, success, macro_stages, watch_coverage, notes?, flagged_reason?}`. Rejects `author` and read-only keys (`grounding`, `arm_timelines`, `raw_vlm_segments`) with `422`; returns `409` if episode is still `ingested`/`vlm_queued`/`vlm_running` or `X-User` does not match claimer; dynamically enforces workspace `min_watch_coverage` (`422`); `Reviewer` `approve`/`freeze_gt` → `403`. |
| `POST /api/jobs` | Lead | Unified async job engine (`201 Created`). Body (`extra="forbid"`): `{job_type: "batch_vlm"\|"render_mp4"\|"export"\|"eval_gt"\|"ingest", episode_ids?, stage?, fresh?, concurrency?, is_eval_sandbox?, custom_objects?, export_params?, eval_params?, ingest_params?}`. `batch_vlm` deduplicates `episode_ids`, rejects already-active (`vlm_queued`/`vlm_running` or reserved) episodes with `409 Conflict`, atomically queues eligible episodes (`episode_updated` SSE), executes concurrently (`ThreadPoolExecutor`, default `batch_vlm_concurrency=4`), supports `custom_objects` for S1 reruns, and supports stage modes `s1_grounding` (S1-only rerun reusing `S2`–`S5` cache), `s3_s4`/`S3/S4` (reusing `S1`/`S2`/`S5` cache), `s1_s5`/`S1-S5` (full fresh rerun), `all` (respecting `fresh`), and `s1..s5`. `ingest_params` accepts `{path\|data_path\|data_paths, adapter?, dataset?, task_id?}` (`task_id` overrides auto-resolution; unknown → `422`). Publishes incremental SSE `job_update` events and registers downloadable `summary.artifacts`. |
| `GET /api/jobs`, `POST /api/jobs/{job_id}/cancel` | Lead | List recent jobs (`status?`, `job_type?`, `limit`, `offset`, exact `total` via `count_jobs`, and workspace-wide `status_counts`) or cooperatively cancel an active `queued`/`running` job: sets `cancel_event`, invokes `VlmClient.cancel()` (~1 ms active socket abort), blocks post-cancel DB commits (`before_commit_check`), and atomically rolls back transient `vlm_queued`/`vlm_running` episodes via `restore_vlm_episodes_on_cancel` with `episode_updated` SSE broadcasts. On server startup (`robo-vlm-qc serve`), any crash-orphaned `queued`/`running` jobs are reconciled to `failed` (`interrupted_by_server_restart`) and stuck VLM episodes are rolled back cleanly. |
| `DELETE /api/jobs/{job_id}`, `DELETE /api/jobs` | Lead | Permanently remove terminal (`completed`\|`failed`\|`cancelled`) job records from SQLite without deleting disk artifacts (VLM runs, reviews, exports, or eval reports). Active `queued`/`running` jobs are rejected with `409 Conflict` (must be cancelled first). Batch `DELETE /api/jobs` accepts `scope=finished\|failed_cancelled` and optional `job_type`, deletes all matching terminal records workspace-wide, returns `{deleted, job_ids, scope, status_counts}`, and broadcasts `jobs_deleted` SSE to keep all open Lead tabs in sync. |
| `GET /api/jobs/{job_id}/artifact` | Lead | Query: `index=0` (supports `X-Auth-Token` or `?token=`). Enforces strict workspace path-containment (`403` on escape) and streams rendered MP4 (`video/mp4`) or export bundle (`application/zip`). |
| `GET /api/jobs/{job_id}`, `/api/jobs/{job_id}/events`, `GET /api/events` | Auth | Job polling and native `StreamingResponse` `text/event-stream` SSE streams (`job_update`, `episode_updated`, `settings_updated`, `config_updated`, `episode_claimed`, `episode_released`, `review_saved`, `export_completed`). |
| `GET /api/ingest/browse` | Lead | Query: `path?`. Read-only directory browser constrained by `--allowed-data-roots` (`403` on escape/symlink traversal, `404` missing, `400` non-dir). Returns `{current_path, parent_path, allowed_roots, entries, detected_adapters}` with bounded 3-adapter episode count preflight (`h5_raw_v3`, `lerobot`, `video_dir`). |
| `POST /api/exports/dry-run`, `POST /api/exports` | Lead | Synchronous dry-run preview & transactional export (Web `ExportPage` submits async `POST /api/jobs {job_type: "export"}`). Body: `{format, recipe, chunk_size?, min_piece_s?, tail_frames?, smooth_tail?, include_statuses?, context_window_s?, state_schema_mode?, split_by_subtask?, dataset_filter?, task_filter?, episode_ids?}`. |
| `GET /api/exports`, `GET /api/exports/{id}`, `GET /api/exports/{id}/download` | Lead | Lists manifests, inspects manifest, or streams ZIP bundle with strict workspace path-containment and symlink-escape protection (`403` on escape). |

### 3.2 Lead-Only `VLM Lab` Endpoints (`/api/lab/*`)

| Method & Route | Purpose & Key Contract Details |
| :--- | :--- |
| `GET /api/lab/labels`, `PUT /api/lab/labels` | Read or replace `LabelRegistry` (rejects duplicate IDs with `422`; snapshots prior file to `.history/`). |
| `GET /api/lab/prompts`, `PUT /api/lab/prompts` | Read or update stage `PromptTemplate`s (`s1_grounding`..`s5_success`) with `.history/` backup. |
| `GET /api/lab/tasks`, `PUT /api/lab/tasks` | Read or update `TaskSpec` YAML overrides (`tasks/<task_id>.yaml`) with `.history/` backup. |
| `POST /api/lab/tasks/draft` | Body (`extra="forbid"`): `{task_id, description, title?, sample_episode_id?, num_frames=5 [4..6]}`. Advisory multimodal AI `TaskSpec` drafter (never writes `configs_override/`); rejects `_default` (`422`), optionally samples 4–6 uniform RGB keyframes from `sample_episode_id` (`404` if missing), accepts Chinese or English descriptions and returns a normalized English `TaskSpec` aligned with S1–S5 plus deterministically scrubbed `match`/`aliases` (filtering out serial numbers, dates, and generic capture device names) and `{advisory_only: true, task_spec, used_sample_episode_id, sampled_frames_count, rationale, warnings}`. |
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

Single React 19 SPA with 4 pages (`/` `QueuePage`, `/review/:dataset/:stem` `ReviewStudioPage`, `/export` `ExportPage`, `/lab` `VlmLabPage`), live **English / Chinese (`EN / ZH`)** toggle (`i18nStore`), Light/Dark theme toggle, floating Toast notifications, Lead-only TopNav **`Model / API`** switcher (`VlmSettingsModal`), and `Live Job Pill` + slide-over `JobCenterDrawer` (hydrating from `GET /api/jobs` + `/api/events` SSE with 1-click artifact download):
- **`Reviewer`**: Claims episodes from `QueuePage`, watches at least workspace `min_watch_coverage` (loaded dynamically from `GET /api/settings`, with a watched/unwatched ruler heatmap bar and `0.5x–3x` playback), switches between single-camera / PiP / `2×2` multi-camera synchronized views, uses NLE timeline manipulation (edge-handle drag with live frame preview, drag-to-create QC segments, `S` subtask split with `±0.08s` cross-track QC magnetic snapping, 1-click **Infer Subtasks** from verified QC segments, `K` cut-clean jump-skip preview, start/end frame thumbnail comparison, live `/check-splice` feedback, redesigned Layer 2 `SuccessPanel`, keyboard shortcuts + `?` cheat sheet, unsaved-changes guard, and optional auto-advance on submit), can reassign non-frozen unrecognized (`_default`) episodes to an existing task directly from the Review Studio header, edits only the 3 allowed layers (`qc_segments`, `success`, `macro_stages`), and submits via `submit_review` or `flag`.
- **`Lead`**: Shares the exact same NLE `StudioPlayer` (with 30fps `requestVideoFrameCallback` + `object-fit: contain` v2 dense 2D grounding canvas, per-object visibility toggles, and custom-object S1 reruns) + `FourTrackTimeline`, toggles `DiffOverlay` (`VLM Raw vs Human`), inspects historical VLM runs (`?run_id=...` read-only compare mode), triggers single-episode `s1_grounding` (S1-only), partial `S3/S4`, or full `S1–S5` reruns (with inline mid-flight cancel + SSE status sync), filters/highlights unrecognized `_default` episodes in `QueuePage` with single/batch **Assign Task** + reusable **"+ New Task (AI)" Wizard** (`TaskDraftWizardModal` auto-attaching a selected episode for 4–6 frame multimodal `TaskSpec` drafting, `gt_frozen` confirmation, and 1-click "Re-run VLM now" prompt), configures workspace settings (`PUT /api/settings`) and VLM endpoints (`VlmSettingsModal`), ingests datasets safely via `IngestModal` (`GET /api/ingest/browse`, optional `task_id` override or inline "+ New Task (AI)", and async `ingest` job), and triggers async `export` (with `state_schema_mode` & `split_by_subtask`), `render_mp4`, and `batch_vlm` jobs.

### 4.2 `VLM Lab` Pure In-Context Iteration SOP (`VlmLabPage.tsx`)

`VlmLabPage` has 4 tabs (`diagnosis`, `few_shots`, `taxonomy` [`labels` / `prompts` / `tasks`], `eval`) and enforces a Reviewer `403` guard (`data-testid="lab-forbidden"`):

1. **Freeze GT Benchmark**: Lead freezes verified episodes (`freeze_gt` → `gt_frozen`).
2. **Error Diagnosis (`diagnosis` tab)**: Inspect the 6-bucket KPI strip (`fp_overflag`, `fn_missed`, `policy_flip`, `boundary`, `success_flip`, `stage`), compact human-readable diffs (with raw JSON folded in `<details>`), and 1-click deep links (`/review/{ds}/{stem}?t={t_start}&seg={seg_id}&tab=diff`) straight to the exact frame and segment in `ReviewStudioPage`.
3. **Unknown Anomaly Triage (`diagnosis` tab)**: `merge` ambiguous `unknown_anomaly` segments into an existing label (generating a counter-example Few-Shot) or `promote` recurring patterns into a new visual label (`LabelRegistry.version` incremented, `.history/` snapshot saved).
4. **2×2 Visual Few-Shot Mistake Book (`few_shots` tab or inline in `diagnosis`)**: Capture 2×2 start-vs-end comparison tiles (`artifacts/few_shots/<id>.jpg`) with `vlm_wrong_json`, `human_right_json`, `rationale`, and scope (`global` / `task:<id>` / `label:<id>`). Toggle `enabled` status; S3/S4 automatically retrieve Top-K within token cap (`4000`).
5. **Versioned Taxonomy, Prompts, TaskSpecs & AI Task Onboarding + `.history/` Rollback (`taxonomy` tab)**: Update labels, stage prompts (`s1_grounding`..`s5_success`), or `TaskSpec`s (`labels` / `prompts` / `tasks` sub-tabs), onboard new tasks via **"+ New Task (AI Assist)"** (`POST /api/lab/tasks/draft` with optional sample episode keyframes → human edit → `PUT /api/lab/tasks`), browse `.history/` snapshots with line-by-line unified diff and 1-click atomic rollback (`LabHistoryDrawer`), and optionally request advisory AI prompt diff suggestions (`POST /api/lab/prompt-assist`).
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
