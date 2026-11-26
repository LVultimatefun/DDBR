# `robo_vlm_qc` System Architecture & Contracts

> **Version**: `2.0` | **Last Synced**: `2026-11-26`  
> Authoritative architecture reference for [`robo_vlm_qc`](../README.md). See [`docs/operations.md`](operations.md) for CLI, REST/SSE API, Studio UI, and `VLM Lab` operations.

---

## 1. Eight Locked Architectural Decisions (D1–D8)

| # | Dimension | Locked Decision | Problem Solved |
| :-: | :--- | :--- | :--- |
| **D1** | **Standalone Packaging & Read-Only Sources** | Standalone Apache-2.0 Python + React platform bundling default YAML configs (`robo_vlm_qc._bundled_configs`) while keeping all `--data` sources strictly read-only. | Guarantees safe wheel distribution and zero accidental mutation of raw robotics datasets. |
| **D2** | **Role of Kinematics** | Pure-visual VLM primary labeling (augmented with optional objective 30 Hz kinematics summaries in S3/S4B when available) + pluggable light post-processing guardrails (`splice_gate` + `hallucination` when `KinematicsTrack != None`; `visual_gate` when `None`). Zero 30 Hz kinematic rule detectors. | Works across embodiments (bimanual 6/7-DoF arms, torso, chassis, or egocentric human-hand video) while preventing cut-induced teleportation. |
| **D3** | **Storage Foundation** | SQLite WAL (`robo_vlm_qc.db`) + immutable JSON artifacts (`artifacts/`) + `.history/` config snapshots. | Replaces fragmented multi-file annotation directories with one atomic source of truth. |
| **D4** | **`VLM Lab` Loop** | Strictly In-Context improvement loop (6-bucket diagnosis, 2×2 visual/text Few-Shot book, versioned taxonomy/prompts/tasks, multimodal AI `TaskSpec` onboarding drafter, GT regression sandbox). **Zero VLM SFT/DPO/LoRA export.** | Drives rapid visual QC accuracy gains and cold-start new-task onboarding through prompt, taxonomy, and Few-Shot iteration without model fine-tuning pipelines. |
| **D5** | **Dual-Role Bilingual Frontend** | Single React 19 + TypeScript + Vite SPA serving `Reviewer` and `Lead` views over identical player & timeline components with live `EN / ZH` switching, Dark/Light themes, one-click `Model / API` switching, and `_default` task intervention. | Eliminates UI drift between reviewer and lead workflows while lowering onboarding friction. |
| **D6** | **3-Layer Human Edits** | Humans edit ONLY `qc_segments`, `success`, and `macro_stages` (subtasks + VLA `instruction`). `grounding`, `arm_timelines`, and `raw_vlm_segments` are strictly read-only. | Keeps annotation fast while preserving grounding & dual-arm visual aids and Lead diffing. |
| **D7** | **3×4 I/O Matrix** | Pluggable `InputAdapter` (`h5_raw_v3`, `lerobot`, `video_dir`) × `OutputExporter` (`lerobot_v3` w/ adaptive `state_schema_mode` & optional `split_by_subtask`, `h5_clean`, `zero_copy_index`, `viz_mp4`) across 3 recipes (`bc_clean`, `recovery_plus`, `badcase_pairs`). | Decouples format serialization from core VLM QC, human review, and trajectory slicing logic. |
| **D8** | **Causal VLM DAG & Cache** | `S0 → S1 → (S2 ∥ S5) → S3 → S4A → S4B → Guardrail → Store`, with provider-isolated SHA256 `StageCache`, stage-level `stage_models` routing, S4A/S4B/S5 cost short-circuits, and cooperative mid-flight cancellation. | Prevents Stage 3 over-cutting via S2 backbone injection, cuts redundant VLM calls, and reuses S1/S2/S5 cache at 100% during S3/label/Few-Shot iteration. |

---

## 2. System Topology, Causal VLM DAG & State Machine

### 2.1 End-to-End 3-In × DAG × Store × Studio × Lab × 4-Out Pipeline

<p align="center">
  <img src="assets/architecture_overview.svg" alt="DDBR / robo_vlm_qc System Architecture Overview" width="100%" />
</p>

<details>
<summary>View editable Mermaid source diagram</summary>

```mermaid
flowchart TB
    classDef source fill:#0c4a6e,stroke:#38bdf8,stroke-width:1.5px,color:#e0f2fe
    classDef io fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#bae6fd
    classDef vlm fill:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,color:#e0e7ff
    classDef store fill:#064e3b,stroke:#34d399,stroke-width:1.5px,color:#d1fae5
    classDef studio fill:#3b0764,stroke:#c084fc,stroke-width:1.5px,color:#f3e8ff
    classDef exp fill:#78350f,stroke:#fbbf24,stroke-width:1.5px,color:#fef3c7

    subgraph RO["Strict Read-Only Sources"]
        I1["h5_raw_v3 (.h5)"]:::source
        I2["lerobot (v2/v3 Parquet + MP4)"]:::source
        I3["video_dir (Multi-Cam MP4, kinematics=None)"]:::source
    end

    subgraph IO["3-In Pluggable I/O Kernel (robo_vlm_qc/io/)"]
        REG_IN["InputAdapter Registry"]:::io
        VIEW["EpisodeView (EpisodeMeta + MediaSource + Optional KinematicsTrack)"]:::io
        TILER["Adaptive 2x2 / 1x2 Tile Composer"]:::io
    end

    subgraph VLM["Causal VLM Engine & Guardrails (vlm/ + guardrail/)"]
        CACHE["StageCache (stage_cache/prefix/sha256.json)"]:::vlm
        DAG["VlmDag: S0 -> S1 -> (S2 || S5) -> S3 -> S4A -> S4B"]:::vlm
        GUARD["Light Guardrails (splice_gate + hallucination | visual_gate)"]:::vlm
    end

    subgraph STORE["Single-Truth Store (store/)"]
        DB["SQLite WAL (robo_vlm_qc.db: 6 tables + claim leases)"]:::store
        ART["Immutable JSON Artifacts (vlm_runs / reviews / few_shots / previews)"]:::store
        HIST[".history/ Config Snapshots + configs_override/"]:::store
    end

    subgraph STUDIO["Unified Dual-Role Studio & VLM Lab (api/ + web/ + lab/)"]
        REV["Reviewer View (Claim, Watch >=95%, Edit 3 Layers, Submit/Flag)"]:::studio
        LEAD["Lead View (DiffOverlay, Arbitrate, Approve, Freeze GT)"]:::studio
        LAB["VLM Lab In-Context Loop (Diagnosis, 2x2 Few-Shot, Unknown Triage, GT Eval)"]:::studio
    end

    subgraph EXPORT["4-Out Transactional Exporters (export/)"]
        SLICER["PieceSlicer (3 recipes, Smoothstep+SO(3) tail, 81-D, Loss Mask, Subtasks)"]:::exp
        O1["lerobot_v3 (_staging -> rename)"]:::exp
        O2["h5_clean (_staging -> rename)"]:::exp
        O3["zero_copy_index (_staging -> rename)"]:::exp
        O4["viz_mp4 (1920x1080@30fps Studio HUD)"]:::exp
    end

    I1 & I2 & I3 --> REG_IN --> VIEW --> TILER
    TILER --> DAG
    VIEW --> DAG
    DAG <--> CACHE
    DAG --> GUARD --> DB & ART
    DB & ART <--> REV & LEAD
    REV --> LEAD --> LAB
    LAB -- "Few-Shots / Labels / Prompts / Tasks" --> HIST
    LAB -- "In-Context Injection (S3/S4)" --> DAG
    DB & ART --> SLICER
    VIEW --> SLICER
    SLICER --> O1 & O2 & O3 & O4
```

</details>

### 2.2 Causal VLM DAG Execution Topology & Guardrail Funnel

<p align="center">
  <img src="assets/vlm_dag_guardrail.svg" alt="Causal VLM DAG StageCache Isolation and Three-Policy Guardrail Funnel" width="100%" />
</p>

<details>
<summary>View editable Mermaid source diagram</summary>

```mermaid
flowchart LR
    classDef cached fill:#064e3b,stroke:#34d399,stroke-width:1.5px,color:#d1fae5
    classDef iter fill:#1e1b4b,stroke:#818cf8,stroke-width:1.5px,color:#e0e7ff
    classDef guard fill:#78350f,stroke:#fbbf24,stroke-width:1.5px,color:#fef3c7

    S0["S0: Open EpisodeView & Adaptive Layout"]:::cached --> S1["S1: Temporal-Batched 2D Grounding + BAFT Dense Tracking"]:::cached
    S0 --> S2["S2: Macro Backbone (macro_stages + L/R arm_timelines)"]:::cached
    S0 --> S5["S5: Dense-Tail Success + 30% Anti-FP Control Check"]:::cached
    S2 --> S3["S3: Sliding-Window QC Discovery (S2 Protected + Precision-First + Few-Shot)"]:::iter
    S3 --> S4A["S4A: Global Deduplication & Merge"]:::iter
    S4A --> S4B["S4B: 0.5s Start-vs-End CoV Diff Refinement"]:::iter
    S4B --> GRD["Guardrails: hallucination + splice_gate | visual_gate"]:::guard
    S1 & S2 & S5 & GRD --> OUT["Persist VlmRunArtifact + SQLite Index"]:::cached
```

</details>

- **Stage 1 Open-Vocabulary Discovery, Reference-Frame Anchoring & Anti-Jitter BAFT (`vlm/stage1_grounding.py`, `vlm/tracker.py`)**:
  - **Pass-0 Canonical Discovery & Custom Objects**: When `TaskSpec.grounding_objects` is empty (e.g., `_default`), Stage 1 runs a Pass-0 discovery call over 4–6 overview frames to lock 1–4 canonical `obj_id`s (prioritizing manipulator/hand-interacted objects and target containers) before launching concurrent grounding batches, preventing cross-batch label drift. Review Studio also supports explicit `custom_objects` S1 reruns.
  - **Direct Fixed-vs-Moving Camera Detection**: `detect_camera_motion_direct` evaluates 7 frames via ORB+RANSAC affine estimation (with `phaseCorrelate` fallback and strict `_warp_overlap_agreement` rigid-warp verification) before S1 VLM calls, preventing articulated robot arms or human hands from false-triggering camera motion and skipping redundant BAFT flow on static containers for fixed cameras.
  - **Batch-0 `ref_frame_0` Anchoring & Hallucination Protection**: Phase 1 runs Batch 0 (`batch_size=4`, `batch_concurrency=3`, `reference_anchor_enabled=true`) first, guards against blind-node phantom teleports/blank static boxes (`_is_batch0_phantom_hallucination`), injects annotated `ref_frame_0` visual+text anchor hints into Batch `1..N`, and applies cross-batch static container consensus clustering (`_select_static_consensus_box`).
  - **Anti-Jitter Offline BAFT Tracker**: Runs locally with **0 extra VLM calls by default** (`adaptive_reanchor_max_calls: 0`) at `track_width=480` (with slender/small-target margin & silhouette collar tuning). Combines DIS optical flow + multi-scale NCC, hallucinated anchor rejection (`reject_hallucinated_anchors`), flow-consistent adaptive Savitzky-Golay anchor weighting (`anchor_min/max_blend_weight`), moderate & 2-frame jitter rejection, Single-Switchpoint Monotonic DP on disjoint bidirectional intervals (eliminating 15 Hz ping-pong), and occlusion hysteresis (`flag=0`).
- **Stage 1 Cache & Rerun Decoupling**: `S1` cache keys (`s1_key`) are fully decoupled from `S2`, `S3`, `S4`, and `S5` (`s2_key` and `s5_key` omit `s1_key` as `upstream_hash`; `s3_key` and `s4_key` depend only on `S2`/`S3` outputs). Running `fresh_stages={"s1_grounding"}` executes only `S1` while reusing `S2`–`S5` `StageCache` at 100%.
- **Stages 2–5 Annotation & Efficiency Upgrades (`vlm/stage2_macro.py`–`stage5_success.py`, `vlm/prompt_builder.py`)**:
  - **S2 Subtask Backbone & On-Demand Re-Inference**: Injects S1 object-state transitions (`format_grounding_summary`) into S2 to produce `MacroStage` spans with VLA training `instruction` strings; also provides `infer_subtasks_from_segments` (`POST /api/episodes/{ep_id}/infer-subtasks`) to re-align subtasks directly against human-verified QC segments.
  - **S3 / S4B Kinematics Telemetry & Context Injection**: When `KinematicsTrack` is present, injects objective 30 Hz English 2s-bucket kinematics summaries (`format_window_kinematics_summary`: path, net displacement, backtrack ratio, reversals, low-speed %) alongside `task_goal`, `task_prompt`, and `labels_desc` (gracefully omitted on pure-video `video_dir`).
  - **S4A Short-Circuit & S4B Kinematics Edge-Idle Skip**: `_needs_s4a_vlm_merge` skips S4A VLM calls when `<=1` candidate exists or all candidates are separated by `>2.0s`. When `vlm.s4_verify.boundary_idle_kin_skip_s4b: true` and kinematics confirm stillness on an edge `boundary_idle`, S4B snaps directly to `0.0` / `round(duration_s, 3)` and skips the VLM call.
  - **S5 Sequential On-Demand Control & `START_vs_END` Comparison**: Primary `s5_success` uses widened `DEFAULT_MAIN_ZOOM`, adaptive tail sampling (`tail_s=12.0`, `tail_ratio=0.22`, `max_tail_s=25.0`), and high-resolution side-by-side `START_vs_END` / `START_vs_CUTOFF` comparison tiles (`compose_start_end_compare`). The 30% anti-FP `s5_control` check runs sequentially **only when primary `verdict == "success"`** (cutting S5 calls in half on failure/unsure episodes), and `_derive_verdict` preserves explicit `failure` verdicts even when individual criteria contain `met=None`.

### 2.3 Episode Status State Machine (`EpisodeStatus`)

```mermaid
flowchart TB
    classDef neutral fill:#1e293b,stroke:#64748b,stroke-width:1.5px,color:#f1f5f9
    classDef active fill:#0c4a6e,stroke:#38bdf8,stroke-width:1.5px,color:#e0f2fe
    classDef warn fill:#78350f,stroke:#f59e0b,stroke-width:1.5px,color:#fef3c7
    classDef good fill:#064e3b,stroke:#10b981,stroke-width:1.5px,color:#d1fae5
    classDef frozen fill:#3b0764,stroke:#c084fc,stroke-width:1.5px,color:#f3e8ff

    ING["ingested (Scanned from read-only source)"]:::neutral
    QUE["vlm_queued (Batch job queued)"]:::neutral
    RUN["vlm_running (DAG executing)"]:::active
    RDY["vlm_ready (VLM + Guardrail complete)"]:::active
    CLM["claimed (SQLite lease lock, TTL=120s)"]:::warn
    REV["reviewed (Reviewer submitted 3-layer edits)"]:::active
    FLG["flagged (Reviewer flagged issue / unknown)"]:::warn
    APP["approved (Lead approved)"]:::good
    GTF["gt_frozen (Immutable GT benchmark)"]:::frozen

    ING -- "batch_vlm / run-vlm" --> QUE
    ING -- "run-vlm direct" --> RUN
    RDY -- "batch_vlm rerun" --> QUE
    QUE --> RUN --> RDY
    QUE -- "cancel rollback (no active_run_id)" --> ING
    QUE -- "cancel rollback (has active_run_id)" --> RDY
    RUN -- "cancel / fail rollback (prev ingested)" --> ING
    RUN -- "cancel / fail rollback (prev vlm_ready)" --> RDY
    RDY -- "POST /claim" --> CLM
    CLM -- "POST /release or TTL expired" --> RDY
    CLM -- "submit_review" --> REV
    CLM -- "flag" --> FLG
    RDY -- "Lead approve" --> APP
    REV -- "Lead approve" --> APP
    FLG -- "Lead arbitrate & approve" --> APP
    APP -- "Lead freeze_gt" --> GTF
    REV -- "Lead freeze_gt" --> GTF
```

**Cooperative Cancel & Atomic Rollback**: When a `batch_vlm` job is cancelled mid-flight (`POST /api/jobs/{job_id}/cancel`) or fails, `VlmClient.cancel()` sets `cancel_event` and shuts down active TCP sockets (~1 ms wakeup, including ephemeral sockets created during blind-node retries), raising `VlmCancelledError` across stages, BAFT tracking, and pre-commit checks (`before_commit_check`). `Repository.restore_vlm_episodes_on_cancel` atomically reverts only episodes still in transient `vlm_queued`/`vlm_running` states back to their pre-job restorable status (`vlm_ready` if previously ready or `active_run_id` exists, otherwise `ingested`) and emits `episode_updated` SSE notifications — never overwriting completed `vlm_ready` or protected review states (`claimed`, `reviewed`, `flagged`, `approved`, `gt_frozen`).

**Non-Destructive Task Reassignment**: Reassigning an episode's `task_id` (`PATCH /api/episodes/{ep_id}/task` or Lead `POST /api/episodes/batch-task`) is rejected with `409 Conflict` if the episode is `vlm_queued` or `vlm_running` (or actively reserved), requires Lead `allow_frozen=true` confirmation on `gt_frozen` episodes, and allows `Reviewer` edits strictly on non-frozen `_default` episodes (`_default` → existing non-`_default` task). Reassignment updates only `episodes.task_id` and `updated_at` — never clearing `active_run_id`, `active_rev_id`, or `status` — emits `episode_updated` SSE, and signals `re_run_recommended=true` whenever an active VLM run already exists so the user can opt into re-running VLM under the new `TaskSpec`.

---

## 3. Authoritative Pydantic v2 Schemas & SQLite WAL Store

### 3.1 Three-Layer Domain Models (`robo_vlm_qc/models/`)

All models enforce `ConfigDict(extra="forbid")` and use clean language-neutral field names (`name`, `title`, `label`) while accepting legacy `*_zh` input keys via a `mode="before"` normalization validator for backward compatibility with older JSON/YAML files. Abolished rule fields (`long_pause`, `back_and_forth`) are explicitly rejected at validation time by `_FORBIDDEN_LEGACY_NAMES` in [`robo_vlm_qc/models/annotation.py`](../robo_vlm_qc/models/annotation.py).

1. **`VlmRunArtifact` (`models/annotation.py`)**:
   - Metadata & provenance: `run_id`, `episode_id`, `created_at`, `is_eval_sandbox`, `config_fingerprint` (`ConfigFingerprintSnapshot`), `stage_refs` (`StageRefs`), `usage` (`UsageSummary`).
   - Read-only visual & kinematic aids:
     - `grounding` (`GroundingResult`, backward-compatible `schema_version=1` sparse or `schema_version=2` BAFT dense): `camera_name` (enforced by `StudioPlayer` and `viz_mp4` so boxes only overlay on the grounded camera), `ref_size`, `keyframes_s`, `track_fps`, `dense_times_s` (monotonically non-decreasing dense sample timestamps validated by `_validate_dense_alignment`), `camera_motion_detected`, and `object_tracks_2d: dict[str, GroundingTrack]`. Each `GroundingTrack` carries `object_id`, `name` (accepting `name_zh` alias), `tag_pos`, `is_static_container`, sparse `keyframes` (`[0, 1000]` normalized bboxes), `tracking_mode` (`"sparse_vlm" | "static_locked" | "bidir_flow_ncc"`), `dense_boxes_norm_1000`, `dense_flags` (`0=invisible` requiring `[0, 0, 0, 0]`, `1=tracked`, `2=vlm_anchor`, `3=static_locked`), and `dense_conf_pct` (`0..100`, validated by `_validate_dense`).
     - `arm_timelines` (`ArmTimelines`: `left` / `right` `ArmSpan` with `state ∈ {idle, work, hold, anom, mask, good}`).
   - Editable target baselines: `macro_stages` (`list[MacroStage]` with `label`, optional `label_en`, and VLA training `instruction`), `qc_segments` (`list[QcSegment]` with `name`), `success` (`SuccessVerdict`).
   - **`QcSegment` Triplet Consistency** (enforced by `_validate_triplet_and_span`):
     - `keep_positive`: requires `mask_side is None` and `training_role ∈ {"necessary", "recovery"}`.
     - `mask_loss`: requires `mask_side ∈ {"left", "right", "both"}` and `training_role ∈ {"badcase", "terminal_fail", "recovery"}`.
     - `cut_clean`: requires `mask_side is None` and `training_role ∈ {"badcase", "terminal_fail"}`.
     - Every segment preserves `vlm_raw_policy` alongside `guardrail` (`GuardrailReport`: `gate_type`, `passed`, `downgraded`, `downgrade_reason`, `guardrail_override`, `override_reason`, `metrics`, `opt_t_start`, `opt_t_end`).

2. **`HumanReviewArtifact` & `ReviewWritePayload` (`models/review.py`)**:
   - `ReviewWritePayload` accepts ONLY `base_run_id`, `action` (`save_draft | submit_review | flag | approve | freeze_gt`), `qc_segments`, `success`, `macro_stages`, `watch_coverage` (`WatchCoverage`), `notes`, and `flagged_reason`.
   - `_reject_readonly_layers` rejects any payload containing `grounding`, `arm_timelines`, or `raw_vlm_segments`, and `extra="forbid"` rejects unknown fields such as `author` (server derives `author` from `X-User` / auth context).
   - `HumanReviewArtifact` records immutable snapshots with computed `diff_summary` (`ReviewDiff`: `fp_deleted`, `fn_added`, `policy_flips`, `label_changes`, `boundary_nudges`, `macro_stage_edits`, `success_flipped`, `total_edits`).

3. **`ResolvedEpisode` (`models/review.py`, `frozen=True`)**:
   - The sole read view consumed by the Studio UI, `viz_mp4`, `PieceSlicer`, and `OutputExporter`s.
   - Overlays `latest_review.{qc_segments, success, macro_stages}` onto `active_vlm_run`, while preserving read-only `grounding`, `arm_timelines`, `raw_vlm_segments`, `raw_vlm_success`, and `raw_vlm_macro_stages`.

### 3.2 SQLite WAL 6-Table Schema (`robo_vlm_qc/store/db.py`)

`DatabasePool` configures `PRAGMA journal_mode = WAL`, `PRAGMA synchronous = NORMAL`, `PRAGMA foreign_keys = ON`, `PRAGMA busy_timeout = 5000`, and serializes writes via `BEGIN IMMEDIATE` under an `RLock`:

| Table | Purpose | Key Columns |
| :--- | :--- | :--- |
| `schema_migrations` | Schema version tracking (`SCHEMA_VERSION = 1`) | `version`, `applied_at` |
| `episodes` | Multi-format registry, status machine, claim lease locks, summary metrics | `episode_id` (`"<dataset>/<stem>"`), `dataset`, `task_id`, `adapter_type`, `source_uri`, `source_mtime`, `duration_s`, `fps`, `has_kinematics`, `camera_roles_json`, `meta_json`, `status`, `active_run_id`, `active_rev_id`, `claimed_by`, `claim_expires_at`, `resolved_passed`, `cut_ratio`, `mask_ratio`, `badcase_count`, `unknown_count` |
| `vlm_runs` | Multi-version VLM run & GT sandbox metadata | `run_id`, `episode_id`, `is_eval_sandbox`, `model_routing_json`, `prompt_version`, `label_version`, `task_spec_version`, `few_shot_pack_id`, `artifact_path`, `status`, `tokens_total`, `elapsed_s`, `error_message` |
| `reviews` | Full audit trail of human review snapshots | `rev_id`, `episode_id`, `base_run_id`, `author`, `role`, `action`, `artifact_path`, `diff_stats_json` |
| `few_shot_examples` | `VLM Lab` mistake book entries | `example_id`, `scope` (`global\|task:<id>\|label:<id>`), `task_id`, `label_id`, `error_type`, `episode_id`, `t_start`, `t_end`, `tile_image_path`, `vlm_wrong_json`, `human_right_json`, `rationale`, `enabled` |
| `jobs` | Async batch VLM, `render_mp4`, `export`, `eval_gt`, and `ingest` job tracking | `job_id`, `job_type`, `status`, `params_json`, `progress_done`, `progress_total`, `summary_json` |

### 3.3 Workspace On-Disk Layout

```text
<workspace>/
├── robo_vlm_qc.db                                  # SQLite WAL index database
├── stage_cache/<sha256[:2]>/<sha256>.json          # Content-addressed VLM stage cache
├── artifacts/
│   ├── vlm_runs/<dataset>/<stem>/<run_id>.json     # Immutable VlmRunArtifact JSON
│   ├── reviews/<dataset>/<stem>/<rev_id>.json      # Immutable HumanReviewArtifact JSON
│   ├── few_shots/<example_id>.jpg                  # 2x2 start-vs-end visual Few-Shot tiles
│   ├── eval_runs/<eval_id>.json                    # GT sandbox EvalRunReport JSON
│   ├── exports/<export_id>.json                    # Persisted ExportManifest JSON
│   ├── previews/<dataset>/<stem>/                  # H.264 CFR browser preview MP4 cache
│   └── unknown_triage.json                         # Triaged unknown anomaly ledger
├── configs_override/                               # Workspace-level YAML overrides
├── exports/<export_id>/                            # Atomic _staging -> rename export bundles
└── .history/<rel_parent>/<rel_stem>/<ts>_<id>.json # Pre-modification config snapshots
```

---

## 4. Pluggable 3-In I/O Kernel & Adaptive 2×2 Tiler

[`robo_vlm_qc/io/base.py`](../robo_vlm_qc/io/base.py) defines `InputAdapter`, `MediaSource`, `KinematicsTrack` (optional), `EpisodeHandle`, and `EpisodeView`. Built-in adapters are registered in priority order in [`robo_vlm_qc/io/__init__.py`](../robo_vlm_qc/io/__init__.py):

| Priority | Adapter | Source Format | Key Capabilities |
| :-: | :--- | :--- | :--- |
| `10` | `h5_raw_v3` | `raw_only_v3` `.h5` | Aligns 5 async streams (`head_rgbd`, `head_stereo`, `torso_rgbd`, `hand`, `robot`) onto common 30 Hz window; quaternion sign-continuity + Slerp; configurable `hand_a`/`hand_b` mapping; native tactile dropout repair; `3200×1296` SBS stereo L/R virtual crop; O(K) JPEG decode; thread-local `h5py` handles. |
| `20` | `lerobot` | LeRobot v2 / v3 | Parses `meta/info.json`, chunked Parquet state/action rows (preserving native `state_names` / `action_names`), `meta/tasks.parquet` / `episodes`, and multi-cam MP4 streams or embedded images. |
| `90` | `video_dir` | Single/Multi-cam MP4 directory | Pure-video ingestion (`kinematics = None`), including egocentric human-hand demonstration videos; triggers `visual_gate` instead of kinematics guardrails. |

### Adaptive Tile Composer & Start/End Compare ([`robo_vlm_qc/io/media.py`](../robo_vlm_qc/io/media.py))

`select_tile_mode(camera_roles)` chooses the tile geometry sent to VLM stages (with widened `DEFAULT_MAIN_ZOOM = (0.12, 0.10, 0.88, 0.86)` covering left-arm reach and foreground drop zones):
- **`wrist_2x2`** (when `left_wrist` or `right_wrist` is present): `[L-WRIST | R-WRIST / MAIN | MAIN-ZOOM]`
- **`torso_head_2x2`** (when `front`, `wide`, `wide_right`, or `secondary` is present without wrist cams): `[TORSO|WIDE|AUX | *-ZOOM / HEAD | HEAD-ZOOM]`
- **`single_1x2`** (single camera): `[MAIN | MAIN-CENTER-ZOOM]`
- **`compose_start_end_compare`**: Renders high-resolution side-by-side `START vs END` / `START vs CUTOFF` comparison tiles (`640×360` per panel default) for Stage 5 completion verification.

---

## 5. `StageCache` Isolation & Multi-Provider `VlmClient`

### 5.1 Content-Addressed `StageCache` Keys (`config.py` + `vlm/cache.py`)

Each stage stores atomic JSON under `stage_cache/<sha256[:2]>/<sha256>.json` keyed by:
`sha256(episode_fingerprint, upstream_hash, stage_basis)`.

Crucial isolation guarantees in [`ConfigManager`](../robo_vlm_qc/config.py):
- Every stage cache basis and `model_routing_sha256` includes `vlm_provider()` (plus any explicit `model_override`), guaranteeing `openai_compat` and `gemini` caches never collide.
- `s1_cache_basis(task_id)`: depends ONLY on `s1_grounding` provider/model/prompt/settings and task `grounding_objects`/`goal`/`task_prompt`.
- `s2_cache_basis(task_id)`: depends ONLY on `s2_macro` provider/model/prompt/settings and task `goal`/`steps`/`task_prompt`.
- `s5_cache_basis(task_id)`: depends ONLY on `s5_success` provider/model/prompt/settings and task `goal`/`criteria`/`task_prompt`.
- `s3_cache_basis(task_id, few_shot_hash)` and `s4_cache_basis(task_id, few_shot_hash)`: include `label_taxonomy_sha256`, full `TaskSpec`, and `few_shot_hash`.

Therefore, updating QC labels (`labels.yaml`), S3/S4 prompts, or Few-Shot examples in `VLM Lab` yields **100% cache hits (0 tokens, 0 upstream HTTP calls)** on `S1`, `S2`, and `S5`, re-executing only `S3` and `S4`. Likewise, because `s2_key`–`s5_key` do not include `s1_key` in `upstream_hash`, re-running `S1` alone (e.g., after tuning `s1_grounding`, `tracking.*`, or `custom_objects`) preserves **100% cache hits** on `S2`–`S5`.

### 5.2 Multi-Provider `VlmClient` & Resilience (`vlm/client.py`)

- **Dual Native Protocols**: Supports `openai_compat` (OpenAI, DashScope Qwen, local vLLM, or any OpenAI-compatible endpoint) and native Google **Gemini `/v1beta`** (`generateContent` with `inlineData`, model-family-aware `thinkingLevel`/`thinkingBudget` + `maxOutputTokens` headroom, `thought: true` separation, safety-block detection, `mediaResolution: MEDIA_RESOLUTION_HIGH`, and `?key=` + `x-goog-api-key` auth).
- **Defaults & Hot-Swapping**: Defaults to `provider="openai_compat"`, `base_url="https://api.openai.com/v1"`, `api_key_env="OPENAI_API_KEY"`, `key_file=".env"`, `default_model="gpt-4o"`, `thinking: false`, and `thinking_mode="auto"`. Ready-to-use provider overrides live in [`configs/presets/`](../configs/presets/), and Leads can switch providers/models/keys on the fly via the TopNav `Model / API` modal (`GET/PUT /api/settings/vlm`, `POST /api/settings/vlm/test`) or `robo-vlm-qc config-vlm` (`0600` `.env` secret isolation).
- **Resilience Guards**:
  - **Blind-Node Vision Token-Density Guard (`VlmBlindVisionError`)**: Detects degraded high-res Gemini vision responses (~258 tok/img instead of HD density) and retries up to `vision_blind_node_retries=5` over fresh ephemeral TCP connections (tracked in `_ephemeral_clients` so `cancel()` aborts them immediately; `vision_blind_node_guard: "auto"` skips loopback test mocks).
  - **Circuit Breaker (`VlmCircuitBreaker` / `VlmCircuitOpenError`)**: Trips immediately on HTTP `403`/quota errors or after `failure_threshold=3` consecutive failures; HTTP `503` `model_not_found` fails fast without burning retries.
  - **Secret Redaction (`redact_secrets`)**: Attaches `_SecretRedactingLogFilter` to `httpx`/`httpcore` loggers and scrubs `?key=`, `Bearer`, `x-goog-api-key`, `sk-*`, and `AIza*` from logs and exception messages.
  - **Opt-In Fallback, Cost & Cancellation**: Supports optional `fallback_model`/`fallback_provider`, stage-level `stage_models` + `model_override`, `(stage, model)` `vlm.pricing` USD cost accounting (preventing reasoning-token double-counting when `reasoning_included_in_completion=True`), DAG wall-clock `usage.total_elapsed_s`, interruptible backoff retries (`1.0s..16.0s`), cooperative `VlmClient.cancel()` (`VlmCancelledError`), and hierarchical `BudgetTracker` (`ceil(duration_s * 5000)`, minimum `200000` tokens unless `unlimited: true`).

---

## 6. Light Guardrails (`robo_vlm_qc/guardrail/`)

Guardrails **never generate new QC segments** and **never run 30 Hz kinematic rule detectors**. They validate VLM proposals and preserve `vlm_raw_policy`:

1. **`hallucination.py` (`apply_hallucination_check`)** — active when `KinematicsTrack != None` (`enabled: true`):
   - Checks `mid_idle_freeze` and `boundary_idle` segments in an **arm-aware** manner (`affected_arm == "left"` checks left arm, `"right"` checks right arm, `"both"` checks if any arm moved monotonically).
   - If the checked arm exhibits monotonic net displacement `> 0.15 m` (`net >= 0.5 * path_length`), overrides `splice_policy` to `keep_positive` (`mask_side=None`, `guardrail_override=True`) and logs `override_reason`.
2. **`splice_gate.py` (`apply_splice_gate`)** — active when `KinematicsTrack != None`:
   - `boundary_idle` within `boundary_edge_tol_s = 0.5s` of trajectory edges (`0` or `duration_s`) passes directly and snaps `t_start`/`opt_t_start` to exact `0.0` (or `t_end`/`opt_t_end` to `round(duration_s, 3)`) to eliminate orphan edge slivers.
   - Interior `cut_clean` candidates undergo a 2D grid search in `[t_start ± 1.0s, t_end ± 1.0s]` minimizing pose jump cost subject to:
     - `max_pos_jump_m <= 0.02` (2.0 cm)
     - `max_rot_jump_rad <= 0.10` (0.10 rad, SO(3) geodesic)
     - `max_hand_jump_rad <= 0.30` (0.30 rad)
     - `max_torso_jump_rad <= 0.05` (0.05 rad)
     - `max_chassis_pos_jump_m <= 0.02` (2.0 cm) & `max_chassis_yaw_jump_rad <= 0.05` (0.05 rad)
     - Bimanual coupling: inactive arm excursion `max_other_arm_disp_m < 0.03` (3.0 cm).
   - If feasible and uncoupled: snaps `[t_start, t_end]` to `(opt_t_start, opt_t_end)`. Otherwise **downgrades `splice_policy` to `mask_loss`** (`mask_side = seg.affected_arm`) and populates `guardrail.downgrade_reason`.
3. **`visual_gate.py` (`apply_visual_gate`)** — active when `KinematicsTrack is None`:
   - Compares start/end main-camera frames for non-edge `cut_clean` candidates; requires `visual_ssim >= 0.85` and `l1_diff <= 0.12`, otherwise downgrades to `mask_loss`.

---

## 7. `PieceSlicer` & 4-Out Transactional Exporters (`robo_vlm_qc/export/`)

### 7.1 `PieceSlicer` (`export/slicer.py`)

Consumes `ResolvedEpisode` + `EpisodeView` + `ExportRecipe` (`min_piece_s = 0.1`, `smooth_tail = on_jump | always | never`, optional `split_by_subtask = true|false`):

- **3 Recipes**:
  - `bc_clean`: keeps passed episodes (`success.passed is True`); removes safe `cut_clean` intervals; assigns `[0.0, 0.0]` / per-arm `action_loss_mask` to `mask_loss` intervals (`recovery`, `badcase`, `terminal_fail`).
  - `recovery_plus`: keeps passed episodes; removes safe `cut_clean` intervals; keeps `recovery` segments unmasked (`1.0`), masking only `badcase` / `terminal_fail`.
  - `badcase_pairs`: includes both passed and failed episodes; slices explicit `badcase` / `terminal_fail` windows padded by `context_window_s` (default `2.0s`).
- **Action Chunk Tail Smoothing**:
  - Computes smoothstep weights $w = t^2(3 - 2t)$ over `tail_rows` and applies SO(3) geodesic blending to quaternions/rotvecs and wrapped blending to `chassis_dyaw`.
  - `NOT_SMOOTHED = ("tau", "torque", "force", "wrench", "tactile", "age_ms")` and timestamp arrays (`*_ns`, `sensor_time`, `timestamps`, `source_timestamps`) are strictly excluded from smoothing.
- **Canonical 81-D Layout (`CANONICAL_81D_NAMES`) & Adaptive Native Preservation**:
  - `chassis_dx, chassis_dy, chassis_dyaw` (3) + `torso_0..3` (4) + `arm_left_0..6` (7) + `arm_right_0..6` (7) + `head_0..1` (2) + `ee_left_{x,y,z,qw,qx,qy,qz}` (7) + `ee_right_{x,y,z,qw,qx,qy,qz}` (7) + `hand_left_0..21` (22) + `hand_right_0..21` (22) = **81 dims**.
  - `lerobot_v3` supports `state_schema_mode` (`auto` | `preserve_native` | `canonical_81d`): `auto` preserves native dimensions and feature names for uniform LeRobot datasets (e.g., 14-D ALOHA, 7-D Franka), omits dummy zero state/action columns on pure-video `video_dir` datasets, and unifies H5 or heterogeneous mixes to canonical 81-D.
- **Per-frame outputs & Subtask Splitting**: `action_loss_mask` `(N, 2)` float32, `is_tail` bool, `subtask_index` int64 (aligned from `macro_stages`). When `split_by_subtask=True`, `PieceSlicer` splits each kept piece along aligned subtask boundaries so every clean subtask span is exported as an independent episode with `task` set to `MacroStage.instruction`.

### 7.2 4 Transactional Exporters (`_staging → os.replace`)

Every exporter runs inside `staged_export_dir` ([`export/base.py`](../robo_vlm_qc/export/base.py)) under `fcntl.flock` + thread lock, writing to `<out>.<name>._staging` and atomically replacing `<out>`:

1. **`lerobot_v3`** (`export/lerobot_v3.py`): Emits LeRobot v3.0 chunked Parquet (`data/chunk-XXX/file-XXX.parquet` with adaptive native or 81-D state/action, `action_loss_mask`, `is_tail`, `subtask_index`), CFR `libx264` MP4 camera streams (`videos/<cam>/chunk-XXX/file-XXX.mp4`), `meta/episodes/`, `meta/info.json` (with configurable `export.robot_type`, default `"bimanual_robot"`), `meta/stats.json`, `meta/tasks.parquet`, `meta/subtasks.parquet` (`subtask_index`, `subtask`, `stage_id`, `label`, `label_en`, `instruction`), and `meta/qc_export_manifest.json`.
2. **`h5_clean`** (`export/h5_clean.py`): Emits `<dataset>__<stem>.h5` preserving sliced raw sensor/JPEG streams plus `/qc/action_loss_mask`, `/qc/is_tail`, `/qc/subtask_index`, and `/qc/resolved_json`.
3. **`zero_copy_index`** (`export/zero_copy_index.py`): Emits `qc_index.parquet`, `manifest.jsonl`, and `subtasks.json` referencing original read-only `source_uri` paths and row bounds without duplicating raw media.
4. **`viz_mp4`** (`export/viz_mp4.py`): Renders `1920×1080@30fps` Studio HUD MP4s with adaptive camera mosaic, crop-projected v2 dense 2D Grounding overlays (guarded by `camera_name`), 4-track timeline (`macro_stages`, `arm_timelines` L/R, `qc_segments`), guardrail badges, and crisp English status overlays.
