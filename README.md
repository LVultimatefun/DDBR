# `robo_vlm_qc`

**VLM-Native Embodied Trajectory Quality Control, Dual-Role Human Review, and Multi-Format Robotics Data Platform**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Node 22+](https://img.shields.io/badge/node-22%2B-brightgreen.svg)](web/package.json)
[![CI](https://github.com/LVultimatefun/DDBR/actions/workflows/ci.yml/badge.svg)](https://github.com/LVultimatefun/DDBR/actions/workflows/ci.yml)

English | [Chinese](README_zh.md)

---

`robo_vlm_qc` is a standalone, embodiment-agnostic quality-control and dataset-curation platform for robotics manipulation trajectories. Instead of brittle 30 Hz kinematic threshold heuristics, it uses a **causal multi-stage Vision-Language Model (VLM) DAG** paired with **lightweight splice-safety guardrails**, a **unified Dual-Role Web Studio (`Reviewer` & `Lead`)**, a **pure In-Context `VLM Lab`** iteration loop (**zero VLM SFT/DPO/LoRA fine-tuning export**), and a pluggable **3-In × 4-Out** I/O architecture.

---

## Key Capabilities

- **3-In Pluggable Read-Only I/O Kernel ([`robo_vlm_qc/io/`](robo_vlm_qc/io/))**: Strictly read-only ingestion across 3 built-in `InputAdapter` implementations (`h5_raw_v3`, `lerobot` v2/v3, `video_dir` with `kinematics=None`), unified `MediaSource` + optional `KinematicsTrack`, and camera-role-adaptive 2×2 / 1×2 tile composition (`wrist_2x2`, `torso_head_2x2`, `single_1x2`).
- **Causal VLM DAG & Light Guardrails ([`robo_vlm_qc/vlm/`](robo_vlm_qc/vlm/), [`robo_vlm_qc/guardrail/`](robo_vlm_qc/guardrail/))**: Multi-provider `httpx` `VlmClient` (OpenAI-compatible, DashScope, Gemini, local vLLM) with stage-level model routing, dynamic duration token budgets, content-addressed `StageCache` (`stage_cache/<sha256[:2]>/<sha256>.json`), and 3-level concurrent DAG `S0 → S1 → (S2 ∥ S5) → S3 → S4A → S4B → Guardrail → Store`. Lightweight guardrails (`hallucination`, `splice_gate`, `visual_gate`) prevent cut-induced teleportation by downgrading unsafe `cut_clean` intervals to `mask_loss`.
- **11 Pure-Visual Labels ([`configs/labels.yaml`](configs/labels.yaml))**: Strictly visual taxonomy decoupled from splice policies, paired with Precision-First stage prompts ([`configs/prompts/`](configs/prompts/)) and generic task specs ([`configs/tasks/`](configs/tasks/)).
- **Single-Truth Store ([`robo_vlm_qc/store/`](robo_vlm_qc/store/))**: Authoritative Pydantic v2 schemas (`VlmRunArtifact`, `HumanReviewArtifact`, `ResolvedEpisode`) backed by SQLite WAL (`robo_vlm_qc.db`), immutable JSON artifacts (`artifacts/`), and `.history/` configuration snapshots.
- **Unified Dual-Role Studio ([`robo_vlm_qc/api/`](robo_vlm_qc/api/), [`web/src/`](web/src/))**: Single FastAPI + React 19 + TypeScript + Vite SPA serving `Reviewer` and `Lead` workflows over identical `StudioPlayer` (2D Grounding canvas) and `FourTrackTimeline` components. Human edits are strictly scoped to 3 layers (`qc_segments`, `success`, `macro_stages`), keeping `arm_timelines`, `grounding`, and `raw_vlm_segments` read-only.
- **`VLM Lab` Pure In-Context Loop ([`robo_vlm_qc/lab/`](robo_vlm_qc/lab/))**: Automated 6-bucket error diagnosis (`VLM Raw vs Human GT`), 1-click 2×2 visual/text Few-Shot mistake book injected into S3/S4, unknown anomaly triage (`merge` / `promote`), versioned config hot-updates with `.history/` diff & rollback, advisory LLM prompt diff suggestions (`prompt_assist`), and a GT regression eval sandbox reusing S1/S2/S5 cache 100% (**strictly In-Context — zero VLM SFT/DPO/LoRA export**).
- **4-Out Transactional Exporters ([`robo_vlm_qc/export/`](robo_vlm_qc/export/))**: Shared `PieceSlicer` (smoothstep + SO(3) geodesic tail smoothing, `is_tail`, `action_loss_mask`, `subtask_index`, canonical 81-D kinematics) and atomic `_staging → rename` exporters (`lerobot_v3`, `h5_clean`, `zero_copy_index`, `viz_mp4`) across 3 training recipes (`bc_clean`, `recovery_plus`, `badcase_pairs`).

---

## Architecture Overview

```mermaid
flowchart LR
    subgraph IN["3-In Read-Only Sources"]
        I1["h5_raw_v3 (.h5)"]
        I2["lerobot (v2/v3)"]
        I3["video_dir (MP4)"]
    end

    subgraph CORE["Causal VLM DAG & Store"]
        DAG["S1 Grounding -> S2||S5 -> S3 Discover -> S4 Verify"]
        GRD["Light Guardrails (splice_gate / visual_gate)"]
        DB["SQLite WAL + Immutable Artifacts + StageCache"]
    end

    subgraph UI["Unified Dual-Role Studio & VLM Lab"]
        STUDIO["Reviewer & Lead Studio (3-Layer Human Edits)"]
        LAB["VLM Lab Pure In-Context Loop (Diagnosis, Few-Shot, GT Eval)"]
    end

    subgraph OUT["4-Out Transactional Exporters"]
        O1["lerobot_v3"]
        O2["h5_clean"]
        O3["zero_copy_index"]
        O4["viz_mp4 (1080p HUD)"]
    end

    I1 & I2 & I3 --> DAG --> GRD --> DB
    DB <--> STUDIO
    STUDIO --> LAB
    LAB -- "In-Context Rules & Few-Shots" --> DAG
    DB --> O1 & O2 & O3 & O4
```

See [`docs/architecture.md`](docs/architecture.md) for full architectural decisions, Pydantic schemas, cache isolation rules, and exporter contracts.

---

## 30-Second Synthetic Demo Quickstart

Try `robo_vlm_qc` end-to-end with zero external datasets or API keys:

### 1. Install Prerequisites & Package

- **Python** `>= 3.12`
- **FFmpeg** (`ffmpeg` and `ffprobe` on `PATH`)
- **Node.js** `>= 22` and `npm` (required once to build the Web Studio frontend)

```bash
git clone https://github.com/LVultimatefun/DDBR.git
cd DDBR

# Install Python package + CLI (`robo-vlm-qc`)
pip install -e ".[dev]"

# Build Web Studio frontend bundle (`web/dist/`)
cd web && npm ci && npm run build && cd ..
```

### 2. Generate Synthetic Demo Data, Ingest, and Launch Studio

```bash
# Generate 2 multi-camera video_dir episodes + 1 LeRobot v3 episode (with 81-D kinematics)
python scripts/generate_demo_dataset.py --out ./demo_data

# Ingest strictly read-only into a local workspace
robo-vlm-qc -w ./workspace ingest --data ./demo_data

# Launch Dual-Role Studio locally on 127.0.0.1:8000 (unauthenticated loopback dev mode)
robo-vlm-qc -w ./workspace serve --no-auth
```

Open `http://127.0.0.1:8000` in your browser to explore the Queue, multi-camera Review Studio, Export center, and `VLM Lab`.

> **Security Note**: `robo-vlm-qc serve` binds `127.0.0.1` by default and requires explicit tokens (`--reviewer-token` / `--lead-token` or `ROBO_VLM_QC_*_TOKEN`) unless `--no-auth` is passed. For shared LAN or Docker deployments, always configure distinct tokens, `--allowed-data-roots`, and `--cors-origin` / `ROBO_VLM_QC_CORS_ORIGINS`.

---

## Running the Full VLM QC & Export Pipeline

### 1. Configure Credentials & Provider Preset

Copy [`.env.example`](.env.example) to `.env` and set your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (or another provider key below)
```

`configs/default.yaml` defaults to an OpenAI-compatible endpoint (`https://api.openai.com/v1`, `OPENAI_API_KEY`, `gpt-4o`, `thinking: false`, `thinking_mode: auto`). Ready-to-use provider presets live in [`configs/presets/`](configs/presets/):

| Preset File | Provider | Default Model | API Key Env | Thinking Mode |
| :--- | :--- | :--- | :--- | :--- |
| [`configs/presets/openai.yaml`](configs/presets/openai.yaml) | `openai_compat` | `gpt-4o` | `OPENAI_API_KEY` | `disabled` (`thinking: false`) |
| [`configs/presets/dashscope.yaml`](configs/presets/dashscope.yaml) | `openai_compat` | `qwen-vl-max` | `DASHSCOPE_API_KEY` | `dashscope` (`thinking: true`) |
| [`configs/presets/gemini.yaml`](configs/presets/gemini.yaml) | `gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` | `auto` (`thinking: true`) |
| [`configs/presets/vllm.yaml`](configs/presets/vllm.yaml) | `openai_compat` | `Qwen/Qwen2.5-VL-72B-Instruct` | `VLLM_API_KEY` | `disabled` (`thinking: false`) |

Activate any preset in your workspace by copying it to `<workspace>/configs_override/default.yaml`:

```bash
mkdir -p ./workspace/configs_override
cp configs/presets/gemini.yaml ./workspace/configs_override/default.yaml
```

### 2. Run VLM DAG, Review, Evaluate, and Export

```bash
# 1. Run causal VLM DAG (S1 -> S2||S5 -> S3 -> S4A -> S4B) + light guardrails
robo-vlm-qc -w ./workspace run-vlm --workers 4

# 2. Launch authenticated Dual-Role Studio
robo-vlm-qc -w ./workspace serve \
  --reviewer-token "$ROBO_VLM_QC_REVIEWER_TOKEN" \
  --lead-token "$ROBO_VLM_QC_LEAD_TOKEN" \
  --allowed-data-roots ./demo_data

# 3. (After Lead freezes GT episodes) Run VLM Lab GT regression sandbox
robo-vlm-qc -w ./workspace eval --few-shot --json

# 4. Export curated training datasets (4 formats x 3 recipes)
robo-vlm-qc -w ./workspace export \
  --format lerobot_v3 \
  --recipe bc_clean \
  --out ./exports/lerobot_bc_clean

# 5. Render standalone 1080p@30fps Studio HUD visualization MP4s
robo-vlm-qc -w ./workspace render-mp4 --all-ready
```

---

## Docker Deployment

Build and run the complete FastAPI + React Studio stack (multi-stage Node 22 + Python 3.12 + FFmpeg) via [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml):

```bash
cp .env.example .env
# Edit .env to set ROBO_VLM_QC_REVIEWER_TOKEN, ROBO_VLM_QC_LEAD_TOKEN, and your VLM API key

mkdir -p workspace data
docker compose up --build
```

The Compose stack mounts `./workspace` (read-write) and `./data` (read-only, enforced by `--allowed-data-roots /data`), reads `.env`, and serves Studio on `http://localhost:8000`.

---

## Core Concepts Summary

| Dimension | Design Summary |
| :--- | :--- |
| **3-In Adapters** | `h5_raw_v3` (5-stream 30 Hz sync, Slerp, tactile repair, SBS L/R crop, O(K) JPEG), `lerobot` (v2/v3 Parquet + MP4), `video_dir` (pure-video `kinematics=None`). |
| **4-Out Exporters** | `lerobot_v3` (81-D Parquet, configurable `export.robot_type` default `"bimanual_robot"`, CFR MP4s, `subtasks.parquet`, `info.json`, `stats.json`), `h5_clean` (sliced raw streams + `/qc/*`), `zero_copy_index` (`qc_index.parquet`, `manifest.jsonl`, `subtasks.json`), `viz_mp4` (1920×1080@30fps HUD). |
| **3 Export Recipes** | `bc_clean` (passed episodes, positive segments + safe `cut_clean` removed, `mask_loss` masked to `0.0`), `recovery_plus` (also unmasks `recovery` segments), `badcase_pairs` (badcase/terminal_fail + context windows for contrast/failure analysis). |
| **11 Pure-Visual Labels** | `boundary_idle`, `mid_idle_freeze`, `missed_grasp_retry`, `object_slip_recovery`, `placement_redo`, `hesitation_oscillation`, `collision_or_knock`, `precision_fine_adjust`, `bimanual_assist_hold`, `terminal_failure`, `unknown_anomaly`. |
| **Example Tasks** | [`_default.yaml`](configs/tasks/_default.yaml), [`pick_and_place.yaml`](configs/tasks/pick_and_place.yaml), [`bimanual_handover.yaml`](configs/tasks/bimanual_handover.yaml). |
| **Causal DAG & Cache** | `S1` (2D Grounding) → `S2 ∥ S5` (Macro Backbone + L/R Timelines ∥ Dense-Tail Success + 30% Anti-FP Control) → `S3` (Sliding QC Discovery w/ S2 protection & Few-Shot) → `S4A/S4B` (Merge + 0.5s CoV Diff). Updating S3/labels/few-shots reuses S1/S2/S5 `StageCache` 100%. |
| **Dual-Role 3-Layer Editing** | `Reviewer` and `Lead` share identical `ReviewStudioPage` rendering. Humans edit only `qc_segments`, `success`, and `macro_stages`; `grounding`, `arm_timelines`, and `raw_vlm_segments` remain read-only. |
| **`VLM Lab` In-Context Loop** | `GT Freeze` → `Diagnosis (6 buckets)` → `2×2 Visual Few-Shot` / `Unknown Triage` / `Versioned Taxonomy` → `GT Sandbox Eval` → `Promote` (with `.history/` backup). **Zero VLM SFT/DPO/LoRA export.** |

---

## Documentation & Community

- **[`docs/architecture.md`](docs/architecture.md)** — System architecture decisions, Mermaid topology/DAG/state diagrams, Pydantic v2 & SQLite WAL schemas, `StageCache` isolation, guardrail thresholds, `PieceSlicer`, and 4 exporters.
- **[`docs/operations.md`](docs/operations.md)** — Complete CLI reference, YAML configs & provider presets, canonical FastAPI REST/SSE & RBAC contracts, Dual-Role Studio workflow, and `VLM Lab` In-Context SOP.
- **[`AGENTS.md`](AGENTS.md)** — Repository architecture map, dev/test/build toolchain, data-safety & architectural redlines, and Living Documentation Rules.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — Development setup, testing guide, and pull request checklist.
- **[`SECURITY.md`](SECURITY.md)** — Security policy and deployment hardening guidelines.
- **[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)** — Community standards (Contributor Covenant v2.1).
- **[`LICENSE`](LICENSE)** — Apache License 2.0 (`Copyright 2026 robo_vlm_qc Contributors`).
