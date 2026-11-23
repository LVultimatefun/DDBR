# Contributing to `robo_vlm_qc`

Thank you for your interest in contributing to `robo_vlm_qc`! Every contribution helps make open-source robotics dataset quality control more reliable, transparent, and embodiment-agnostic.

Before submitting a pull request, please review our [Code of Conduct](CODE_OF_CONDUCT.md) and architectural guidelines in [AGENTS.md](AGENTS.md).

---

## Development Setup

### Prerequisites

- **Python**: `>= 3.12`
- **System Video Tools**: `ffmpeg` and `ffprobe` (with `libx264`)
- **Frontend Toolchain** (for Web Studio changes): Node.js `>= 22` and `npm`

### Install Backend & Test Dependencies

```bash
git clone https://github.com/LVultimatefun/DDBR.git
cd DDBR
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Install & Build Web Studio Frontend

```bash
cd web
npm ci
CI=true npm test && npm run build
cd ..
```

### Generate Local Synthetic Demo Data

```bash
python scripts/generate_demo_dataset.py --out ./demo_data
robo-vlm-qc -w ./workspace ingest --data ./demo_data
```

---

## Running Tests

### Python Test Suite

```bash
# Run the full Python test suite (unit, api, and e2e)
pytest tests -q

# Run by semantic suite
pytest tests/unit -q
pytest tests/api -q
pytest tests/e2e -q
```

All tests must run offline against `tmp_path` workspaces and mock VLM servers — never depend on live external API keys or local machine paths.

### Frontend Test & Typecheck Suite

```bash
cd web
npm run typecheck
CI=true npm test
npm run build
```

---

## Core Architectural Invariants

Please keep these invariants intact in every contribution (see [docs/architecture.md](docs/architecture.md) for full details):

1. **Strict Read-Only Source Datasets**: Input adapters in `robo_vlm_qc/io/` must never write, rename, or delete files inside `--data` directories.
2. **Pure-Visual VLM Primary Labeling + Light Guardrails**: Do not add 30 Hz kinematic rule detectors or legacy rule fields (`long_pause`, `back_and_forth`). Guardrails validate VLM segments without generating new segments.
3. **3-Layer Human Edit Scope**: Human reviews edit only `qc_segments`, `success`, and `macro_stages`. Read-only visual layers (`grounding`, `arm_timelines`, `raw_vlm_segments`) remain immutable.
4. **Pure In-Context `VLM Lab`**: `robo_vlm_qc/lab/` is strictly an In-Context improvement loop (error diagnosis, 2×2 visual/text Few-Shot mistake book, versioned configs with `.history/`, GT regression sandbox). Do not add VLM SFT, DPO, LoRA, or PEFT fine-tuning export pipelines.
5. **Atomic Writes & Staged Exports**: Workspace overrides and artifacts must use atomic writes with `.history/` snapshots (`robo_vlm_qc/store/artifacts.py`), and exporters must stage under `_staging` before atomic rename (`robo_vlm_qc/export/base.py`).
6. **Living Documentation**: Keep [README.md](README.md), [AGENTS.md](AGENTS.md), [docs/architecture.md](docs/architecture.md), and [docs/operations.md](docs/operations.md) synchronized with code and schema changes.

---

## Pull Request Process

1. Fork the repository and create a descriptive feature or bugfix branch from `main`.
2. Write clean, typed code (Python type hints + Pydantic v2 models; TypeScript strict mode in `web/`).
3. Add or update tests covering your changes under `tests/unit/`, `tests/api/`, `tests/e2e/`, or `web/src/`.
4. Verify all Python tests, frontend tests/build, and wheel packaging pass cleanly.
5. Open a Pull Request using the [PR template](.github/PULL_REQUEST_TEMPLATE.md) and describe the motivation, design, and verification steps.

## Licensing

By contributing to `robo_vlm_qc`, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
