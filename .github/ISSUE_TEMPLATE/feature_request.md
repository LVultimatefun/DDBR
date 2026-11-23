---
name: Feature request
about: Suggest an enhancement, adapter, exporter, or workflow improvement
title: "[Feature]: "
labels: enhancement
---

## Problem / Motivation

Describe the robotics QC, review, or dataset curation workflow problem you want to solve.

## Proposed Design

Describe the proposed solution (e.g., new `InputAdapter`, `OutputExporter`, VLM provider preset, Studio UX improvement, or `VLM Lab` diagnostic metric).

## Alignment with Architectural Invariants

- [ ] Keeps source datasets strictly read-only
- [ ] Preserves pure-visual VLM primary labeling + light guardrails (no 30 Hz rule detectors)
- [ ] Keeps `VLM Lab` strictly In-Context (no VLM SFT/DPO/LoRA export)
- [ ] Maintains 3-layer human review scope (`qc_segments`, `success`, `macro_stages`)
