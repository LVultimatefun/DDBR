"""Layered YAML configuration manager and stage-isolated SHA256 fingerprints."""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .models.taxonomy import LabelRegistry, PromptBundle, PromptTemplate, TaskSpec

DEFAULT_VLM_PROVIDER = "openai_compat"
DEFAULT_VLM_MODEL = "gpt-4o"
DEFAULT_OPENAI_COMPAT_MODEL = "gpt-4o"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

PROMPT_STAGE_FILES: dict[str, tuple[str, ...]] = {
    "s1_grounding": ("s1_grounding.yaml",),
    "s2_macro": ("s2_macro.yaml", "s2_macro_backbone.yaml"),
    "s3_discover": ("s3_discover.yaml", "s3_qc_discovery.yaml"),
    "s4_verify": ("s4_verify.yaml", "s4_local_verify.yaml"),
    "s5_success": ("s5_success.yaml", "s5_end_state.yaml"),
}


def default_config_dir() -> Path:
    env_dir = os.environ.get("ROBO_VLM_QC_CONFIG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    pkg_dir = Path(__file__).resolve().parent
    candidates = (
        pkg_dir.parent / "configs",
        pkg_dir / "_bundled_configs",
        Path.cwd() / "configs",
    )
    for cand in candidates:
        if (cand / "default.yaml").is_file():
            return cand
    return candidates[0]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a deep copy of base."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def canonical_sha256(obj: Any) -> str:
    """Deterministic SHA256 hex digest of a JSON-serializable object."""
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")
    return raw


class ConfigFingerprints(BaseModel):
    config_sha256: str
    model_routing_sha256: str
    prompt_bundle_sha256: str
    stage_prompt_hashes: dict[str, str] = Field(default_factory=dict)
    label_taxonomy_sha256: str
    task_spec_sha256: str


class ConfigManager:
    """Loads base configs + workspace overrides and computes stage-isolated hashes."""

    def __init__(
        self,
        config_dir: Path | None = None,
        workspace_dir: Path | None = None,
    ) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else default_config_dir()
        self.workspace_dir = Path(workspace_dir) if workspace_dir is not None else None
        self.override_dir = (
            self.workspace_dir / "configs_override"
            if self.workspace_dir is not None
            else None
        )

        self.settings: dict[str, Any] = {}
        self.labels: LabelRegistry = LabelRegistry()
        self.prompts: PromptBundle
        self.tasks: dict[str, TaskSpec] = {}
        self.fingerprints: ConfigFingerprints
        self.reload()

    def reload(self) -> None:
        base_default = self.config_dir / "default.yaml"
        if not base_default.is_file():
            raise FileNotFoundError(f"Base config missing: {base_default}")
        settings = _load_yaml_dict(base_default)
        if self.override_dir is not None:
            ov_default = self.override_dir / "default.yaml"
            if ov_default.is_file():
                settings = deep_merge(settings, _load_yaml_dict(ov_default))
        self.settings = settings

        # Labels (replacement if overridden)
        labels_path = self.config_dir / "labels.yaml"
        if self.override_dir is not None and (self.override_dir / "labels.yaml").is_file():
            labels_path = self.override_dir / "labels.yaml"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels config missing: {labels_path}")
        self.labels = LabelRegistry.model_validate(_load_yaml_dict(labels_path))

        # Prompts (canonical and legacy filename alias resolution)
        prompt_objs: dict[str, PromptTemplate] = {}
        for stage, filenames in PROMPT_STAGE_FILES.items():
            found: Path | None = None
            if self.override_dir is not None:
                ov_prompts = self.override_dir / "prompts"
                for fname in filenames:
                    cand = ov_prompts / fname
                    if cand.is_file():
                        found = cand
                        break
            if found is None:
                base_prompts = self.config_dir / "prompts"
                for fname in filenames:
                    cand = base_prompts / fname
                    if cand.is_file():
                        found = cand
                        break
            if found is None:
                raise FileNotFoundError(
                    f"Prompt YAML for stage {stage!r} not found (tried {filenames})"
                )
            prompt_objs[stage] = PromptTemplate.model_validate(_load_yaml_dict(found))
        self.prompts = PromptBundle(**prompt_objs)

        # Tasks (deep_merge per task_id)
        tasks_raw: dict[str, dict[str, Any]] = {}
        base_tasks_dir = self.config_dir / "tasks"
        if base_tasks_dir.is_dir():
            for p in sorted(base_tasks_dir.glob("*.yaml")):
                d = _load_yaml_dict(p)
                tid = str(d.get("task_id") or p.stem)
                d.setdefault("task_id", tid)
                tasks_raw[tid] = d
        if self.override_dir is not None and (self.override_dir / "tasks").is_dir():
            for p in sorted((self.override_dir / "tasks").glob("*.yaml")):
                d = _load_yaml_dict(p)
                tid = str(d.get("task_id") or p.stem)
                d.setdefault("task_id", tid)
                if tid in tasks_raw:
                    tasks_raw[tid] = deep_merge(tasks_raw[tid], d)
                else:
                    tasks_raw[tid] = d
        self.tasks = {
            tid: TaskSpec.model_validate(data) for tid, data in sorted(tasks_raw.items())
        }
        if "_default" not in self.tasks:
            self.tasks["_default"] = TaskSpec(task_id="_default")

        self.fingerprints = self._compute_fingerprints()

    def _compute_fingerprints(self) -> ConfigFingerprints:
        stage_models = self.settings.get("vlm", {}).get("stage_models", {})
        stage_prompt_hashes = {
            stage: canonical_sha256(getattr(self.prompts, stage).model_dump(mode="json"))
            for stage in PROMPT_STAGE_FILES
        }
        model_routing_sha256 = canonical_sha256(
            {"provider": self.vlm_provider(), "stage_models": stage_models}
        )
        prompt_bundle_sha256 = canonical_sha256(stage_prompt_hashes)
        label_taxonomy_sha256 = canonical_sha256(self.labels.model_dump(mode="json"))
        task_spec_sha256 = canonical_sha256(
            {tid: spec.model_dump(mode="json") for tid, spec in sorted(self.tasks.items())}
        )
        config_sha256 = canonical_sha256(
            {
                "settings": self.settings,
                "model_routing_sha256": model_routing_sha256,
                "prompt_bundle_sha256": prompt_bundle_sha256,
                "label_taxonomy_sha256": label_taxonomy_sha256,
                "task_spec_sha256": task_spec_sha256,
            }
        )
        return ConfigFingerprints(
            config_sha256=config_sha256,
            model_routing_sha256=model_routing_sha256,
            prompt_bundle_sha256=prompt_bundle_sha256,
            stage_prompt_hashes=stage_prompt_hashes,
            label_taxonomy_sha256=label_taxonomy_sha256,
            task_spec_sha256=task_spec_sha256,
        )

    def get_task(self, task_id: str) -> TaskSpec:
        return self.tasks.get(task_id) or self.tasks["_default"]

    def task_patterns(self) -> dict[str, list[str]]:
        return {
            tid: list(spec.match)
            for tid, spec in sorted(self.tasks.items())
            if tid != "_default" and spec.match
        }

    def resolve_task(
        self, dataset: str, h5_task_name: str | None = None
    ) -> TaskSpec:
        ds_low = dataset.lower()
        ordered_ids = sorted(tid for tid in self.tasks if tid != "_default")

        # 1. Match dataset globs first
        for tid in ordered_ids:
            spec = self.tasks[tid]
            for pat in spec.match:
                if fnmatch.fnmatchcase(ds_low, pat.lower()):
                    return spec

        # 2. Match h5_task_name / hint against task_id, aliases, or match globs
        if h5_task_name:
            hint_low = h5_task_name.strip().lower()
            for tid in ordered_ids:
                spec = self.tasks[tid]
                if tid.lower() == hint_low:
                    return spec
                if any(al.lower() == hint_low for al in spec.aliases):
                    return spec
                if any(fnmatch.fnmatchcase(hint_low, pat.lower()) for pat in spec.match):
                    return spec

        return self.tasks["_default"]

    def vlm_provider(self) -> str:
        return str(
            self.settings.get("vlm", {}).get("provider", DEFAULT_VLM_PROVIDER)
        ).strip().lower()

    def stage_model(self, stage: str) -> str:
        vlm_cfg = self.settings.get("vlm", {})
        stage_models = vlm_cfg.get("stage_models", {})
        if stage in stage_models and stage_models[stage]:
            return str(stage_models[stage])
        prov = self.vlm_provider()
        fallback = DEFAULT_GEMINI_MODEL if prov == "gemini" else DEFAULT_OPENAI_COMPAT_MODEL
        return str(vlm_cfg.get("default_model") or fallback)

    def s1_cache_basis(
        self,
        task_id: str,
        *,
        model_override: str | None = None,
    ) -> str:
        """Isolated S1 basis: excludes labels, S3/S4/S5 prompts, and few-shots."""
        task = self.get_task(task_id)
        s1_cfg = self.settings.get("vlm", {}).get("s1_grounding", {})
        return canonical_sha256(
            {
                "stage": "s1_grounding",
                "provider": self.vlm_provider(),
                "model": model_override or self.stage_model("s1_grounding"),
                "prompt": self.fingerprints.stage_prompt_hashes["s1_grounding"],
                "task_id": task.task_id,
                "grounding_objects": [
                    g.model_dump(mode="json") for g in task.grounding_objects
                ],
                "task_prompt": task.task_prompt,
                "goal": task.goal,
                "s1_cfg": s1_cfg,
            }
        )

    def s2_cache_basis(
        self,
        task_id: str,
        *,
        model_override: str | None = None,
    ) -> str:
        """Isolated S2 basis: excludes labels, S3/S4/S5 prompts, and few-shots."""
        task = self.get_task(task_id)
        s2_cfg = self.settings.get("vlm", {}).get("s2_macro", {})
        return canonical_sha256(
            {
                "stage": "s2_macro",
                "provider": self.vlm_provider(),
                "model": model_override or self.stage_model("s2_macro"),
                "prompt": self.fingerprints.stage_prompt_hashes["s2_macro"],
                "task_id": task.task_id,
                "goal": task.goal,
                "steps": task.steps,
                "task_prompt": task.task_prompt,
                "s2_cfg": s2_cfg,
            }
        )

    def s5_cache_basis(
        self,
        task_id: str,
        *,
        model_override: str | None = None,
    ) -> str:
        """Isolated S5 basis: excludes labels, S3/S4/S5 prompts, and few-shots."""
        task = self.get_task(task_id)
        s5_cfg = self.settings.get("vlm", {}).get("s5_success", {})
        return canonical_sha256(
            {
                "stage": "s5_success",
                "provider": self.vlm_provider(),
                "model": model_override or self.stage_model("s5_success"),
                "prompt": self.fingerprints.stage_prompt_hashes["s5_success"],
                "task_id": task.task_id,
                "goal": task.goal,
                "criteria": task.criteria,
                "task_prompt": task.task_prompt,
                "s5_cfg": s5_cfg,
            }
        )

    def s3_cache_basis(
        self,
        task_id: str,
        few_shot_hash: str = "",
        *,
        model_override: str | None = None,
    ) -> str:
        task = self.get_task(task_id)
        s3_cfg = self.settings.get("vlm", {}).get("s3_discover", {})
        return canonical_sha256(
            {
                "stage": "s3_discover",
                "provider": self.vlm_provider(),
                "model": model_override or self.stage_model("s3_discover"),
                "prompt": self.fingerprints.stage_prompt_hashes["s3_discover"],
                "labels": self.fingerprints.label_taxonomy_sha256,
                "task": task.model_dump(mode="json"),
                "few_shot_hash": few_shot_hash,
                "s3_cfg": s3_cfg,
            }
        )

    def s4_cache_basis(
        self,
        task_id: str,
        few_shot_hash: str = "",
        *,
        model_override: str | None = None,
    ) -> str:
        task = self.get_task(task_id)
        s4_cfg = self.settings.get("vlm", {}).get("s4_verify", {})
        return canonical_sha256(
            {
                "stage": "s4_verify",
                "provider": self.vlm_provider(),
                "model": model_override or self.stage_model("s4_verify"),
                "prompt": self.fingerprints.stage_prompt_hashes["s4_verify"],
                "labels": self.fingerprints.label_taxonomy_sha256,
                "task": task.model_dump(mode="json"),
                "few_shot_hash": few_shot_hash,
                "s4_cfg": s4_cfg,
            }
        )

    def resolve_episode_budget_tokens(self, duration_s: float) -> int | None:
        budget_cfg = self.settings.get("vlm", {}).get("budget", {})
        if bool(budget_cfg.get("unlimited", False)):
            return None
        tps = int(budget_cfg.get("tokens_per_second", 5000))
        min_tok = int(budget_cfg.get("min_episode_tokens", 200000))
        max_tok = budget_cfg.get("max_tokens_per_episode")
        budget = max(min_tok, int(math.ceil(max(0.0, duration_s) * tps)))
        if max_tok is not None:
            budget = min(budget, int(max_tok))
        return budget
