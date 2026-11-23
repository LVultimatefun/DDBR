"""Advisory LLM-assisted Prompt/TaskSpec draft suggestion generator (never writes configs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from ..config import PROMPT_STAGE_FILES, ConfigManager
from ..vlm.client import VlmClient, normalize_stage_name
from .diagnosis import DiagnosisReport, diagnose_workspace


def _format_errors_for_prompt(
    diagnosis: DiagnosisReport | Sequence[dict[str, Any]] | dict[str, Any] | None,
) -> str:
    if diagnosis is None:
        return "(No aggregated error cases yet; review template clarity under Precision-First principles.)"
    if isinstance(diagnosis, DiagnosisReport):
        clusters = [c.model_dump(mode="json") for c in diagnosis.clusters[:10]]
        samples = [e.model_dump(mode="json") for e in diagnosis.top_errors[:10]]
        return json.dumps(
            {
                "episodes_evaluated": diagnosis.episodes_evaluated,
                "total_errors": diagnosis.total_errors,
                "counts_by_type": diagnosis.counts_by_type,
                "clusters": clusters,
                "top_errors": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(diagnosis, ensure_ascii=False, indent=2)


def suggest_prompt_updates(
    workspace_or_cfg: Path | str | ConfigManager | None = None,
    *,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    config: ConfigManager | None = None,
    cfg: ConfigManager | None = None,
    config_manager: ConfigManager | None = None,
    cfg_mgr: ConfigManager | None = None,
    config_dir: Path | str | None = None,
    configs_dir: Path | str | None = None,
    diagnosis: DiagnosisReport | Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    diagnosis_report: DiagnosisReport | Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    errors: Sequence[dict[str, Any]] | None = None,
    top_errors: Sequence[dict[str, Any]] | None = None,
    stage: str = "s3_discover",
    task_id: str | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
) -> dict[str, Any]:
    """Generate advisory Prompt/TaskSpec diff suggestions without mutating configs_override/."""
    ws_val = workspace or workspace_dir or (
        workspace_or_cfg if not isinstance(workspace_or_cfg, ConfigManager) else None
    )
    ws = Path(ws_val).resolve() if ws_val is not None else None
    raw_cfg_dir = config_dir or configs_dir

    eff_cfg = (
        config
        or cfg
        or config_manager
        or cfg_mgr
        or (workspace_or_cfg if isinstance(workspace_or_cfg, ConfigManager) else None)
        or ConfigManager(
            config_dir=Path(raw_cfg_dir) if raw_cfg_dir is not None else None,
            workspace_dir=ws,
        )
    )
    if ws is None and eff_cfg.workspace_dir is not None:
        ws = eff_cfg.workspace_dir

    norm_stage = normalize_stage_name(stage)
    if norm_stage not in PROMPT_STAGE_FILES:
        norm_stage = "s3_discover"

    tpl = getattr(eff_cfg.prompts, norm_stage)
    tpl_yaml = yaml.safe_dump(tpl.model_dump(mode="json"), sort_keys=False, allow_unicode=True)

    err_source = (
        diagnosis
        if diagnosis is not None
        else (
            diagnosis_report
            if diagnosis_report is not None
            else (errors if errors is not None else top_errors)
        )
    )
    if err_source is None and ws is not None and (ws / "robo_vlm_qc.db").is_file():
        err_source = diagnose_workspace(ws, top_n=10)
    err_text = _format_errors_for_prompt(err_source)

    task_ctx = ""
    if task_id:
        spec = eff_cfg.get_task(task_id)
        task_ctx = f"\n[Target TaskSpec ({spec.task_id})]: goal={spec.goal}\n"

    system_prompt = (
        "You are the VLM Lab Prompt revision advisor (prompt_assist). Based on the current stage template "
        "and aggregated error diagnosis, propose minimal, targeted, actionable prompt revisions adhering to "
        "Precision-First principles (advisory draft only).\n"
        'Output strict JSON with fields "suggested_prompt_diff", "rationale", '
        '"proposed_user_template_patch", "proposed_system_template_patch", and "task_spec_notes".'
    )
    user_prompt = (
        f"Stage: {norm_stage} (template file: {norm_stage}.yaml){task_ctx}\n"
        f"[Current Template]:\n{tpl_yaml}\n"
        f"[Aggregated Error Diagnosis Report]:\n{err_text}\n"
        "Provide concrete prompt revision suggestions (suggested_prompt_diff)."
    )

    owns_client = client is None and vlm_client is None
    eff_client = client or vlm_client or VlmClient.from_config(eff_cfg, workspace_dir=ws)
    try:
        resp = eff_client.chat_json(
            stage="prompt_assist",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking=False,
            max_tokens=1536,
        )
    finally:
        if owns_client:
            eff_client.close()

    result = dict(resp)
    result.setdefault("stage", norm_stage)
    result.setdefault("advisory_only", True)
    result.setdefault("base_version", tpl.version)
    return result


generate_prompt_suggestion = suggest_prompt_updates
suggest_draft = suggest_prompt_updates
assist_prompt = suggest_prompt_updates
suggest_updates = suggest_prompt_updates
generate_suggestions = suggest_prompt_updates
