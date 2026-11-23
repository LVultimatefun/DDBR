"""Core E2E feature coverage: causal VLM DAG, light guardrails, and 1080p Studio HUD MP4.

Covers:
- 11 pure-visual labels across 7 categories, orthogonal splice_policy decoupling,
  zero rule/action operators, and lookup/validation.
- Multi-provider VlmClient over MockVlmServer (OpenAI defaults, httpx transport,
  [t=01.20s] prompt tags, stage routing, DashScope thinking_mode vs OpenAI-compat
  response_format handling, 429/500/JSON retries, dynamic token budgets) & SHA256 StageCache.
- Camera-role-adaptive PromptBuilder + S2 backbone / Few-Shot injection, Stage 1 grounding
  [0,1000] + static container stabilization, and Stage 2 macro stages + L/R arm timelines.
- Stage 5 dense-tail success + 30% anti-FP control check, Stage 3 Precision-First discovery +
  S2 protection + Few-Shot injection, Stage 4A merge + Stage 4B CoV refinement, and full VlmDag.
- Light guardrails: splice_gate (boundary_idle, stationary cut_clean, moving-arm >2cm downgrade,
  bimanual >3cm downgrade), hallucination (>15cm motion override), and visual_gate (pure video).
- 1920x1080@30fps Studio HUD MP4 renderer, `run-vlm` & `render-mp4` CLIs, ResolvedEpisode
  human-edit reflection, and adaptive camera layouts.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from tests.e2e import helpers as h

# ---------------------------------------------------------------------------
# Lazy Contract Invocation Helpers
# ---------------------------------------------------------------------------


def _call_with_accepted_kwargs(fn: Any, **candidates: Any) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        return fn(**{k: v for k, v in candidates.items() if v is not None or k in params})
    kwargs = {k: v for k, v in candidates.items() if k in params}
    return fn(**kwargs)


def _build_vlm_client(cfg_mgr: Any) -> Any:
    mod = importlib.import_module("robo_vlm_qc.vlm.client")
    cls = getattr(mod, "VlmClient")
    if hasattr(cls, "from_config"):
        return _call_with_accepted_kwargs(
            cls.from_config, config=cfg_mgr, cfg=cfg_mgr, config_manager=cfg_mgr
        )
    vlm_cfg = cfg_mgr.settings.get("vlm", {})
    sig = inspect.signature(cls.__init__)
    if len(sig.parameters) <= 2 and any(
        p in sig.parameters for p in ("config", "cfg", "config_manager", "settings")
    ):
        return _call_with_accepted_kwargs(
            cls,
            config=cfg_mgr,
            cfg=cfg_mgr,
            config_manager=cfg_mgr,
            settings=vlm_cfg,
        )
    return _call_with_accepted_kwargs(
        cls,
        config=cfg_mgr,
        cfg=cfg_mgr,
        config_manager=cfg_mgr,
        settings=vlm_cfg,
        provider=vlm_cfg.get("provider", "openai_compat"),
        base_url=vlm_cfg.get("base_url"),
        api_key_env=vlm_cfg.get("api_key_env", "MOCK_VLM_API_KEY"),
        thinking_mode=vlm_cfg.get("thinking_mode", "auto"),
        timeout_s=vlm_cfg.get("timeout_s", 30.0),
        retries=vlm_cfg.get("retries", 3),
        retry_min_wait_s=vlm_cfg.get("retry_min_wait_s", 0.01),
        retry_max_wait_s=vlm_cfg.get("retry_max_wait_s", 0.05),
        stage_models=vlm_cfg.get("stage_models", {}),
    )


def _build_stage_cache(workspace: Path) -> Any:
    mod = importlib.import_module("robo_vlm_qc.vlm.cache")
    cls = getattr(mod, "StageCache")
    sig = inspect.signature(cls.__init__)
    if "cache_dir" in sig.parameters:
        return cls(cache_dir=workspace / "stage_cache")
    if "workspace" in sig.parameters:
        return cls(workspace=workspace)
    if "workspace_dir" in sig.parameters:
        return cls(workspace_dir=workspace)
    return cls(workspace)


def _build_vlm_dag(workspace: Path, cfg_mgr: Any) -> Any:
    mod = importlib.import_module("robo_vlm_qc.vlm.dag")
    cls = getattr(mod, "VlmDag")
    repo_mod = importlib.import_module("robo_vlm_qc.store")
    repo = repo_mod.Repository(workspace)
    client = _build_vlm_client(cfg_mgr)
    cache = _build_stage_cache(workspace)
    return _call_with_accepted_kwargs(
        cls,
        workspace=workspace,
        workspace_dir=workspace,
        config=cfg_mgr,
        cfg=cfg_mgr,
        config_manager=cfg_mgr,
        client=client,
        vlm_client=client,
        cache=cache,
        stage_cache=cache,
        repo=repo,
        repository=repo,
    )


def _run_dag_episode(
    dag: Any,
    view: Any,
    *,
    task_spec: Any = None,
    few_shots: list[Any] | None = None,
) -> Any:
    fn = (
        getattr(dag, "run_episode", None)
        or getattr(dag, "run", None)
        or getattr(dag, "execute", None)
    )
    assert fn is not None, "VlmDag must expose run_episode / run / execute"
    return _call_with_accepted_kwargs(
        fn,
        view=view,
        episode_view=view,
        task_spec=task_spec,
        task=task_spec,
        few_shots=few_shots or [],
        few_shot_items=few_shots or [],
    )


def _ingest_one(workspace: Path, cfg_dir: Path, data_path: Path) -> str:
    res = h.run_cli(
        [
            "-w",
            str(workspace),
            "-c",
            str(cfg_dir),
            "ingest",
            "--data",
            str(data_path),
            "--json",
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    payload = json.loads(res.stdout)
    assert payload["episodes"], payload
    return str(payload["episodes"][0]["episode_id"])


def _make_kinematics_track(
    duration_s: float = 4.0,
    fps: float = 30.0,
    *,
    left_disp_fn: Any = None,
    right_disp_fn: Any = None,
) -> Any:
    from robo_vlm_qc.io.base import KinematicsData

    T = max(2, int(round(duration_s * fps)))
    ts = np.arange(T, dtype=np.float64) / fps
    eef_pos = np.zeros((T, 2, 3), dtype=np.float64)
    eef_pos[:, 0, :] = [0.3, 0.2, 0.4]
    eef_pos[:, 1, :] = [0.3, -0.2, 0.4]
    if left_disp_fn is not None:
        for i, t in enumerate(ts):
            eef_pos[i, 0, 0] += float(left_disp_fn(float(t)))
    if right_disp_fn is not None:
        for i, t in enumerate(ts):
            eef_pos[i, 1, 0] += float(right_disp_fn(float(t)))
    eef_rotvec = np.zeros((T, 2, 3), dtype=np.float64)
    hand_q = np.zeros((T, 44), dtype=np.float64)
    torso_q = np.zeros((T, 4), dtype=np.float64)
    chassis_pose = np.zeros((T, 3), dtype=np.float64)
    valid_mask = np.ones(T, dtype=bool)
    return KinematicsData(
        timestamps_s=ts,
        eef_pos=eef_pos,
        eef_rotvec=eef_rotvec,
        hand_q=hand_q,
        torso_q=torso_q,
        chassis_pose=chassis_pose,
        valid_mask=valid_mask,
    )


def _invoke_guardrail_module(
    mod_path: str,
    candidate_names: tuple[str, ...],
    *,
    segments: list[Any],
    kinematics: Any = None,
    media: Any = None,
    cfg: Any = None,
    duration_s: float = 4.0,
) -> list[Any]:
    mod = importlib.import_module(mod_path)
    target = None
    for nm in candidate_names:
        if hasattr(mod, nm):
            target = getattr(mod, nm)
            break
    assert target is not None, f"None of {candidate_names} found in {mod_path}"
    if inspect.isclass(target):
        inst = _call_with_accepted_kwargs(target, cfg=cfg, config=cfg, settings=cfg)
        fn = (
            getattr(inst, "apply", None)
            or getattr(inst, "check", None)
            or getattr(inst, "verify", None)
            or inst
        )
    else:
        fn = target
    return _call_with_accepted_kwargs(
        fn,
        segments=segments,
        qc_segments=segments,
        kinematics=kinematics,
        kin=kinematics,
        media=media,
        cfg=cfg,
        config=cfg,
        settings=cfg,
        duration_s=duration_s,
    )


def _render_viz(
    resolved: Any, view: Any, out_mp4: Path, *, cfg: Any = None
) -> Path:
    mod = importlib.import_module("robo_vlm_qc.export.viz_mp4")
    fn = (
        getattr(mod, "render_viz_mp4", None)
        or getattr(mod, "render_episode_mp4", None)
        or getattr(mod, "render_studio_hud_mp4", None)
    )
    if fn is None and hasattr(mod, "VizMp4Exporter"):
        exp_cls = getattr(mod, "VizMp4Exporter")
        inst = _call_with_accepted_kwargs(exp_cls, cfg=cfg, config=cfg)
        fn = getattr(inst, "render_episode", None) or getattr(inst, "render", None)
    assert fn is not None, "robo_vlm_qc.export.viz_mp4 must expose render_viz_mp4 or VizMp4Exporter"
    res = _call_with_accepted_kwargs(
        fn,
        resolved=resolved,
        episode=resolved,
        view=view,
        episode_view=view,
        out_path=out_mp4,
        output_path=out_mp4,
        mp4_path=out_mp4,
        cfg=cfg,
        config=cfg,
    )
    return Path(res) if isinstance(res, (str, Path)) else out_mp4


# ===========================================================================
# Pure-Visual 11-Label Taxonomy (configs/labels.yaml)
# ===========================================================================


def test_labels_yaml_has_11_core_visual_labels_and_7_categories(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir)
    expected_categories = [
        "idle",
        "failure_recovery",
        "inefficiency",
        "failure",
        "necessary_skill",
        "terminal",
        "unknown",
    ]
    assert list(cfg.labels.categories) == expected_categories
    expected_ids = {
        "boundary_idle",
        "mid_idle_freeze",
        "missed_grasp_retry",
        "object_slip_recovery",
        "placement_redo",
        "hesitation_oscillation",
        "collision_or_knock",
        "precision_fine_adjust",
        "bimanual_assist_hold",
        "terminal_failure",
        "unknown_anomaly",
    }
    assert {lbl.id for lbl in cfg.labels.labels} == expected_ids
    assert all(lbl.status == "active" for lbl in cfg.labels.labels)


def test_labels_zero_rule_or_action_operators(e2e_config_dir: Path) -> None:
    raw = yaml.safe_load((e2e_config_dir / "labels.yaml").read_text(encoding="utf-8"))
    h.assert_no_legacy_fields(raw)
    forbidden_rule_keys = {
        "action_operator",
        "when",
        "conditions",
        "rule_id",
        "physical_operator",
        "threshold",
    }
    for lbl in raw["labels"]:
        assert not (forbidden_rule_keys & set(lbl.keys())), (
            f"Rule operator leaked into pure-visual label {lbl['id']}"
        )


def test_labels_orthogonal_decoupling_from_splice_policy(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    by_id = {lbl.id: lbl for lbl in cfg.labels.labels}

    assert by_id["boundary_idle"].default_splice_policy == "cut_clean"
    assert by_id["mid_idle_freeze"].default_splice_policy == "cut_clean"
    assert by_id["precision_fine_adjust"].default_splice_policy == "keep_positive"
    assert by_id["bimanual_assist_hold"].default_splice_policy == "keep_positive"
    for rec_id in ("missed_grasp_retry", "object_slip_recovery", "placement_redo"):
        assert by_id[rec_id].default_training_role == "recovery"
        assert by_id[rec_id].default_splice_policy == "mask_loss"

    # Verify runtime orthogonal decoupling: mid_idle_freeze can be downgraded to mask_loss
    seg = QcSegment(
        seg_id="decoupled_1",
        t_start=1.0,
        t_end=2.0,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="mask_loss",
        mask_side="both",
        vlm_raw_policy="cut_clean",
    )
    assert seg.label_id == "mid_idle_freeze" and seg.splice_policy == "mask_loss"


def test_labels_collector_ui_and_vlm_semantics_populated(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir)
    for lbl in cfg.labels.labels:
        assert lbl.name.strip()
        assert lbl.collector_ui.title.strip()
        assert lbl.collector_ui.hint.strip()
        assert lbl.vlm_semantics.definition.strip()
        assert len(lbl.vlm_semantics.visual_cues) >= 1


def test_labels_lookup_and_legacy_rejection(e2e_config_dir: Path) -> None:
    from pydantic import ValidationError

    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.taxonomy import LabelRegistry

    cfg = ConfigManager(config_dir=e2e_config_dir)
    lbl = cfg.labels.get("object_slip_recovery")
    assert lbl is not None and lbl.category == "failure_recovery"
    assert cfg.labels.get("non_existent_label") is None

    with pytest.raises(ValidationError):
        LabelRegistry.model_validate(
            {
                "categories": ["idle"],
                "labels": [
                    {
                        **lbl.model_dump(),
                        "id": "long_pause",
                    }
                ],
            }
        )


# ===========================================================================
# Multi-Provider VlmClient & Content-Addressed StageCache
# ===========================================================================


def test_vlm_client_httpx_transport_and_json_response(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    try:
        chat_fn = (
            getattr(client, "chat_json", None)
            or getattr(client, "complete_json", None)
            or getattr(client, "call_stage", None)
        )
        assert chat_fn is not None, "VlmClient must expose chat_json / complete_json / call_stage"

        dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
        resp = _call_with_accepted_kwargs(
            chat_fn,
            stage="s1_grounding",
            model="gpt-4o",
            system_prompt="You are an embodied manipulation visual grounding expert (2D Grounding).",
            user_prompt="Target objects to ground:\n- target_object: Target Object",
            images=[dummy_img],
            timestamps_s=[1.2],
        )
        assert isinstance(resp, dict)
        assert len(mock_vlm_server.requests) == 1
        req = mock_vlm_server.requests[0]
        assert req["headers"].get("authorization") == "Bearer mock-e2e-key"
        assert req["stage"] == "S1"
        user_content = req["messages"][-1]["content"]
        text_blocks = [
            b["text"]
            for b in user_content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert any("[t=01.20s]" in blk for blk in text_blocks), (
            f"Expected English timestamp tag '[t=01.20s]', got {text_blocks}"
        )
    finally:
        if hasattr(client, "close"):
            client.close()


def test_vlm_client_stage_model_routing(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter

    h.write_workspace_config(
        e2e_workspace,
        base_url=mock_vlm_server.base_url,
        stage_models={
            "s1_grounding": "gpt-4o-mini",
            "s2_macro": "gpt-4o-mini",
            "s3_discover": "gpt-4o",
            "s4_verify": "gpt-4o",
            "s5_success": "gpt-4o-mini",
        },
    )
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_route", duration_s=4.0)
    _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        _run_dag_episode(dag, view, task_spec=cfg.get_task("pick_and_place"))

    models_by_stage = {r["stage"]: r["model"] for r in mock_vlm_server.requests}
    assert models_by_stage.get("S1") == "gpt-4o-mini"
    assert models_by_stage.get("S2") == "gpt-4o-mini"
    assert models_by_stage.get("S5") == "gpt-4o-mini"
    assert models_by_stage.get("S3") == "gpt-4o"


def test_vlm_client_thinking_mode_openai_compat_vs_dashscope(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.vlm.client import VlmClient

    # Verify OpenAI-compatible open-source defaults
    default_client = VlmClient()
    try:
        assert default_client.base_url == "https://api.openai.com/v1"
        assert default_client.api_key_env == "OPENAI_API_KEY"
        assert default_client.key_file == ".env"
        assert default_client.default_model == "gpt-4o"
        assert default_client.thinking_mode == "auto"
    finally:
        default_client.close()

    # 1. Standard OpenAI-compatible endpoint (non-DashScope base_url, thinking_mode="auto"):
    #    must omit DashScope-only fields and keep response_format={"type": "json_object"}
    cfg_std = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client_std = _build_vlm_client(cfg_std)
    try:
        chat_std = getattr(client_std, "chat_json")
        _call_with_accepted_kwargs(
            chat_std,
            stage="s1_grounding",
            model="gpt-4o",
            system_prompt="2D Grounding",
            user_prompt="Target objects to ground:\n- target_object: Target Object",
            thinking=True,
            thinking_budget=768,
            max_tokens=800,
        )
    finally:
        client_std.close()

    std_body = mock_vlm_server.requests[-1]["body"]
    assert int(std_body["max_tokens"]) == 800
    assert "enable_thinking" not in std_body
    assert "thinking_budget" not in std_body
    assert std_body.get("response_format") == {"type": "json_object"}

    # 2. Explicit DashScope thinking mode (thinking_mode="dashscope", thinking=True):
    #    sends enable_thinking/thinking_budget, adds budget to max_tokens, omits response_format
    h.write_workspace_config(
        e2e_workspace,
        extra_overrides={"vlm": {"thinking_mode": "dashscope"}},
    )
    cfg_ds = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client_ds = _build_vlm_client(cfg_ds)
    try:
        chat_ds = getattr(client_ds, "chat_json")
        _call_with_accepted_kwargs(
            chat_ds,
            stage="s1_grounding",
            model="qwen-vl-max",
            system_prompt="2D Grounding",
            user_prompt="Target objects to ground:\n- target_object: Target Object",
            thinking=True,
            thinking_budget=768,
            max_tokens=800,
        )
    finally:
        client_ds.close()

    ds_body = mock_vlm_server.requests[-1]["body"]
    assert int(ds_body["max_tokens"]) == 768 + 800
    assert ds_body.get("enable_thinking") is True
    assert ds_body.get("thinking_budget") == 768
    assert "response_format" not in ds_body


def test_vlm_client_transient_429_500_and_json_retries(
    e2e_workspace: Path, e2e_config_dir: Path, mock_vlm_server: Any
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    try:
        chat_fn = (
            getattr(client, "chat_json", None)
            or getattr(client, "complete_json", None)
            or getattr(client, "call_stage", None)
        )

        mock_vlm_server.fail_next(429, count=1, stage="S1")
        mock_vlm_server.fail_next(500, count=1, stage="S1")
        mock_vlm_server.fail_with_malformed_json(count=1, stage="S1")

        res = _call_with_accepted_kwargs(
            chat_fn,
            stage="s1_grounding",
            model="gpt-4o",
            system_prompt="2D Grounding",
            user_prompt="Target objects to ground:\n- target_object: Target Object",
        )
        assert isinstance(res, dict) and "objects" in res
        assert len(mock_vlm_server.requests) == 4
    finally:
        if hasattr(client, "close"):
            client.close()


def test_vlm_client_dynamic_duration_token_budget(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    assert cfg.resolve_episode_budget_tokens(4.0) == 200000
    assert cfg.resolve_episode_budget_tokens(120.0) == 600000

    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_budget", duration_s=4.0)
    _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        art = _run_dag_episode(dag, view, task_spec=cfg.get_task("pick_and_place"))
    assert art.usage.total_tokens > 0
    assert len(art.usage.stages) >= 4


def test_stage_cache_sha256_layout_and_s1_s2_s5_isolation(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter

    # Direct StageCache contract check: stage_cache/<sha256[:2]>/<sha256>.json
    cache = _build_stage_cache(e2e_workspace)
    test_key = "a1b2c3d4" + "0" * 56
    cache.put(test_key, {"cached": True, "stage": "s1"})
    expected_file = e2e_workspace / "stage_cache" / "a1" / f"{test_key}.json"
    assert expected_file.is_file()
    assert cache.get(test_key) == {"cached": True, "stage": "s1"}

    # Full DAG run -> cold cache
    cfg1 = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_cache", duration_s=4.0)
    _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        art1 = _run_dag_episode(
            _build_vlm_dag(e2e_workspace, cfg1), view, task_spec=cfg1.get_task("pick_and_place")
        )
    assert len(mock_vlm_server.requests) >= 5
    assert art1.stage_refs.s1_grounding and art1.stage_refs.s2_macro and art1.stage_refs.s5_success

    # Modify labels.yaml + s3_discover.yaml -> S1, S2, S5 MUST hit cache (0 HTTP calls)
    labels_raw = yaml.safe_load((e2e_config_dir / "labels.yaml").read_text("utf-8"))
    labels_raw["version"] = 99
    (e2e_workspace / "configs_override" / "labels.yaml").write_text(
        yaml.safe_dump(labels_raw, allow_unicode=True), encoding="utf-8"
    )
    ov_prompts = e2e_workspace / "configs_override" / "prompts"
    ov_prompts.mkdir(parents=True, exist_ok=True)
    s3_raw = yaml.safe_load((e2e_config_dir / "prompts" / "s3_discover.yaml").read_text("utf-8"))
    s3_raw["version"] = "2.0.0"
    (ov_prompts / "s3_discover.yaml").write_text(
        yaml.safe_dump(s3_raw, allow_unicode=True), encoding="utf-8"
    )

    mock_vlm_server.requests.clear()
    cfg2 = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        art2 = _run_dag_episode(
            _build_vlm_dag(e2e_workspace, cfg2), view, task_spec=cfg2.get_task("pick_and_place")
        )

    rerun_stages = {r["stage"] for r in mock_vlm_server.requests}
    assert "S1" not in rerun_stages
    assert "S2" not in rerun_stages
    assert "S5" not in rerun_stages
    assert "S5_CONTROL" not in rerun_stages
    assert "S3" in rerun_stages
    assert art2.stage_refs.s1_grounding == art1.stage_refs.s1_grounding
    assert art2.stage_refs.s2_macro == art1.stage_refs.s2_macro
    assert art2.stage_refs.s5_success == art1.stage_refs.s5_success
    assert art2.stage_refs.s3_discover != art1.stage_refs.s3_discover


# ===========================================================================
# PromptBuilder, Stage 1 Grounding & Stage 2 Macro
# ===========================================================================


def test_prompt_builder_camera_role_adaptive_descriptions(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager

    cfg = ConfigManager(config_dir=e2e_config_dir)
    pb_mod = importlib.import_module("robo_vlm_qc.vlm.prompt_builder")
    builder_cls = getattr(pb_mod, "PromptBuilder", None)
    render_fn = getattr(pb_mod, "describe_camera_layout", None) or getattr(
        pb_mod, "build_camera_layout_desc", None
    )

    wrist_roles = {"main": "main", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
    torso_roles = {"main": "head_rgbd", "front": "torso_rgbd"}
    single_roles = {"main": "main"}

    if builder_cls is not None:
        pb = _call_with_accepted_kwargs(
            builder_cls, config=cfg, cfg=cfg, prompts=cfg.prompts, labels=cfg.labels
        )
        s2_fn = getattr(pb, "build_s2", None) or getattr(pb, "render_s2", None)
        assert s2_fn is not None
        task = cfg.get_task("pick_and_place")
        _, u_wrist = _call_with_accepted_kwargs(
            s2_fn, task=task, task_spec=task, camera_roles=wrist_roles, duration_s=10.0
        )
        _, u_torso = _call_with_accepted_kwargs(
            s2_fn, task=task, task_spec=task, camera_roles=torso_roles, duration_s=10.0
        )
        _, u_single = _call_with_accepted_kwargs(
            s2_fn, task=task, task_spec=task, camera_roles=single_roles, duration_s=10.0
        )
        assert u_wrist != u_torso and u_torso != u_single
        assert "WRIST" in u_wrist.upper()
        assert "TORSO" in u_torso.upper()
    else:
        assert render_fn is not None
        assert render_fn(wrist_roles) != render_fn(torso_roles) != render_fn(single_roles)


def test_prompt_builder_s2_backbone_and_few_shot_injection(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import MacroStage
    from robo_vlm_qc.models.taxonomy import FewShotItem

    cfg = ConfigManager(config_dir=e2e_config_dir)
    pb_mod = importlib.import_module("robo_vlm_qc.vlm.prompt_builder")
    pb = _call_with_accepted_kwargs(
        getattr(pb_mod, "PromptBuilder"),
        config=cfg,
        cfg=cfg,
        prompts=cfg.prompts,
        labels=cfg.labels,
    )
    s3_fn = getattr(pb, "build_s3", None) or getattr(pb, "render_s3", None)
    assert s3_fn is not None

    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Steadily grasp target object")]
    fs = FewShotItem(
        example_id="fs_demo",
        scope="global",
        error_type="fp_overflag",
        episode_id="pick_ds/ep_0",
        t_start=1.0,
        t_end=2.0,
        vlm_wrong_json={"label_id": "hesitation_oscillation"},
        human_right_json={"splice_policy": "keep_positive"},
        rationale="Slow precision alignment over the target bin is normal behavior",
        created_at=1700000000.0,
    )
    sys_p, usr_p = _call_with_accepted_kwargs(
        s3_fn,
        task=cfg.get_task("pick_and_place"),
        task_spec=cfg.get_task("pick_and_place"),
        camera_roles={"main": "head_rgbd", "front": "torso_rgbd"},
        macro_stages=stages,
        few_shots=[fs],
        few_shot_items=[fs],
        window_start_s=0.0,
        window_end_s=4.0,
        duration_s=4.0,
    )
    assert "Precision-First" in sys_p
    assert "Steadily grasp target object" in usr_p
    assert "Slow precision alignment over the target bin is normal behavior" in usr_p


def test_stage1_grounding_normalized_boxes_and_keyframes(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import GroundingResult

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s1", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s1_mod = importlib.import_module("robo_vlm_qc.vlm.stage1_grounding")
    s1_fn = getattr(s1_mod, "run_stage1_grounding", None) or getattr(s1_mod, "run_s1", None)
    assert s1_fn is not None

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s1_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    grounding = res[0] if isinstance(res, tuple) else res
    assert isinstance(grounding, GroundingResult)
    assert "target_object" in grounding.object_tracks_2d
    for kf in grounding.object_tracks_2d["target_object"].keyframes:
        if kf.visible:
            x1, y1, x2, y2 = kf.bbox_norm_1000
            assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def test_stage1_grounding_static_container_median_stabilization(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter

    # Inject small jitter across calls for static container "target_bin"
    jitter_boxes = [
        [400, 200, 620, 520],
        [408, 205, 626, 524],
        [394, 198, 616, 518],
        [402, 201, 621, 521],
        [404, 202, 622, 522],
        [399, 199, 619, 519],
    ]
    call_counter = {"n": 0}

    def _s1_jitter(_req: dict[str, Any]) -> dict[str, Any]:
        idx = call_counter["n"] % len(jitter_boxes)
        call_counter["n"] += 1
        return {
            "objects": {
                "target_object": {
                    "visible": True,
                    "bbox": [100 + idx * 10, 300, 220 + idx * 10, 440],
                    "state": "moving",
                    "status_kind": "neutral",
                },
                "target_bin": {
                    "visible": True,
                    "bbox": jitter_boxes[idx],
                    "state": "stationary",
                    "status_kind": "neutral",
                },
            }
        }

    mock_vlm_server.set_stage_response("s1", _s1_jitter)
    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_stab", duration_s=12.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s1_mod = importlib.import_module("robo_vlm_qc.vlm.stage1_grounding")
    s1_fn = getattr(s1_mod, "run_stage1_grounding", None) or getattr(s1_mod, "run_s1", None)

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s1_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    grounding = res[0] if isinstance(res, tuple) else res
    bin_track = grounding.object_tracks_2d["target_bin"]
    assert bin_track.is_static_container is True
    vis_boxes = [tuple(k.bbox_norm_1000) for k in bin_track.keyframes if k.visible]
    assert len(vis_boxes) >= 5
    assert len(set(vis_boxes)) == 1, f"Static container boxes should stabilize to median, got {vis_boxes}"


def test_stage2_macro_stages_cover_duration_and_colors(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s2", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s2_mod = importlib.import_module("robo_vlm_qc.vlm.stage2_macro")
    s2_fn = getattr(s2_mod, "run_stage2_macro", None) or getattr(s2_mod, "run_s2", None)
    assert s2_fn is not None

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s2_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    macro_stages = (
        res[0]
        if isinstance(res, tuple)
        else res["macro_stages"]
        if isinstance(res, dict)
        else res.macro_stages
    )
    assert len(macro_stages) >= 2
    assert all(isinstance(st, MacroStage) for st in macro_stages)
    assert macro_stages[0].t_start == pytest.approx(0.0, abs=0.2)
    assert macro_stages[-1].t_end == pytest.approx(view.duration_s, abs=0.2)


def test_stage2_arm_timelines_left_and_right(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import ArmTimelines

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_arms", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s2_mod = importlib.import_module("robo_vlm_qc.vlm.stage2_macro")
    s2_fn = getattr(s2_mod, "run_stage2_macro", None) or getattr(s2_mod, "run_s2", None)

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s2_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    arm_tl = (
        res[1]
        if isinstance(res, tuple)
        else res["arm_timelines"]
        if isinstance(res, dict)
        else res.arm_timelines
    )
    assert isinstance(arm_tl, ArmTimelines)
    assert len(arm_tl.left) >= 1 and len(arm_tl.right) >= 1
    assert all(
        sp.state in {"idle", "work", "hold", "anom", "mask", "good"}
        for sp in arm_tl.left + arm_tl.right
    )


# ===========================================================================
# Stage 5 Success, Stage 3 Discover, Stage 4A/4B & Causal DAG
# ===========================================================================


def test_stage5_success_dense_tail_and_criteria(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import SuccessVerdict

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s5", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s5_mod = importlib.import_module("robo_vlm_qc.vlm.stage5_success")
    s5_fn = getattr(s5_mod, "run_stage5_success", None) or getattr(s5_mod, "run_s5", None)
    assert s5_fn is not None

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s5_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    verdict = res[0] if isinstance(res, tuple) else res
    assert isinstance(verdict, SuccessVerdict)
    assert verdict.passed is True and verdict.verdict == "success"
    assert len(verdict.criteria) >= 1
    assert verdict.criteria[0].met is True and verdict.criteria[0].satisfied_at_s is not None


def test_stage5_anti_fp_30pct_control_check(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s5_ctrl", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s5_mod = importlib.import_module("robo_vlm_qc.vlm.stage5_success")
    s5_fn = getattr(s5_mod, "run_stage5_success", None) or getattr(s5_mod, "run_s5", None)

    # First run: default mock S5_CONTROL returns failure -> control check passes
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res_ok = _call_with_accepted_kwargs(
            s5_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    v_ok = res_ok[0] if isinstance(res_ok, tuple) else res_ok
    assert v_ok.control_check is not None
    assert v_ok.control_check.cut_ratio == pytest.approx(0.30, abs=0.05)
    assert v_ok.control_check.passed is True
    assert any(r["stage"] == "S5_CONTROL" for r in mock_vlm_server.requests)

    # Second run: S5_CONTROL falsely claims success on 30% prefix -> anti-FP gate triggers
    mock_vlm_server.set_stage_response(
        "s5_control",
        {
            "verdict": "success",
            "confidence": 0.9,
            "criteria": [
                {"criterion": "done", "met": True, "satisfied_at_s": 1.0, "evidence": "fp"}
            ],
            "summary": "false positive",
        },
    )
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res_fp = _call_with_accepted_kwargs(
            s5_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    v_fp = res_fp[0] if isinstance(res_fp, tuple) else res_fp
    assert v_fp.control_check is not None and v_fp.control_check.passed is False


def test_stage3_discover_sliding_window_and_precision_first(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s3", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s3_mod = importlib.import_module("robo_vlm_qc.vlm.stage3_discover")
    s3_fn = getattr(s3_mod, "run_stage3_discover", None) or getattr(s3_mod, "run_s3", None)
    assert s3_fn is not None

    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Pick and place object")]
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s3_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            macro_stages=stages,
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    segs = res[0] if isinstance(res, tuple) else res
    assert isinstance(segs, list) and len(segs) >= 1
    h.assert_no_legacy_fields([s.model_dump() if hasattr(s, "model_dump") else s for s in segs])
    s3_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S3"]
    assert len(s3_reqs) >= 1


def test_stage3_discover_injects_few_shot_examples(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage
    from robo_vlm_qc.models.taxonomy import FewShotItem

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s3_fs", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s3_mod = importlib.import_module("robo_vlm_qc.vlm.stage3_discover")
    s3_fn = getattr(s3_mod, "run_stage3_discover", None) or getattr(s3_mod, "run_s3", None)

    fs = FewShotItem(
        example_id="fs_s3",
        scope="task:pick_and_place",
        task_id="pick_and_place",
        label_id="precision_fine_adjust",
        error_type="fp_overflag",
        episode_id="pick_ds/ep_0",
        t_start=1.0,
        t_end=2.0,
        vlm_wrong_json={"label_id": "hesitation_oscillation"},
        human_right_json={"splice_policy": "keep_positive"},
        rationale="Fine alignment above the target bin must not be mislabeled as hesitation",
        created_at=1700000000.0,
    )
    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Pick and place object")]
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        _call_with_accepted_kwargs(
            s3_fn,
            view=view,
            episode_view=view,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            macro_stages=stages,
            few_shots=[fs],
            few_shot_items=[fs],
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    s3_reqs = [r for r in mock_vlm_server.requests if r["stage"] == "S3"]
    assert s3_reqs
    msg_blob = json.dumps(s3_reqs[-1]["messages"], ensure_ascii=False)
    assert "Fine alignment above the target bin must not be mislabeled as hesitation" in msg_blob


def test_stage4a_global_merge_dedup_keeps_distinct_retries(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s4a", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s4_mod = importlib.import_module("robo_vlm_qc.vlm.stage4_verify")
    s4a_fn = (
        getattr(s4_mod, "run_stage4a_merge", None)
        or getattr(s4_mod, "merge_candidates", None)
        or getattr(s4_mod, "run_stage4_verify", None)
    )
    assert s4a_fn is not None

    # Mock S4A merging overlapping [1.0,1.8] & [1.1,1.85] into 1 while keeping distinct [3.0,3.6]
    mock_vlm_server.set_stage_response(
        "s4a",
        {
            "merged_segments": [
                {
                    "t_start": 1.0,
                    "t_end": 1.85,
                    "label_id": "missed_grasp_retry",
                    "affected_arm": "left",
                    "splice_policy": "mask_loss",
                    "confidence": 0.92,
                },
                {
                    "t_start": 3.0,
                    "t_end": 3.6,
                    "label_id": "missed_grasp_retry",
                    "affected_arm": "left",
                    "splice_policy": "mask_loss",
                    "confidence": 0.89,
                },
            ]
        },
    )
    raw_cands = [
        QcSegment(
            seg_id="c1",
            t_start=1.0,
            t_end=1.8,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        ),
        QcSegment(
            seg_id="c2",
            t_start=1.1,
            t_end=1.85,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        ),
        QcSegment(
            seg_id="c3",
            t_start=3.0,
            t_end=3.6,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        ),
    ]
    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Pick and place object")]
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s4a_fn,
            view=view,
            episode_view=view,
            candidates=raw_cands,
            segments=raw_cands,
            raw_segments=raw_cands,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            macro_stages=stages,
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    merged = res[0] if isinstance(res, tuple) else res
    assert len(merged) == 2


def test_stage4b_dense_start_vs_end_cov_refinement(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    client = _build_vlm_client(cfg)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_s4b", duration_s=4.0)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    s4_mod = importlib.import_module("robo_vlm_qc.vlm.stage4_verify")
    s4_fn = (
        getattr(s4_mod, "run_stage4_verify", None)
        or getattr(s4_mod, "run_stage4b_verify", None)
        or getattr(s4_mod, "run_s4", None)
    )
    assert s4_fn is not None

    cands = [
        QcSegment(
            seg_id="c_cov",
            t_start=1.0,
            t_end=1.8,
            label_id="missed_grasp_retry",
            training_role="recovery",
            splice_policy="mask_loss",
            mask_side="left",
            vlm_raw_policy="mask_loss",
        )
    ]
    stages = [MacroStage(stage_id="S1", t_start=0.0, t_end=4.0, label="Pick and place object")]
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        res = _call_with_accepted_kwargs(
            s4_fn,
            view=view,
            episode_view=view,
            candidates=cands,
            segments=cands,
            raw_segments=cands,
            task=cfg.get_task("pick_and_place"),
            task_spec=cfg.get_task("pick_and_place"),
            macro_stages=stages,
            client=client,
            vlm_client=client,
            cfg=cfg,
            config=cfg,
        )
    verified = res[0] if isinstance(res, tuple) else res
    assert len(verified) >= 1
    seg0 = verified[0]
    assert seg0.visual_evidence.initial_state
    assert seg0.visual_evidence.process_change
    assert seg0.visual_evidence.end_state
    assert any(r["stage"] == "S4B" for r in mock_vlm_server.requests)


def test_dag_full_causal_pipeline_and_store_persistence(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import VlmRunArtifact
    from robo_vlm_qc.models.review import EpisodeStatus
    from robo_vlm_qc.store import Repository

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_dag", duration_s=4.0)
    ep_id = _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        art = _run_dag_episode(dag, view, task_spec=cfg.get_task("pick_and_place"))

    assert isinstance(art, VlmRunArtifact)
    h.assert_no_legacy_fields(art)
    assert art.stage_refs.s1_grounding
    assert art.stage_refs.s2_macro
    assert art.stage_refs.s3_discover
    assert art.stage_refs.s4_verify
    assert art.stage_refs.s5_success

    # Verify causal DAG stage coverage in MockVlmServer requests
    stages_seen = {r["stage"] for r in mock_vlm_server.requests}
    assert {"S1", "S2", "S5", "S5_CONTROL", "S3", "S4A", "S4B"}.issubset(stages_seen)

    repo = Repository(e2e_workspace)
    try:
        row = repo.get_episode_row(ep_id)
        assert row["status"] == EpisodeStatus.VLM_READY.value
        assert row["active_run_id"] == art.run_id
        resolved = repo.resolve_episode(ep_id)
        assert resolved.active_run_id == art.run_id
    finally:
        repo.close()


# ===========================================================================
# Light Guardrails (splice_gate, hallucination, visual_gate)
# ===========================================================================


def test_splice_gate_boundary_idle_passes_cut_clean(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    kin = _make_kinematics_track(
        duration_s=4.0,
        left_disp_fn=lambda t: max(0.0, (t - 0.5) * 0.08),
    )
    seg = QcSegment(
        seg_id="b_idle",
        t_start=0.0,
        t_end=0.4,
        label_id="boundary_idle",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )
    out = _invoke_guardrail_module(
        "robo_vlm_qc.guardrail.splice_gate",
        ("apply_splice_gate", "run_splice_gate", "check_splice_gate", "SpliceGate"),
        segments=[seg],
        kinematics=kin,
        cfg=cfg.settings.get("guardrail", {}),
        duration_s=4.0,
    )
    assert out[0].splice_policy == "cut_clean"
    assert out[0].guardrail.passed is True
    assert out[0].guardrail.downgraded is False


def test_splice_gate_stationary_mid_idle_passes_cut_clean(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    kin = _make_kinematics_track(duration_s=4.0)  # strictly stationary
    seg = QcSegment(
        seg_id="mid_stat",
        t_start=1.2,
        t_end=2.0,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )
    out = _invoke_guardrail_module(
        "robo_vlm_qc.guardrail.splice_gate",
        ("apply_splice_gate", "run_splice_gate", "check_splice_gate", "SpliceGate"),
        segments=[seg],
        kinematics=kin,
        cfg=cfg.settings.get("guardrail", {}),
        duration_s=4.0,
    )
    assert out[0].splice_policy == "cut_clean"
    assert out[0].guardrail.passed is True
    assert out[0].guardrail.metrics.eef_pos_m <= 0.02


def test_splice_gate_moving_arm_jump_downgrades_to_mask_loss(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Continuous monotonic motion (> 2.0 cm everywhere around [1.0, 2.2])
    kin = _make_kinematics_track(
        duration_s=4.0,
        left_disp_fn=lambda t: t * 0.12,
    )
    seg = QcSegment(
        seg_id="bad_cut",
        t_start=1.2,
        t_end=2.2,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="left",
        vlm_raw_policy="cut_clean",
    )
    out = _invoke_guardrail_module(
        "robo_vlm_qc.guardrail.splice_gate",
        ("apply_splice_gate", "run_splice_gate", "check_splice_gate", "SpliceGate"),
        segments=[seg],
        kinematics=kin,
        cfg=cfg.settings.get("guardrail", {}),
        duration_s=4.0,
    )
    res = out[0]
    assert res.vlm_raw_policy == "cut_clean"
    assert res.splice_policy == "mask_loss"
    assert res.mask_side in {"left", "both"}
    assert res.guardrail.downgraded is True
    assert res.guardrail.downgrade_reason
    assert res.guardrail.metrics.eef_pos_m > 0.02


def test_splice_gate_bimanual_coupling_downgrades_over_3cm(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Left arm is stationary; right arm moves monotonically by 0.10m/s (> 3cm across [1.2, 2.2])
    kin = _make_kinematics_track(
        duration_s=4.0,
        left_disp_fn=lambda t: 0.0,
        right_disp_fn=lambda t: t * 0.10,
    )
    seg = QcSegment(
        seg_id="bim_cpl",
        t_start=1.2,
        t_end=2.2,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="left",
        vlm_raw_policy="cut_clean",
    )
    out = _invoke_guardrail_module(
        "robo_vlm_qc.guardrail.splice_gate",
        ("apply_splice_gate", "run_splice_gate", "check_splice_gate", "SpliceGate"),
        segments=[seg],
        kinematics=kin,
        cfg=cfg.settings.get("guardrail", {}),
        duration_s=4.0,
    )
    res = out[0]
    assert res.splice_policy == "mask_loss"
    assert res.guardrail.downgraded is True
    assert res.guardrail.metrics.bimanual_coupled is True
    assert res.guardrail.metrics.other_arm_disp_m > 0.03


def test_hallucination_verifier_overrides_idle_freeze_on_large_motion(
    e2e_config_dir: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Both arms advance monotonically by 0.25m (> 15cm threshold) during [1.0, 2.5]
    kin = _make_kinematics_track(
        duration_s=4.0,
        left_disp_fn=lambda t: max(0.0, (t - 0.8) * 0.20),
        right_disp_fn=lambda t: max(0.0, (t - 0.8) * 0.20),
    )
    seg = QcSegment(
        seg_id="hallu_idle",
        t_start=1.0,
        t_end=2.5,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )
    out = _invoke_guardrail_module(
        "robo_vlm_qc.guardrail.hallucination",
        (
            "apply_hallucination_check",
            "run_hallucination_guardrail",
            "verify_hallucination",
            "HallucinationGuardrail",
        ),
        segments=[seg],
        kinematics=kin,
        cfg=cfg.settings.get("guardrail", {}),
        duration_s=4.0,
    )
    res = out[0]
    assert res.splice_policy == "keep_positive"
    assert res.mask_side is None
    assert res.guardrail.guardrail_override is True
    assert res.guardrail.override_reason


def test_visual_gate_stationary_pure_video_passes_cut_clean(
    e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Build identical static video frames so SSIM ~= 1.0 and L1 ~= 0.0
    vid_ep = tmp_path / "static_vid" / "ep_0000"
    vid_ep.mkdir(parents=True, exist_ok=True)
    static_frame = np.full((240, 320, 3), 90, dtype=np.uint8)
    cv2.rectangle(static_frame, (80, 60), (200, 180), (40, 180, 220), -1)
    h._write_mp4_frames(vid_ep / "main.mp4", [static_frame.copy() for _ in range(60)], fps=30.0)

    handles = auto_detect(tmp_path / "static_vid", adapter_name="video_dir")
    seg = QcSegment(
        seg_id="vis_stat",
        t_start=0.5,
        t_end=1.4,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )
    with get_adapter("video_dir").open(handles[0]) as view:
        assert view.kinematics is None
        out = _invoke_guardrail_module(
            "robo_vlm_qc.guardrail.visual_gate",
            ("apply_visual_gate", "run_visual_gate", "check_visual_gate", "VisualGate"),
            segments=[seg],
            kinematics=None,
            media=view.media,
            cfg=cfg.settings.get("guardrail", {}),
            duration_s=view.duration_s,
        )
    assert out[0].splice_policy == "cut_clean"
    assert out[0].guardrail.gate_type == "visual_gate"
    assert out[0].guardrail.passed is True


def test_visual_gate_changed_frames_downgrades_to_mask_loss(
    e2e_config_dir: Path, tmp_path: Path
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import QcSegment

    cfg = ConfigManager(config_dir=e2e_config_dir)
    # Build video whose first half is dark blue and second half has large bright yellow block
    vid_ep = tmp_path / "changed_vid" / "ep_0000"
    vid_ep.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(60):
        fr = np.full((240, 320, 3), (80, 20, 20), dtype=np.uint8)
        if i >= 30:
            fr[:, :] = (20, 220, 240)
            cv2.circle(fr, (160, 120), 70, (255, 255, 255), -1)
        frames.append(fr)
    h._write_mp4_frames(vid_ep / "main.mp4", frames, fps=30.0)

    handles = auto_detect(tmp_path / "changed_vid", adapter_name="video_dir")
    seg = QcSegment(
        seg_id="vis_chg",
        t_start=0.4,
        t_end=1.6,
        label_id="mid_idle_freeze",
        training_role="badcase",
        splice_policy="cut_clean",
        mask_side=None,
        affected_arm="both",
        vlm_raw_policy="cut_clean",
    )
    with get_adapter("video_dir").open(handles[0]) as view:
        out = _invoke_guardrail_module(
            "robo_vlm_qc.guardrail.visual_gate",
            ("apply_visual_gate", "run_visual_gate", "check_visual_gate", "VisualGate"),
            segments=[seg],
            kinematics=None,
            media=view.media,
            cfg=cfg.settings.get("guardrail", {}),
            duration_s=view.duration_s,
        )
    res = out[0]
    assert res.vlm_raw_policy == "cut_clean"
    assert res.splice_policy == "mask_loss"
    assert res.guardrail.gate_type == "visual_gate"
    assert res.guardrail.downgraded is True
    assert res.guardrail.downgrade_reason


# ===========================================================================
# 1080p Studio HUD MP4 Renderer & `run-vlm` / `render-mp4` CLIs
# ===========================================================================


def test_viz_mp4_1080p_30fps_hud_renderer(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.store import Repository

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_viz", duration_s=1.5)
    ep_id = _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")
    out_mp4 = tmp_path / "hud_1080p.mp4"

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        _run_dag_episode(dag, view, task_spec=cfg.get_task("pick_and_place"))
        repo = Repository(e2e_workspace)
        try:
            resolved = repo.resolve_episode(ep_id)
        finally:
            repo.close()
        rendered = _render_viz(resolved, view, out_mp4, cfg=cfg)

    assert rendered.is_file()
    info = h.ffprobe_video_info(rendered)
    assert info["width"] == 1920 and info["height"] == 1080
    assert info["fps"] == pytest.approx(30.0, abs=0.5)
    assert (info["nb_frames"] or 0) >= 40


def test_cli_run_vlm_with_mock_server(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_cli_vlm", duration_s=2.0)
    ep_id = _ingest_one(e2e_workspace, e2e_config_dir, h5p)

    res = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "run-vlm",
            "--episode",
            ep_id,
        ]
    )
    assert res.returncode == 0, res.stderr + res.stdout
    assert len(mock_vlm_server.requests) >= 5

    conn = sqlite3.connect(str(e2e_workspace / "robo_vlm_qc.db"))
    try:
        row = conn.execute(
            "SELECT status, active_run_id FROM episodes WHERE episode_id = ?",
            (ep_id,),
        ).fetchone()
        assert row is not None
        assert row[0] == "vlm_ready"
        assert row[1] is not None
    finally:
        conn.close()


def test_cli_render_mp4_from_vlm_ready_episode(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_cli_mp4", duration_s=1.2)
    ep_id = _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    r_vlm = h.run_cli(
        ["-w", str(e2e_workspace), "-c", str(e2e_config_dir), "run-vlm", "--episode", ep_id]
    )
    assert r_vlm.returncode == 0, r_vlm.stderr + r_vlm.stdout

    out_mp4 = tmp_path / "cli_render.mp4"
    r_mp4 = h.run_cli(
        [
            "-w",
            str(e2e_workspace),
            "-c",
            str(e2e_config_dir),
            "render-mp4",
            "--episode",
            ep_id,
            "--out",
            str(out_mp4),
        ]
    )
    assert r_mp4.returncode == 0, r_mp4.stderr + r_mp4.stdout
    assert out_mp4.is_file()
    info = h.ffprobe_video_info(out_mp4)
    assert info["width"] == 1920 and info["height"] == 1080
    assert info["fps"] == pytest.approx(30.0, abs=0.5)


def test_viz_mp4_reflects_resolved_episode_human_edits(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.models.annotation import MacroStage, QcSegment, SuccessVerdict
    from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
    from robo_vlm_qc.store import Repository

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    h5p = h.make_synthetic_h5_raw_v3(tmp_path / "pick_hud_diff", duration_s=1.2)
    ep_id = _ingest_one(e2e_workspace, e2e_config_dir, h5p)
    handles = auto_detect(h5p, adapter_name="h5_raw_v3")

    with get_adapter("h5_raw_v3").open(handles[0]) as view:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        art = _run_dag_episode(dag, view, task_spec=cfg.get_task("pick_and_place"))
        repo = Repository(e2e_workspace)
        try:
            res_raw = repo.resolve_episode(ep_id)
            mp4_raw = _render_viz(res_raw, view, tmp_path / "hud_raw.mp4", cfg=cfg)

            # Submit human edit flipping success to failure and adding terminal_failure segment
            repo.save_review(
                ep_id,
                author="lead_hud",
                role="lead",
                payload=ReviewWritePayload(
                    base_run_id=art.run_id,
                    action="approve",
                    qc_segments=[
                        QcSegment(
                            seg_id="h_term",
                            t_start=0.4,
                            t_end=1.0,
                            label_id="terminal_failure",
                            training_role="terminal_fail",
                            splice_policy="mask_loss",
                            mask_side="both",
                            vlm_raw_policy="mask_loss",
                        )
                    ],
                    success=SuccessVerdict(passed=False, verdict="failure", confidence=1.0),
                    macro_stages=[
                        MacroStage(
                            stage_id="S1",
                            t_start=0.0,
                            t_end=view.duration_s,
                            label="Human revised failure stage",
                        )
                    ],
                    watch_coverage=WatchCoverage(
                        duration_s=view.duration_s,
                        watched_s=view.duration_s,
                        coverage_ratio=1.0,
                        confirmed=True,
                    ),
                ),
            )
            res_human = repo.resolve_episode(ep_id)
            mp4_human = _render_viz(res_human, view, tmp_path / "hud_human.mp4", cfg=cfg)
        finally:
            repo.close()

    cap1 = cv2.VideoCapture(str(mp4_raw))
    cap2 = cv2.VideoCapture(str(mp4_human))
    try:
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 18)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 18)
        ok1, f1 = cap1.read()
        ok2, f2 = cap2.read()
        assert ok1 and ok2
        assert f1.shape == (1080, 1920, 3) and f2.shape == (1080, 1920, 3)
        assert not np.array_equal(f1, f2), "HUD video must reflect ResolvedEpisode human edits"
    finally:
        cap1.release()
        cap2.release()


def test_viz_mp4_adaptive_camera_layout_wrist_and_single(
    e2e_workspace: Path,
    e2e_config_dir: Path,
    mock_vlm_server: Any,
    tmp_path: Path,
) -> None:
    from robo_vlm_qc.config import ConfigManager
    from robo_vlm_qc.io import auto_detect, get_adapter
    from robo_vlm_qc.store import Repository

    cfg = ConfigManager(config_dir=e2e_config_dir, workspace_dir=e2e_workspace)
    lr_dir = h.make_synthetic_lerobot(
        tmp_path / "lr_wrist_viz",
        version="v3",
        n_episodes=1,
        duration_s=1.0,
        task_name="bimanual_handover",
    )
    vid_dir = h.make_synthetic_video_dir(
        tmp_path / "single_viz", multi_cam=False, duration_s=1.0
    )

    ep_lr = _ingest_one(e2e_workspace, e2e_config_dir, lr_dir)
    ep_vid = _ingest_one(e2e_workspace, e2e_config_dir, vid_dir)

    h_lr = auto_detect(lr_dir, adapter_name="lerobot")[0]
    h_vid = auto_detect(vid_dir, adapter_name="video_dir")[0]

    with get_adapter("lerobot").open(h_lr) as view_lr, get_adapter("video_dir").open(h_vid) as view_vid:
        dag = _build_vlm_dag(e2e_workspace, cfg)
        _run_dag_episode(dag, view_lr, task_spec=cfg.get_task("bimanual_handover"))
        _run_dag_episode(dag, view_vid, task_spec=cfg.get_task("_default"))
        repo = Repository(e2e_workspace)
        try:
            mp4_lr = _render_viz(repo.resolve_episode(ep_lr), view_lr, tmp_path / "lr_1080p.mp4", cfg=cfg)
            mp4_vid = _render_viz(repo.resolve_episode(ep_vid), view_vid, tmp_path / "vid_1080p.mp4", cfg=cfg)
        finally:
            repo.close()

    info_lr = h.ffprobe_video_info(mp4_lr)
    info_vid = h.ffprobe_video_info(mp4_vid)
    assert info_lr["width"] == 1920 and info_lr["height"] == 1080
    assert info_vid["width"] == 1920 and info_vid["height"] == 1080
