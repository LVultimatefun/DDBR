"""Unit tests for AI-assisted new-task TaskSpec drafting and anti-hijack match sanitization."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robo_vlm_qc.config import ConfigManager, default_config_dir
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.lab import draft_task_spec as lab_draft_task_spec
from robo_vlm_qc.lab.task_assist import _sanitize_match_rules, draft_task_spec
from robo_vlm_qc.models.taxonomy import TaskSpec
from robo_vlm_qc.store import NotFoundError, Repository
from tests.e2e.helpers import make_synthetic_lerobot, make_synthetic_video_dir

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


class _FakeVlmClient:
    """Deterministic in-memory stand-in for VlmClient.chat_json."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.response)

    def close(self) -> None:
        self.closed = True


def _make_cfg(tmp_path: Path) -> tuple[Path, Path, ConfigManager]:
    cfg_dir = tmp_path / "configs"
    shutil.copytree(default_config_dir(), cfg_dir)
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
    return ws, cfg_dir, cfg


def _ingest_source(repo: Repository, cfg: ConfigManager, src_dir: Path) -> list[str]:
    handles = auto_detect(src_dir, task_patterns=cfg.task_patterns())
    assert handles, f"No episodes discovered under {src_dir}"
    ep_ids: list[str] = []
    for handle in handles:
        adapter = get_adapter(handle.adapter_type)
        with adapter.open(handle) as view:
            meta = view.meta
            hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get("ep_task_str")
            resolved = cfg.resolve_task(meta.dataset, str(hint) if hint else None).task_id
            if resolved != meta.task_id:
                meta = meta.model_copy(update={"task_id": resolved})
            repo.ingest_episode(meta)
            ep_ids.append(meta.episode_id)
    return ep_ids


def _sample_vlm_draft_payload(
    *,
    match: list[str] | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "title": "Towel Folding",
        "description": "Fold a flat tabletop towel twice into a neat rectangle.",
        "goal": "Fold the towel cleanly twice and place it centered on the tabletop.",
        "steps": [
            "Grasp opposite corners of the towel with the active manipulators.",
            "Fold the towel in half along the first crease and smooth it flat.",
            "Fold the towel a second time into a compact square and release cleanly.",
        ],
        "criteria": [
            "Towel is folded cleanly twice with aligned edges.",
            "Towel rests flat in the workspace center after manipulators retract.",
        ],
        "task_prompt": (
            "Evaluate grasp stability on fabric corners, crease alignment during both folds, "
            "and whether the towel remains folded after manipulator release."
        ),
        "grounding_objects": [
            {
                "id": "Target Towel!",
                "name": "Towel",
                "desc": "Deformable fabric towel on the workspace table",
                "tag_pos": "top",
                "is_static": False,
            },
            {
                "id": "Target Towel!",
                "name": "Duplicate Towel Ref",
                "desc": "Duplicate id should be deduplicated cleanly",
                "tag_pos": "invalid_pos",
                "is_static": False,
            },
            {
                "id": "fold_zone",
                "name": "Folding Zone",
                "desc": "Designated central region of the table",
                "tag_pos": "bottom",
                "is_static": True,
            },
        ],
        "match": match if match is not None else ["*fold_towel*", "*towel_fold*"],
        "aliases": aliases if aliases is not None else ["fold towel", "towel folding"],
        "rationale": "Drafted embodiment-agnostic towel folding spec aligned with S1-S5.",
    }


def test_draft_task_spec_text_only_chinese_input_advisory_only(tmp_path: Path) -> None:
    assert lab_draft_task_spec is draft_task_spec

    ws, _cfg_dir, cfg = _make_cfg(tmp_path)
    fake_client = _FakeVlmClient(_sample_vlm_draft_payload())

    # Chinese user description encoded as ASCII unicode escapes to preserve OSS hygiene
    zh_description = (
        "\u5c06\u684c\u9762\u4e0a\u7684\u6bdb\u5dfe\u5bf9\u6298\u4e24\u6b21"
        "\u5e76\u6574\u9f50\u653e\u7f6e\u5728\u4e2d\u592e\u533a\u57df"
    )
    res = draft_task_spec(
        cfg,
        task_id="fold_towel",
        description=zh_description,
        title=None,
        client=fake_client,
    )

    assert res["advisory_only"] is True
    assert res["used_sample_episode_id"] is None
    assert res["sampled_frames_count"] == 0
    assert isinstance(res["rationale"], str) and res["rationale"].strip()
    assert isinstance(res["warnings"], list)

    # Validate resulting TaskSpec strictly and confirm normalized English fields
    spec = TaskSpec.model_validate(res["task_spec"])
    assert spec.task_id == "fold_towel"
    assert spec.title == "Towel Folding"
    assert len(spec.steps) >= 2
    assert len(spec.criteria) >= 2
    assert spec.task_prompt.strip()
    for text_val in (
        spec.title,
        spec.description,
        spec.goal,
        spec.task_prompt,
        *spec.steps,
        *spec.criteria,
    ):
        assert not _CJK_RE.search(text_val), f"Unexpected non-English text in TaskSpec: {text_val!r}"

    # Grounding object IDs normalized to ^[a-z0-9][a-z0-9_-]{0,63}$, deduplicated, tag_pos clamped
    g_ids = [g.id for g in spec.grounding_objects]
    assert len(g_ids) == len(set(g_ids))
    assert all(re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", gid) for gid in g_ids), g_ids
    assert all(g.tag_pos in {"top", "bottom"} for g in spec.grounding_objects)

    # Inspect VLM client invocation
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call.get("stage") == "task_assist"
    assert call.get("thinking") is False
    assert not call.get("images")
    prompt_blob = f"{call.get('system_prompt', '')}\n{call.get('user_prompt', '')}\n{call.get('messages', '')}"
    assert zh_description in prompt_blob
    assert fake_client.closed is False

    # Strictly advisory: MUST NOT write to configs_override/tasks/
    assert not (ws / "configs_override" / "tasks").exists()
    assert not list((ws / "configs_override").rglob("*.yaml")) if (ws / "configs_override").exists() else True


def test_draft_task_spec_multimodal_with_lerobot_and_video_episodes(tmp_path: Path) -> None:
    ws, _cfg_dir, cfg = _make_cfg(tmp_path)

    # 1. Synthetic LeRobot dataset with noisy serial dataset dir + capture device task hint
    lr_dir = tmp_path / "data" / "episode_001389"
    make_synthetic_lerobot(
        lr_dir,
        version="v3",
        n_episodes=1,
        duration_s=2.0,
        fps=15,
        task_name="PICO tracker capture",
    )

    # 2. Synthetic pure-video directory episode
    vid_dir = tmp_path / "data" / "raw_video_20261126"
    make_synthetic_video_dir(vid_dir, multi_cam=False, duration_s=2.0, fps=15, n_episodes=1)

    repo = Repository(ws)
    try:
        lr_eps = _ingest_source(repo, cfg, lr_dir)
        vid_eps = _ingest_source(repo, cfg, vid_dir)
        lr_ep_id = lr_eps[0]
        vid_ep_id = vid_eps[0]

        assert repo.get_episode_row(lr_ep_id)["task_id"] == "_default"
        assert repo.get_episode_row(vid_ep_id)["task_id"] == "_default"

        # VLM response intentionally echoes noisy serial/device rules to verify anti-hijack filter
        noisy_payload = _sample_vlm_draft_payload(
            match=["*fold_towel*", "*episode_001389*", "*PICO tracker capture*"],
            aliases=["fold towel", "PICO tracker capture", "episode_001389"],
        )

        # Multimodal draft against LeRobot episode (5 frames)
        fake_lr = _FakeVlmClient(noisy_payload)
        res_lr = draft_task_spec(
            cfg,
            task_id="fold_towel",
            description="Fold a towel neatly on the tabletop.",
            title="Fold Towel",
            sample_episode_id=lr_ep_id,
            num_frames=5,
            repo=repo,
            client=fake_lr,
        )
        assert res_lr["advisory_only"] is True
        assert res_lr["used_sample_episode_id"] == lr_ep_id
        assert res_lr["sampled_frames_count"] == 5
        assert len(fake_lr.calls) == 1

        call_lr = fake_lr.calls[0]
        assert call_lr.get("stage") == "task_assist"
        images_lr = call_lr.get("images")
        ts_lr = call_lr.get("timestamps_s")
        assert images_lr is not None and len(images_lr) == 5
        assert ts_lr is not None and len(ts_lr) == 5
        assert all(isinstance(im, (np.ndarray, str)) for im in images_lr)
        assert all(ts_lr[i] <= ts_lr[i + 1] for i in range(len(ts_lr) - 1))

        # Noisy serial / capture-device globs must be stripped while semantic glob remains
        spec_lr = TaskSpec.model_validate(res_lr["task_spec"])
        assert "*fold_towel*" in spec_lr.match
        assert not any("episode_001389" in m.lower() for m in spec_lr.match)
        assert not any("pico" in m.lower() or "tracker" in m.lower() or "capture" in m.lower() for m in spec_lr.match)
        assert not any("episode_001389" in a.lower() for a in spec_lr.aliases)
        assert not any("pico" in a.lower() or "tracker" in a.lower() for a in spec_lr.aliases)
        assert res_lr["warnings"], "Expected anti-hijack warnings for stripped noisy rules"

        # Multimodal draft against video_dir episode + num_frames clamping (2 -> 4, 9 -> 6)
        fake_vid_low = _FakeVlmClient(_sample_vlm_draft_payload())
        res_vid_low = draft_task_spec(
            cfg,
            task_id="fold_towel",
            description="Fold a towel neatly on the tabletop.",
            sample_episode_id=vid_ep_id,
            num_frames=2,
            repo=repo,
            client=fake_vid_low,
        )
        assert res_vid_low["used_sample_episode_id"] == vid_ep_id
        assert res_vid_low["sampled_frames_count"] == 4
        assert len(fake_vid_low.calls[0]["images"]) == 4
        assert len(fake_vid_low.calls[0]["timestamps_s"]) == 4

        fake_vid_high = _FakeVlmClient(_sample_vlm_draft_payload())
        res_vid_high = draft_task_spec(
            cfg,
            task_id="fold_towel",
            description="Fold a towel neatly on the tabletop.",
            sample_episode_id=vid_ep_id,
            num_frames=10,
            repo=repo,
            client=fake_vid_high,
        )
        assert res_vid_high["sampled_frames_count"] == 6
        assert len(fake_vid_high.calls[0]["images"]) == 6
        assert len(fake_vid_high.calls[0]["timestamps_s"]) == 6

        # Advisory-only: still zero writes to configs_override/tasks/
        assert not (ws / "configs_override" / "tasks").exists()
    finally:
        repo.close()


def test_sanitize_match_rules_strips_serial_date_and_capture_patterns() -> None:
    raw_match = [
        "*fold_towel*",
        "*fold-towel*",
        "*episode_001389*",
        "*chunk-000*",
        "*2026-11-26*",
        "*20261126*",
        "*2026_11_26*",
        "*PICO tracker capture*",
        "*tracker*",
        "*capture*",
        "*camera*",
        "*dataset*",
        "*default*",
        "*demo*",
        "*123456*",
        "*",
    ]
    raw_aliases = [
        "fold towel",
        "towel_folding",
        "episode_001389",
        "PICO tracker capture",
        "tracker",
        "capture",
        "2026-11-26",
        "001389",
    ]

    clean_match, clean_aliases, warnings = _sanitize_match_rules(
        raw_match,
        raw_aliases,
        "fold_towel",
        "episode_001389",
        "PICO tracker capture",
    )

    # Semantic patterns MUST be preserved
    assert "*fold_towel*" in clean_match
    assert "*fold-towel*" in clean_match
    assert "fold towel" in clean_aliases
    assert "towel_folding" in clean_aliases

    # Serial, date, wildcard, and capture-device patterns MUST be stripped
    forbidden_substrings = (
        "episode_001389",
        "chunk-000",
        "2026",
        "pico",
        "tracker",
        "capture",
        "camera",
        "dataset",
        "default",
        "demo",
        "123456",
        "001389",
    )
    for pat in clean_match:
        assert pat.strip("*") != ""
        low = pat.lower()
        for bad in forbidden_substrings:
            assert bad not in low, f"Pattern {pat!r} should have been stripped (matched {bad!r})"

    for al in clean_aliases:
        low = al.lower()
        for bad in forbidden_substrings:
            assert bad not in low, f"Alias {al!r} should have been stripped (matched {bad!r})"

    assert warnings, "Expected anti-hijack warnings to be emitted when patterns are stripped"


def test_draft_task_spec_validation_and_missing_sample_errors(tmp_path: Path) -> None:
    ws, _cfg_dir, cfg = _make_cfg(tmp_path)
    fake_client = _FakeVlmClient(_sample_vlm_draft_payload())

    # Reject reserved _default task_id
    with pytest.raises(ValueError):
        draft_task_spec(
            cfg,
            task_id="_default",
            description="Valid description for a manipulation task.",
            client=fake_client,
        )

    # Reject invalid task_id values
    for bad_id in ("", "   ", "../escape", "bad/id", "has spaces", "bad;id"):
        with pytest.raises(ValueError):
            draft_task_spec(
                cfg,
                task_id=bad_id,
                description="Valid description for a manipulation task.",
                client=fake_client,
            )

    # Reject empty / whitespace / too-short description
    for bad_desc in ("", "   ", "ab"):
        with pytest.raises(ValueError):
            draft_task_spec(
                cfg,
                task_id="valid_task",
                description=bad_desc,
                client=fake_client,
            )

    # Reject nonexistent sample_episode_id with NotFoundError
    repo = Repository(ws)
    try:
        with pytest.raises(NotFoundError):
            draft_task_spec(
                cfg,
                task_id="fold_towel",
                description="Fold a towel cleanly on the table.",
                sample_episode_id="missing_ds/ep_999999",
                repo=repo,
                client=fake_client,
            )
    finally:
        repo.close()

    # None of the invalid calls should have hit the VLM client
    assert fake_client.calls == []
