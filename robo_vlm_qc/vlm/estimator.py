"""Zero-API-call pre-dispatch VLM token & cost estimator (<50ms)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from ..config import ConfigManager, canonical_sha256
from ..models.episode import EpisodeMeta
from ..models.taxonomy import TaskSpec
from ..store.repository import Repository
from .cache import StageCache
from .client import (
    estimate_stage_cost_usd,
    is_gemini_hd_vision_model,
    normalize_image_detail,
    normalize_stage_name,
)
from .stage1_grounding import _plan_keyframe_times
from .stage2_macro import _plan_s2_times
from .stage3_discover import _plan_windows
from .stage4_verify import _plan_s4b_frame_count
from .stage5_success import _plan_control_times, _plan_dense_tail_times

_STAGES_ORDER: tuple[str, ...] = (
    "s1_grounding",
    "s2_macro",
    "s3_discover",
    "s4_verify",
    "s5_success",
)

_DOWNSTREAM_STAGE_MAP: dict[str, set[str]] = {
    "s1_grounding": {"s1_grounding"},
    "s2_macro": {"s2_macro", "s3_discover", "s4_verify"},
    "s3_discover": {"s3_discover", "s4_verify"},
    "s4_verify": {"s4_verify"},
    "s5_success": {"s5_success"},
}

_VALID_STAGE_INPUTS = frozenset(
    {
        "all",
        "s1",
        "s1_grounding",
        "s1_s5",
        "s1-s5",
        "s1/s5",
        "s2",
        "s2_macro",
        "s3",
        "s3_discover",
        "s3_s4",
        "s3/s4",
        "s3-s4",
        "s4",
        "s4a",
        "s4a_merge",
        "s4_merge",
        "s4b",
        "s4b_verify",
        "s4_verify",
        "s5",
        "s5_success",
    }
)


def _resolve_estimator_stage(
    stage: str | None, fresh: bool
) -> tuple[str, bool, set[str]]:
    raw = (stage or "all").strip().lower() or "all"
    if raw == "all":
        return "all", bool(fresh), set()
    if raw in {"s1_s5", "s1-s5", "s1/s5"}:
        return raw, True, set()
    if raw in {"s3_s4", "s3/s4", "s3-s4"}:
        return raw, False, {"s3_discover", "s4_verify"}
    if raw not in _VALID_STAGE_INPUTS:
        raise ValueError(
            f"Invalid stage {stage!r}; expected 'all', 's1_s5', 's3_s4', or s1..s5"
        )
    canon = normalize_stage_name(raw)
    return raw, bool(fresh), set(_DOWNSTREAM_STAGE_MAP.get(canon, {canon}))


def _tokens_per_image(
    provider: str,
    model: str,
    image_detail: str | None,
    width_px: int,
    height_px: int,
) -> int:
    """Estimate vision prompt tokens for a single image based on provider & detail."""
    norm_det = normalize_image_detail(image_detail) or "auto"
    if provider == "gemini":
        if norm_det == "low":
            return 258
        if is_gemini_hd_vision_model(model):
            return 1120
        return 258

    # OpenAI-compatible vision tiling (GPT-4o / GPT-4.1 style)
    if norm_det == "low":
        return 85
    w = max(1, int(width_px))
    h = max(1, int(height_px))
    if max(w, h) > 2048:
        scale = 2048.0 / float(max(w, h))
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))
    if min(w, h) > 768:
        scale = 768.0 / float(min(w, h))
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))
    tiles = math.ceil(w / 512.0) * math.ceil(h / 512.0)
    return 85 + 170 * max(1, tiles)


def _read_cached_json(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / key[:2] / f"{key}.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _resolve_task_for_meta(cfg: ConfigManager, meta: EpisodeMeta) -> TaskSpec:
    if meta.task_id and meta.task_id != "_default":
        return cfg.get_task(meta.task_id)
    hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get("ep_task_str")
    return cfg.resolve_task(meta.dataset, str(hint) if hint else None)


def _tile_dims(camera_roles: dict[str, str], cell_w: int, cell_h: int) -> tuple[int, int]:
    has_2x2 = any(
        k in camera_roles
        for k in ("left_wrist", "right_wrist", "front", "wide", "wide_right", "secondary")
    )
    if has_2x2:
        return cell_w * 2, cell_h * 2 + 26
    return cell_w * 2, cell_h + 26


def estimate_batch_vlm_cost(
    cfg: ConfigManager,
    repo: Repository,
    workspace: Path | str,
    episode_ids: Sequence[str],
    *,
    stage: str | None = "all",
    fresh: bool = False,
) -> dict[str, Any]:
    """Estimate VLM calls, images, tokens, cache hits, and USD cost without any VLM API calls."""
    stage_norm, eff_fresh, fresh_stages = _resolve_estimator_stage(stage, fresh)
    unique_ep_ids = list(dict.fromkeys(str(eid) for eid in episode_ids if eid))

    vlm_cfg = cfg.settings.get("vlm", {})
    provider = cfg.vlm_provider()
    pricing = vlm_cfg.get("pricing") if isinstance(vlm_cfg.get("pricing"), dict) else {}

    s1_cfg = dict(vlm_cfg.get("s1_grounding", {}))
    s2_cfg = dict(vlm_cfg.get("s2_macro", {}))
    s3_cfg = dict(vlm_cfg.get("s3_discover", {}))
    s4_cfg = dict(vlm_cfg.get("s4_verify", {}))
    s5_cfg = dict(vlm_cfg.get("s5_success", {}))

    stage_models = {st: cfg.stage_model(st) for st in _STAGES_ORDER}
    stage_details: dict[str, str] = {
        "s1_grounding": str(normalize_image_detail(s1_cfg.get("image_detail")) or "high"),
        "s2_macro": str(normalize_image_detail(s2_cfg.get("image_detail")) or "low"),
        "s3_discover": str(normalize_image_detail(s3_cfg.get("image_detail")) or "auto"),
        "s4_verify": str(normalize_image_detail(s4_cfg.get("image_detail")) or "auto"),
        "s5_success": str(normalize_image_detail(s5_cfg.get("image_detail")) or "auto"),
    }
    pass0_detail = str(normalize_image_detail(s1_cfg.get("pass0_image_detail")) or "low")

    low_detail_stages: list[str] = []
    if pass0_detail == "low":
        low_detail_stages.append("s1_pass0")
    for st in _STAGES_ORDER:
        if stage_details[st] == "low":
            low_detail_stages.append(st)

    token_saving_active = {
        "low_detail_stages": low_detail_stages,
        "static_frame_dedup": bool(
            s2_cfg.get("static_dedup_enabled", True)
            or s3_cfg.get("static_dedup_enabled", True)
        ),
        "adaptive_s3_step": bool(s3_cfg.get("adaptive_step_enabled", True)),
        "short_episode_s1_merge": bool(s1_cfg.get("merge_pass0_short_episode", True)),
        "s4b_dynamic_frames": bool(s4_cfg.get("dynamic_refine_frames", True)),
    }

    accum: dict[str, dict[str, Any]] = {
        st: {
            "stage": st,
            "model": stage_models[st],
            "estimated_calls": 0,
            "estimated_images": 0,
            "image_detail": stage_details[st],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "cached_episodes": 0,
        }
        for st in _STAGES_ORDER
    }

    ws_path = Path(workspace).resolve()
    cache_dir = ws_path / "stage_cache"
    task_bases_cache: dict[str, dict[str, str]] = {}

    def _get_task_bases(task_id: str) -> dict[str, str]:
        if task_id not in task_bases_cache:
            fs_list = repo.list_few_shots(task_id=task_id, enabled_only=True)
            fs_hash = (
                canonical_sha256([fs.model_dump(mode="json") for fs in fs_list])
                if fs_list
                else "none"
            )
            task_bases_cache[task_id] = {
                "s1": cfg.s1_cache_basis(task_id),
                "s2": cfg.s2_cache_basis(task_id),
                "s3": cfg.s3_cache_basis(task_id, fs_hash),
                "s4": cfg.s4_cache_basis(task_id, fs_hash),
                "s5": cfg.s5_cache_basis(task_id),
            }
        return task_bases_cache[task_id]

    total_duration_s = 0.0
    for ep_id in unique_ep_ids:
        meta = repo.get_episode_meta(ep_id)
        dur = max(0.05, float(meta.duration_s))
        total_duration_s += dur
        task = _resolve_task_for_meta(cfg, meta)
        bases = _get_task_bases(task.task_id)
        ep_fp = meta.episode_fingerprint
        cam_roles = meta.camera_roles or {"main": "main"}

        # Compute deterministic stage cache keys (exact parity with VlmDag)
        s1_key = StageCache.compute_key(episode_fingerprint=ep_fp, stage_basis=bases["s1"])
        s2_key = StageCache.compute_key(episode_fingerprint=ep_fp, stage_basis=bases["s2"])
        s5_key = StageCache.compute_key(episode_fingerprint=ep_fp, stage_basis=bases["s5"])

        # --- Stage 1: 2D Grounding ---
        s1_force = eff_fresh or ("s1_grounding" in fresh_stages)
        s1_hit = (not s1_force) and (cache_dir / s1_key[:2] / f"{s1_key}.json").is_file()
        if s1_hit:
            accum["s1_grounding"]["cached_episodes"] += 1
        elif bool(s1_cfg.get("enabled", True)):
            max_kf = max(1, int(s1_cfg.get("max_keyframes", 32)))
            hard_max_kf = max(1, int(s1_cfg.get("hard_max_keyframes", 72)))
            kfs = _plan_keyframe_times(
                dur, float(s1_cfg.get("interval_s", 2.5)), max_kf, hard_max_kf
            )
            n_kf = len(kfs)
            batch_size = max(1, int(s1_cfg.get("batch_size", 4)))
            n_batches = math.ceil(n_kf / batch_size) if batch_size > 1 else n_kf
            ego_h = int(s1_cfg.get("ego_height", 540))
            ego_w = int(round(ego_h * 16.0 / 9.0))
            needs_pass0 = (
                (not task.grounding_objects)
                and bool(s1_cfg.get("canonical_discovery_enabled", True))
                and n_kf > 1
            )
            merge_short = bool(s1_cfg.get("merge_pass0_short_episode", True))
            s1_tok_img = _tokens_per_image(
                provider, stage_models["s1_grounding"], stage_details["s1_grounding"], ego_w, ego_h
            )
            p0_tok_img = _tokens_per_image(
                provider, stage_models["s1_grounding"], pass0_detail, ego_w, ego_h
            )
            if needs_pass0 and merge_short and batch_size > 1 and n_kf <= batch_size:
                calls = 1
                imgs = n_kf
                p_tok = 700 + n_kf * s1_tok_img
                c_tok = min(int(s1_cfg.get("max_tokens", 2400)), 180 * n_kf + 150)
                cached_p = 0
            else:
                p0_calls = 1 if needs_pass0 else 0
                p0_imgs = (
                    min(n_kf, int(s1_cfg.get("canonical_discovery_frames", 5)))
                    if needs_pass0
                    else 0
                )
                ref_imgs = (
                    (n_batches - 1)
                    if (
                        batch_size > 1
                        and n_batches > 1
                        and bool(s1_cfg.get("reference_anchor_enabled", True))
                    )
                    else 0
                )
                calls = p0_calls + n_batches
                imgs = p0_imgs + n_kf + ref_imgs
                p_tok = (
                    p0_calls * 350
                    + p0_imgs * p0_tok_img
                    + n_batches * 650
                    + (n_kf + ref_imgs) * s1_tok_img
                )
                c_tok = p0_calls * 180 + n_batches * min(
                    int(s1_cfg.get("max_tokens", 2400)),
                    int(160 * (n_kf / max(1, n_batches))),
                )
                cached_p = 0
            accum["s1_grounding"]["estimated_calls"] += calls
            accum["s1_grounding"]["estimated_images"] += imgs
            accum["s1_grounding"]["prompt_tokens"] += p_tok
            accum["s1_grounding"]["completion_tokens"] += c_tok
            accum["s1_grounding"]["cached_prompt_tokens"] += cached_p

        # --- Stage 2: Macro Backbone ---
        s2_force = eff_fresh or ("s2_macro" in fresh_stages)
        s2_cached_data = None if s2_force else _read_cached_json(cache_dir, s2_key)
        s2_hit = s2_cached_data is not None
        if s2_hit:
            accum["s2_macro"]["cached_episodes"] += 1
        else:
            s2_times = _plan_s2_times(
                dur,
                float(s2_cfg.get("interval_s", 2.0)),
                int(s2_cfg.get("max_frames", 48)),
                float(s2_cfg.get("long_episode_threshold_s", 96.0)),
                int(s2_cfg.get("long_episode_max_frames", 64)),
            )
            n_s2 = len(s2_times)
            if bool(s2_cfg.get("static_dedup_enabled", True)) and n_s2 > 2:
                n_s2 = max(2, int(round(n_s2 * 0.80)))
            tw, th = _tile_dims(
                cam_roles, int(s2_cfg.get("cell_w", 320)), int(s2_cfg.get("cell_h", 180))
            )
            s2_tok_img = _tokens_per_image(
                provider, stage_models["s2_macro"], stage_details["s2_macro"], tw, th
            )
            accum["s2_macro"]["estimated_calls"] += 1
            accum["s2_macro"]["estimated_images"] += n_s2
            accum["s2_macro"]["prompt_tokens"] += 850 + n_s2 * s2_tok_img
            accum["s2_macro"]["completion_tokens"] += min(
                int(s2_cfg.get("max_tokens", 4096)), 650
            )

        # --- Stage 3: Sliding QC Discovery ---
        s3_force = eff_fresh or ("s3_discover" in fresh_stages)
        s3_cached_data: dict[str, Any] | None = None
        s3_key: str | None = None
        s2_out_hash = ""
        if s2_hit and s2_cached_data is not None:
            s2_out_hash = canonical_sha256(
                {
                    "macro_stages": s2_cached_data.get("macro_stages", []),
                    "arm_timelines": s2_cached_data.get("arm_timelines", {}),
                }
            )
            s3_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                upstream_hash=f"{s2_key}:{s2_out_hash}",
                stage_basis=bases["s3"],
            )
            if not s3_force:
                s3_cached_data = _read_cached_json(cache_dir, s3_key)
        s3_hit = s3_cached_data is not None
        if s3_hit:
            accum["s3_discover"]["cached_episodes"] += 1
        else:
            win_s = float(s3_cfg.get("window_s", 16.0))
            stride_s = float(s3_cfg.get("window_stride_s", 12.0))
            step_s = max(0.2, float(s3_cfg.get("window_step_s", 0.8)))
            idle_step_s = max(step_s, float(s3_cfg.get("idle_step_s", 1.4)))
            adaptive_on = bool(s3_cfg.get("adaptive_step_enabled", True))
            dedup_on = bool(s3_cfg.get("static_dedup_enabled", True))
            eff_step = (0.65 * step_s + 0.35 * idle_step_s) if adaptive_on else step_s
            windows = _plan_windows(dur, win_s, stride_s)
            n_wins = len(windows)
            s3_imgs = 0
            for w0, w1 in windows:
                span = max(0.02, w1 - w0)
                wf = max(4 if adaptive_on else 2, int(math.floor(span / eff_step)) + 1)
                if dedup_on and wf > 2:
                    wf = max(2, int(round(wf * 0.85)))
                s3_imgs += wf
            tw, th = _tile_dims(
                cam_roles, int(s3_cfg.get("cell_w", 420)), int(s3_cfg.get("cell_h", 236))
            )
            s3_tok_img = _tokens_per_image(
                provider, stage_models["s3_discover"], stage_details["s3_discover"], tw, th
            )
            # Stable system prefix (~1800 tok) hits prefix cache across windows 2..N
            sys_prefix_tok = 1800
            dyn_text_tok = 350
            p_tok = n_wins * (sys_prefix_tok + dyn_text_tok) + s3_imgs * s3_tok_img
            cached_p = max(0, n_wins - 1) * sys_prefix_tok
            accum["s3_discover"]["estimated_calls"] += n_wins
            accum["s3_discover"]["estimated_images"] += s3_imgs
            accum["s3_discover"]["prompt_tokens"] += p_tok
            accum["s3_discover"]["completion_tokens"] += n_wins * 350
            accum["s3_discover"]["cached_prompt_tokens"] += cached_p

        # --- Stage 4: Merge + CoV Refinement ---
        s4_force = eff_fresh or ("s4_verify" in fresh_stages)
        s4_hit = False
        if s2_hit and s3_hit and s3_key is not None and s3_cached_data is not None and not s4_force:
            s3_out_hash = canonical_sha256(s3_cached_data.get("segments", []))
            s4_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                upstream_hash=f"{s2_key}:{s2_out_hash}:{s3_key}:{s3_out_hash}",
                stage_basis=bases["s4"],
            )
            s4_hit = (cache_dir / s4_key[:2] / f"{s4_key}.json").is_file()
        if s4_hit:
            accum["s4_verify"]["cached_episodes"] += 1
        else:
            est_cands = max(1, min(5, int(round(dur / 25.0)) or 1))
            s4a_calls = 1 if est_cands >= 2 else 0
            s4b_calls = est_cands
            margin_s = float(s4_cfg.get("refine_margin_s", 1.5))
            step_s = float(s4_cfg.get("refine_step_s", 0.5))
            min_f = int(s4_cfg.get("refine_min_frames", 4))
            max_f = int(s4_cfg.get("refine_max_frames", 18))
            dyn_f = bool(s4_cfg.get("dynamic_refine_frames", True))
            typical_win = min(dur, 2.5 + 2.0 * margin_s)
            per_cand_imgs = _plan_s4b_frame_count(
                0.0,
                typical_win,
                refine_step_s=step_s,
                refine_min_frames=min_f,
                refine_max_frames=max_f,
                dynamic_refine_frames=dyn_f,
            )
            s4_imgs = s4b_calls * per_cand_imgs
            tw, th = _tile_dims(
                cam_roles, int(s4_cfg.get("cell_w", 384)), int(s4_cfg.get("cell_h", 216))
            )
            s4_tok_img = _tokens_per_image(
                provider, stage_models["s4_verify"], stage_details["s4_verify"], tw, th
            )
            sys_prefix_s4b = 1750
            p_tok = (
                s4a_calls * 1200
                + s4b_calls * (sys_prefix_s4b + 300)
                + s4_imgs * s4_tok_img
            )
            cached_p = max(0, s4b_calls - 1) * sys_prefix_s4b
            accum["s4_verify"]["estimated_calls"] += s4a_calls + s4b_calls
            accum["s4_verify"]["estimated_images"] += s4_imgs
            accum["s4_verify"]["prompt_tokens"] += p_tok
            accum["s4_verify"]["completion_tokens"] += s4a_calls * 400 + s4b_calls * 350
            accum["s4_verify"]["cached_prompt_tokens"] += cached_p

        # --- Stage 5: Dense-Tail Success + On-Demand Control ---
        s5_force = eff_fresh or ("s5_success" in fresh_stages)
        s5_hit = (not s5_force) and (cache_dir / s5_key[:2] / f"{s5_key}.json").is_file()
        if s5_hit:
            accum["s5_success"]["cached_episodes"] += 1
        elif bool(s5_cfg.get("enabled", True)):
            full_times = _plan_dense_tail_times(
                dur,
                float(s5_cfg.get("tail_s", 12.0)),
                int(s5_cfg.get("body_frames", 6)),
                int(s5_cfg.get("tail_frames", 10)),
                tail_ratio=float(s5_cfg.get("tail_ratio", 0.22)),
                max_tail_s=float(s5_cfg.get("max_tail_s", 25.0)),
            )
            _, _, ctrl_times = _plan_control_times(
                dur,
                float(s5_cfg.get("control_ratio", 0.30)),
                int(s5_cfg.get("control_frames", 8)),
            )
            has_cmp = bool(s5_cfg.get("start_end_compare", True))
            tw, th = _tile_dims(
                cam_roles, int(s5_cfg.get("cell_w", 420)), int(s5_cfg.get("cell_h", 236))
            )
            s5_tok_img = _tokens_per_image(
                provider, stage_models["s5_success"], stage_details["s5_success"], tw, th
            )
            cmp_w = int(s5_cfg.get("compare_panel_w", 640)) * 2 + 4
            cmp_h = int(s5_cfg.get("compare_panel_h", 360)) + 30
            cmp_tok = (
                _tokens_per_image(
                    provider,
                    stage_models["s5_success"],
                    stage_details["s5_success"],
                    cmp_w,
                    cmp_h,
                )
                if has_cmp
                else 0
            )
            full_imgs = len(full_times) + (1 if has_cmp else 0)
            ctrl_imgs = len(ctrl_times) + (1 if has_cmp else 0)
            # Conservative upper bound including on-demand 30% anti-FP control check
            calls = 2
            imgs = full_imgs + ctrl_imgs
            p_tok = (
                650
                + len(full_times) * s5_tok_img
                + cmp_tok
                + 450
                + len(ctrl_times) * s5_tok_img
                + cmp_tok
            )
            c_tok = 2 * min(int(s5_cfg.get("max_tokens", 1536)), 380)
            accum["s5_success"]["estimated_calls"] += calls
            accum["s5_success"]["estimated_images"] += imgs
            accum["s5_success"]["prompt_tokens"] += p_tok
            accum["s5_success"]["completion_tokens"] += c_tok

    per_stage: list[dict[str, Any]] = []
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    total_cache_hits = 0

    for st in _STAGES_ORDER:
        row = accum[st]
        p_tok = int(row["prompt_tokens"])
        c_tok = int(row["completion_tokens"])
        tot_tok = p_tok + c_tok
        st_cost = estimate_stage_cost_usd(
            row["model"],
            p_tok,
            c_tok,
            pricing,
            cached_tokens=int(row["cached_prompt_tokens"]),
        )
        total_prompt += p_tok
        total_completion += c_tok
        total_cost += st_cost
        total_cache_hits += int(row["cached_episodes"])
        per_stage.append(
            {
                "stage": st,
                "model": row["model"],
                "estimated_calls": int(row["estimated_calls"]),
                "estimated_images": int(row["estimated_images"]),
                "image_detail": row["image_detail"],
                "estimated_tokens": tot_tok,
                "estimated_cost_usd": round(st_cost, 6),
                "cached_episodes": int(row["cached_episodes"]),
            }
        )

    return {
        "episode_count": len(unique_ep_ids),
        "total_duration_s": round(total_duration_s, 3),
        "stage": stage_norm,
        "fresh": bool(eff_fresh),
        "estimated_prompt_tokens": total_prompt,
        "estimated_completion_tokens": total_completion,
        "estimated_total_tokens": total_prompt + total_completion,
        "estimated_cost_usd": round(total_cost, 6),
        "estimated_cache_hits": total_cache_hits,
        "per_stage": per_stage,
        "token_saving_active": token_saving_active,
    }
