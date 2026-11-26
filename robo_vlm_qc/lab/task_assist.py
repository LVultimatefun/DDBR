"""Advisory AI-assisted TaskSpec drafter (never writes configs_override/)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ConfigManager
from ..io import get_adapter
from ..io.base import EpisodeHandle, EpisodeView, validate_id_part
from ..io.media import encode_jpeg_data_url
from ..models.episode import EpisodeMeta
from ..models.taxonomy import GroundingObjectSpec, TaskSpec
from ..store.repository import Repository
from ..vlm.client import VlmClient

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
_GROUNDING_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_SERIAL_PATTERN_RE = re.compile(
    r"(?:^|[^a-z])(?:episode|ep|chunk|clip|traj|trajectory|seq|sequence|run|session|batch|trial|rec|record|sample)[-_ ]?\d+",
    re.IGNORECASE,
)
_DATE_PATTERN_RE = re.compile(
    r"(?:(?:^|[^0-9])20\d{2}[-_./]\d{1,2}[-_./]\d{1,2}(?=$|[^0-9])"
    r"|(?:^|[^0-9])20\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?=$|[^0-9])"
    r"|(?:^|[^0-9])\d{1,2}[-_./]\d{1,2}[-_./]20\d{2}(?=$|[^0-9])"
    r"|(?:^|[^0-9])\d{6,8}[_-]\d{4,6}(?=$|[^0-9]))",
    re.IGNORECASE,
)
_LONG_DIGITS_RE = re.compile(r"\d{4,}")
_HEX_SERIAL_RE = re.compile(r"\b(?:[0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.IGNORECASE)

_GENERIC_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "pico",
        "tracker",
        "capture",
        "camera",
        "cam",
        "video",
        "videos",
        "dataset",
        "datasets",
        "data",
        "default",
        "demo",
        "demos",
        "test",
        "tests",
        "sample",
        "samples",
        "record",
        "recording",
        "recordings",
        "episode",
        "episodes",
        "ep",
        "chunk",
        "chunks",
        "clip",
        "clips",
        "traj",
        "trajectory",
        "trajectories",
        "session",
        "run",
        "runs",
        "raw",
        "lerobot",
        "hdf5",
        "h5",
        "mp4",
        "rgb",
        "rgbd",
        "ego",
        "vr",
        "headset",
        "quest",
        "gopro",
        "realsense",
        "kinect",
        "sensor",
        "robot",
        "human",
        "hand",
        "hands",
        "arm",
        "arms",
        "task",
        "tasks",
        "train",
        "eval",
        "val",
        "valid",
        "split",
        "batch",
        "seq",
        "sequence",
        "frame",
        "frames",
        "log",
        "logs",
        "tmp",
        "temp",
        "misc",
        "other",
        "unknown",
    }
)

_CAPTURE_DEVICE_TOKENS: frozenset[str] = frozenset(
    {
        "pico",
        "tracker",
        "capture",
        "quest",
        "gopro",
        "realsense",
        "kinect",
        "headset",
        "camera",
        "video",
        "dataset",
        "default",
    }
)

_STATIC_OBJECT_HINTS: frozenset[str] = frozenset(
    {
        "bin",
        "tray",
        "container",
        "rack",
        "bowl",
        "basket",
        "box",
        "table_zone",
        "target_bin",
        "source_tray",
        "target_tray",
        "plate",
        "sink",
        "shelf",
        "drawer",
        "mat",
    }
)


def _tokenize_alpha_num(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", str(text or "").lower()) if tok]


def _slugify_grounding_id(raw: str, *, fallback_idx: int = 1) -> str:
    """Normalize an object identifier to match ^[a-z0-9][a-z0-9_-]{0,63}$."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(raw or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    if not slug:
        slug = f"object_{max(1, int(fallback_idx))}"
    elif not slug[0].isalnum():
        slug = f"obj_{slug}"
    slug = slug[:64].rstrip("_-")
    if not _GROUNDING_ID_RE.match(slug):
        slug = f"object_{max(1, int(fallback_idx))}"
    return slug


def _open_view_from_meta(meta: EpisodeMeta, cfg_mgr: ConfigManager) -> EpisodeView:
    """Open a read-only EpisodeView from persisted EpisodeMeta."""
    src_path = Path(meta.source_uri)
    if not src_path.exists():
        raise FileNotFoundError(f"Episode source not found: {meta.source_uri}")
    adapter = get_adapter(meta.adapter_type)
    h5_cfg = cfg_mgr.settings.get("io", {}).get("h5", {})
    md = dict(meta.extra_meta)
    md.setdefault("hand_mapping", h5_cfg.get("hand_mapping"))
    md.setdefault("min_invalid_gap_s", h5_cfg.get("min_invalid_gap_s", 0.5))
    handle = EpisodeHandle(
        episode_id=meta.episode_id,
        dataset=meta.dataset,
        stem=meta.stem,
        task_id=meta.task_id,
        adapter_type=meta.adapter_type,
        source_uri=meta.source_uri,
        source_mtime=meta.source_mtime,
        metadata=md,
    )
    view = adapter.open(handle)
    if view.meta.episode_id != meta.episode_id or view.meta.task_id != meta.task_id:
        view.meta = meta
    return view


def _is_non_semantic_rule(
    raw_value: str,
    *,
    task_tokens: set[str],
    blocked_generic: frozenset[str],
    blocked_devices: frozenset[str],
    sample_dataset_low: str | None,
    sample_hint_low: str | None,
) -> str | None:
    """Return a human-readable rejection reason if raw_value is overly generic or hijack-prone."""
    val = str(raw_value or "").strip()
    if not val:
        return "empty rule"

    core = re.sub(r"[\*\?\[\]]+", "", val).strip()
    if len(core) < 3:
        return f"too short or wildcard-only ({val!r})"

    if re.fullmatch(r"[\d_\-\s.:/]+", core):
        return f"numeric-only pattern ({val!r})"

    if _SERIAL_PATTERN_RE.search(core):
        return f"episode/chunk serial-number pattern ({val!r})"

    if _DATE_PATTERN_RE.search(core):
        return f"date/timestamp pattern ({val!r})"

    if _LONG_DIGITS_RE.search(core) and not any(
        d in task_tokens for d in _LONG_DIGITS_RE.findall(core)
    ):
        return f"serial/numeric identifier ({val!r})"

    if _HEX_SERIAL_RE.search(core) and not any(
        h.lower() in task_tokens for h in _HEX_SERIAL_RE.findall(core)
    ):
        return f"hex/UUID serial identifier ({val!r})"

    tokens = _tokenize_alpha_num(core)
    if not tokens:
        return f"no alphanumeric semantic tokens ({val!r})"

    significant_tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    if not significant_tokens:
        return f"no meaningful semantic words ({val!r})"

    if all(t in blocked_generic for t in significant_tokens):
        return f"generic or capture-device terms ({val!r})"

    device_hits = [t for t in significant_tokens if t in blocked_devices]
    if device_hits:
        return f"capture-device/generic token {device_hits[0]!r} in ({val!r})"

    core_low = core.lower()
    if sample_dataset_low and core_low == sample_dataset_low:
        if not (set(significant_tokens) & task_tokens):
            return f"copies non-semantic sample dataset name ({val!r})"
    if sample_hint_low and core_low == sample_hint_low:
        if not (set(significant_tokens) & task_tokens):
            return f"copies non-semantic sample metadata hint ({val!r})"

    return None


def _sanitize_match_rules(
    match: Sequence[str] | None,
    aliases: Sequence[str] | None,
    task_id: str,
    sample_dataset: str | None = None,
    sample_hint: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministically strip overly generic, serial-number, date, or capture-device match/alias rules."""
    task_tokens = {tok for tok in _tokenize_alpha_num(task_id) if len(tok) >= 2}
    blocked_generic = _GENERIC_NOISE_TOKENS - task_tokens
    blocked_devices = _CAPTURE_DEVICE_TOKENS - task_tokens

    sample_dataset_low = sample_dataset.strip().lower() if sample_dataset and sample_dataset.strip() else None
    sample_hint_low = sample_hint.strip().lower() if sample_hint and sample_hint.strip() else None

    clean_match: list[str] = []
    seen_match: set[str] = set()
    clean_aliases: list[str] = []
    seen_aliases: set[str] = set()
    warnings: list[str] = []

    for item in match or ():
        if not isinstance(item, str):
            continue
        val = item.strip()
        if not val:
            continue
        reason = _is_non_semantic_rule(
            val,
            task_tokens=task_tokens,
            blocked_generic=blocked_generic,
            blocked_devices=blocked_devices,
            sample_dataset_low=sample_dataset_low,
            sample_hint_low=sample_hint_low,
        )
        if reason is not None:
            warnings.append(f"Stripped unsafe match rule {val!r}: {reason}")
            continue
        low = val.lower()
        if low not in seen_match:
            seen_match.add(low)
            clean_match.append(val)

    for item in aliases or ():
        if not isinstance(item, str):
            continue
        val = item.strip()
        if not val:
            continue
        reason = _is_non_semantic_rule(
            val,
            task_tokens=task_tokens,
            blocked_generic=blocked_generic,
            blocked_devices=blocked_devices,
            sample_dataset_low=sample_dataset_low,
            sample_hint_low=sample_hint_low,
        )
        if reason is not None:
            warnings.append(f"Stripped unsafe alias {val!r}: {reason}")
            continue
        low = val.lower()
        if low not in seen_aliases:
            seen_aliases.add(low)
            clean_aliases.append(val)

    return clean_match, clean_aliases, warnings


def _coerce_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(raw, Sequence):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    return []


def _coerce_grounding_objects(raw: Any) -> list[GroundingObjectSpec]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    specs: list[GroundingObjectSpec] = []
    used_ids: set[str] = set()
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, str):
            name = item.strip()
            if not name:
                continue
            base_id = _slugify_grounding_id(name, fallback_idx=idx)
            desc = ""
            is_static = base_id in _STATIC_OBJECT_HINTS
            tag_pos = "bottom" if is_static else "top"
        elif isinstance(item, dict):
            raw_id = str(item.get("id") or item.get("object_id") or "").strip()
            raw_name = str(item.get("name") or item.get("label") or raw_id).strip()
            if not raw_id and not raw_name:
                continue
            base_id = _slugify_grounding_id(raw_id or raw_name, fallback_idx=idx)
            name = raw_name or base_id.replace("_", " ").title()
            desc = str(item.get("desc") or item.get("description") or "").strip()
            raw_static = item.get("is_static")
            is_static = (
                bool(raw_static)
                if isinstance(raw_static, bool)
                else (base_id in _STATIC_OBJECT_HINTS)
            )
            raw_tp = str(item.get("tag_pos") or "").strip().lower()
            tag_pos = raw_tp if raw_tp in ("top", "bottom") else ("bottom" if is_static else "top")
        else:
            continue

        oid = base_id
        suffix = 2
        while oid in used_ids:
            suffix_str = f"_{suffix}"
            oid = f"{base_id[: 64 - len(suffix_str)]}{suffix_str}"
            suffix += 1
        used_ids.add(oid)
        specs.append(
            GroundingObjectSpec(
                id=oid,
                name=name,
                desc=desc,
                tag_pos=tag_pos,  # type: ignore[arg-type]
                is_static=is_static,
            )
        )
    return specs


def _sample_episode_frames(
    meta: EpisodeMeta,
    cfg_mgr: ConfigManager,
    num_frames: int,
) -> tuple[list[np.ndarray], list[float], str | None]:
    """Uniformly sample 4-6 RGB frames from the main camera of a read-only sample episode."""
    target_n = max(4, min(6, int(num_frames)))
    view = _open_view_from_meta(meta, cfg_mgr)
    try:
        main_cam = (
            view.camera_roles.get("main")
            or meta.camera_roles.get("main")
            or (view.media.cameras[0].name if view.media.cameras else meta.cameras[0].name)
        )
        dur = max(0.0, float(view.duration_s))
        if dur <= 1e-6:
            times_s = [0.0] * target_n
        else:
            times_s = [round(float(t), 3) for t in np.linspace(0.0, dur, target_n)]
        frames = list(view.media.read_frames_batch(main_cam, times_s))
        # Validate frames are JPEG-encodable without modifying episode files
        for fr in frames:
            encode_jpeg_data_url(fr, quality=70)
        return frames, times_s, main_cam
    finally:
        view.close()


def draft_task_spec(
    workspace_or_cfg: Path | str | ConfigManager | None = None,
    *,
    task_id: str,
    description: str,
    title: str | None = None,
    sample_episode_id: str | None = None,
    num_frames: int = 5,
    workspace: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    config: ConfigManager | None = None,
    cfg: ConfigManager | None = None,
    config_dir: Path | str | None = None,
    repo: Repository | None = None,
    client: VlmClient | None = None,
    vlm_client: VlmClient | None = None,
) -> dict[str, Any]:
    """Generate an advisory-only normalized English TaskSpec draft (never writes configs_override/)."""
    clean_tid = str(task_id or "").strip()
    validate_id_part(clean_tid, "task_id")
    if clean_tid.lower() == "_default":
        raise ValueError("task_id '_default' is reserved and cannot be drafted as a new task")
    if not _TASK_ID_RE.match(clean_tid):
        raise ValueError(
            f"Invalid task_id {clean_tid!r}; must match {_TASK_ID_RE.pattern}"
        )

    clean_desc = str(description or "").strip()
    if len(clean_desc) < 3:
        raise ValueError("description must be at least 3 non-whitespace characters")

    clean_title = str(title).strip() if title is not None and str(title).strip() else None
    clean_sample_ep = (
        str(sample_episode_id).strip()
        if sample_episode_id is not None and str(sample_episode_id).strip()
        else None
    )

    ws_val = workspace or workspace_dir or (
        workspace_or_cfg if not isinstance(workspace_or_cfg, ConfigManager) else None
    )
    ws = Path(ws_val).resolve() if ws_val is not None else None

    eff_cfg = (
        config
        or cfg
        or (workspace_or_cfg if isinstance(workspace_or_cfg, ConfigManager) else None)
        or ConfigManager(
            config_dir=Path(config_dir) if config_dir is not None else None,
            workspace_dir=ws,
        )
    )
    if ws is None and eff_cfg.workspace_dir is not None:
        ws = eff_cfg.workspace_dir

    frames: list[np.ndarray] = []
    timestamps_s: list[float] = []
    sample_dataset: str | None = None
    sample_hint: str | None = None
    sample_context_lines: list[str] = []

    if clean_sample_ep is not None:
        owns_repo = repo is None
        if owns_repo:
            if ws is None:
                raise ValueError(
                    "repo or workspace is required when sample_episode_id is provided"
                )
            eff_repo = Repository(ws)
        else:
            eff_repo = repo
        try:
            meta = eff_repo.get_episode_meta(clean_sample_ep)
        finally:
            if owns_repo:
                eff_repo.close()

        sample_dataset = meta.dataset
        raw_hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get("ep_task_str")
        sample_hint = str(raw_hint).strip() if raw_hint else None
        frames, timestamps_s, main_cam = _sample_episode_frames(
            meta, eff_cfg, num_frames=num_frames
        )
        sample_context_lines.append(
            f"Sample episode (read-only visual reference): episode_id={meta.episode_id}, "
            f"dataset={meta.dataset}, duration_s={meta.duration_s:.2f}, camera={main_cam}"
        )
        if sample_hint:
            sample_context_lines.append(
                f"Raw episode metadata hint (DO NOT copy device/serial names into match/aliases): {sample_hint}"
            )

    existing_ids = sorted(tid for tid in eff_cfg.tasks if tid != "_default")

    system_prompt = (
        "You are the VLM Lab TaskSpec onboarding advisor (task_assist). "
        "Draft an advisory-only, normalized English TaskSpec aligned with pipeline stages S1-S5 "
        "(S1 2D object grounding, S2 macro subtask segmentation, S3/S4 precision-first anomaly QC, "
        "S5 start-vs-end success verification).\n"
        "Strict requirements:\n"
        "1. Language: The user's input description may be in Chinese or English, but EVERY output field "
        "(title, description, goal, steps, criteria, task_prompt, grounding_objects, match, aliases, rationale) "
        "MUST be written in clear, objective English.\n"
        "2. Embodiment-agnostic phrasing: Support both robot arms/grippers and human hands / egocentric demonstrations.\n"
        "3. grounding_objects (S1): List 1 to 5 primary manipulated physical objects and key stationary target containers/zones. "
        "Never include robot arms, grippers, human hands, or irrelevant background clutter. "
        "Use snake_case ids matching ^[a-z0-9][a-z0-9_-]{0,63}$, tag_pos='bottom' for static containers/zones and 'top' for movable objects.\n"
        "4. steps (S2): 3 to 6 concise, physically observable milestone steps.\n"
        "5. criteria (S5): 2 to 5 concrete, visually verifiable completion criteria suitable for START-vs-END diff verification and drop scanning.\n"
        "6. task_prompt (S1-S5): Concise task-specific visual cues and false-positive guardrails (distinguish normal fine adjustment or intentional cyclic motion from true slips, retries, drops, or failures).\n"
        "7. Conservative match & aliases: Suggest semantic globs (e.g. '*{task_id}*') and semantic aliases ONLY from the semantic task name/id. "
        "NEVER include serial numbers (episode_001389, chunk_000), dates, generic tokens (dataset, data, demo, test, sample, default, video, camera, record), "
        "or capture device names (PICO, tracker, capture).\n"
        "Output strict JSON only with shape:\n"
        '{"task_spec": {"task_id": str, "version": "1.0.0", "title": str, "description": str, '
        '"goal": str, "steps": [str], "criteria": [str], "task_prompt": str, '
        '"match": [str], "aliases": [str], '
        '"grounding_objects": [{"id": str, "name": str, "desc": str, "tag_pos": "top|bottom", "is_static": bool}]}, '
        '"rationale": str, "warnings": [str]}'
    )

    user_lines = [
        f"Target task_id: {clean_tid}",
        f"Preferred title: {clean_title}" if clean_title else "Preferred title: (auto-generate concise English title)",
        f"User task description (may be English or Chinese): {clean_desc}",
        f"Existing configured task_ids (avoid overlapping match rules): {', '.join(existing_ids) or '(none)'}",
    ]
    if sample_context_lines:
        user_lines.append("[Sample Episode Context]")
        user_lines.extend(sample_context_lines)
        user_lines.append(
            f"Inspect the {len(frames)} uniformly sampled frames to identify concrete objects, containers, steps, and completion criteria."
        )
    else:
        user_lines.append("No sample episode frames provided; draft strictly from the task description.")
    user_lines.append("Return strict JSON matching the TaskSpec draft schema.")
    user_prompt = "\n".join(user_lines)

    owns_client = client is None and vlm_client is None
    eff_client = client or vlm_client or VlmClient.from_config(eff_cfg, workspace_dir=ws)
    try:
        resp = eff_client.chat_json(
            stage="task_assist",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=frames if frames else None,
            timestamps_s=timestamps_s if frames else None,
            thinking=False,
            max_tokens=2048,
        )
    finally:
        if owns_client:
            eff_client.close()

    if not isinstance(resp, dict):
        raise ValueError(f"Expected JSON object from task_assist VLM, got {type(resp).__name__}")

    raw_spec = resp.get("task_spec") if isinstance(resp.get("task_spec"), dict) else resp

    default_title = clean_title or clean_tid.replace("_", " ").replace("-", " ").title()
    spec_title = str(raw_spec.get("title") or "").strip() or default_title
    spec_desc = str(raw_spec.get("description") or "").strip() or clean_desc
    spec_goal = str(raw_spec.get("goal") or "").strip() or spec_desc
    spec_version = str(raw_spec.get("version") or "1.0.0").strip() or "1.0.0"
    spec_steps = _coerce_str_list(raw_spec.get("steps"))
    spec_criteria = _coerce_str_list(raw_spec.get("criteria"))
    spec_task_prompt = str(raw_spec.get("task_prompt") or "").strip()
    spec_objects = _coerce_grounding_objects(raw_spec.get("grounding_objects"))

    raw_match = _coerce_str_list(raw_spec.get("match"))
    raw_aliases = _coerce_str_list(raw_spec.get("aliases"))
    clean_match, clean_aliases, sanitize_warnings = _sanitize_match_rules(
        raw_match,
        raw_aliases,
        clean_tid,
        sample_dataset=sample_dataset,
        sample_hint=sample_hint,
    )

    warnings: list[str] = []
    seen_warn: set[str] = set()
    for w in [*_coerce_str_list(resp.get("warnings")), *sanitize_warnings]:
        if w not in seen_warn:
            seen_warn.add(w)
            warnings.append(w)

    raw_hand_map = raw_spec.get("hand_mapping")
    hand_mapping = (
        {str(k): str(v) for k, v in raw_hand_map.items()}
        if isinstance(raw_hand_map, dict)
        else None
    )
    raw_cam_override = raw_spec.get("camera_roles_override")
    camera_roles_override = (
        {str(k): str(v) for k, v in raw_cam_override.items()}
        if isinstance(raw_cam_override, dict)
        else None
    )

    spec = TaskSpec.model_validate(
        {
            "task_id": clean_tid,
            "version": spec_version,
            "match": clean_match,
            "aliases": clean_aliases,
            "title": spec_title,
            "description": spec_desc,
            "goal": spec_goal,
            "criteria": spec_criteria,
            "steps": spec_steps,
            "task_prompt": spec_task_prompt,
            "grounding_objects": [obj.model_dump(mode="json") for obj in spec_objects],
            "hand_mapping": hand_mapping,
            "camera_roles_override": camera_roles_override,
        }
    )

    rationale = str(resp.get("rationale") or "").strip()

    return {
        "advisory_only": True,
        "task_spec": spec.model_dump(mode="json"),
        "used_sample_episode_id": clean_sample_ep,
        "sampled_frames_count": len(frames),
        "rationale": rationale,
        "warnings": warnings,
    }


__all__ = [
    "_sanitize_match_rules",
    "draft_task_spec",
]
