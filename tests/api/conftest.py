"""Shared fixtures and synthetic helpers for tests/api (strictly tmp_path-isolated)."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_e2e_modules() -> tuple[Any, Any]:
    try:
        from tests.e2e import conftest as _e2e_conftest
        from tests.e2e import helpers as _e2e_helpers

        return _e2e_conftest, _e2e_helpers
    except ModuleNotFoundError:
        e2e_dir = REPO_ROOT / "tests" / "e2e"
        tests_pkg = sys.modules.get("tests")
        if tests_pkg is None:
            tests_pkg = types.ModuleType("tests")
            tests_pkg.__path__ = [str(REPO_ROOT / "tests")]  # type: ignore[attr-defined]
            sys.modules["tests"] = tests_pkg
        else:
            existing_paths = list(getattr(tests_pkg, "__path__", []) or [])
            repo_tests_str = str(REPO_ROOT / "tests")
            if repo_tests_str not in existing_paths:
                tests_pkg.__path__ = [repo_tests_str, *existing_paths]  # type: ignore[attr-defined]

        if "tests.e2e" not in sys.modules:
            e2e_spec = importlib.util.spec_from_file_location(
                "tests.e2e",
                e2e_dir / "__init__.py",
                submodule_search_locations=[str(e2e_dir)],
            )
            assert e2e_spec is not None and e2e_spec.loader is not None
            e2e_mod = importlib.util.module_from_spec(e2e_spec)
            sys.modules["tests.e2e"] = e2e_mod
            tests_pkg.e2e = e2e_mod  # type: ignore[attr-defined]
            e2e_spec.loader.exec_module(e2e_mod)

        if "tests.e2e.helpers" not in sys.modules:
            h_spec = importlib.util.spec_from_file_location(
                "tests.e2e.helpers", e2e_dir / "helpers.py"
            )
            assert h_spec is not None and h_spec.loader is not None
            h_mod = importlib.util.module_from_spec(h_spec)
            sys.modules["tests.e2e.helpers"] = h_mod
            sys.modules["tests.e2e"].helpers = h_mod  # type: ignore[attr-defined]
            h_spec.loader.exec_module(h_mod)

        if "tests.e2e.conftest" not in sys.modules:
            c_spec = importlib.util.spec_from_file_location(
                "tests.e2e.conftest", e2e_dir / "conftest.py"
            )
            assert c_spec is not None and c_spec.loader is not None
            c_mod = importlib.util.module_from_spec(c_spec)
            sys.modules["tests.e2e.conftest"] = c_mod
            sys.modules["tests.e2e"].conftest = c_mod  # type: ignore[attr-defined]
            c_spec.loader.exec_module(c_mod)

        return sys.modules["tests.e2e.conftest"], sys.modules["tests.e2e.helpers"]


_e2e_conftest, _e2e_helpers = _ensure_e2e_modules()
MockVlmServer = _e2e_conftest.MockVlmServer
_sanitize_test_env = _e2e_conftest._sanitize_test_env
_make_frame_bgr = _e2e_helpers._make_frame_bgr
_write_mp4_frames = _e2e_helpers._write_mp4_frames
make_synthetic_h5_raw_v3 = _e2e_helpers.make_synthetic_h5_raw_v3
write_workspace_config = _e2e_helpers.write_workspace_config

from robo_vlm_qc.api import create_app
from robo_vlm_qc.config import ConfigManager
from robo_vlm_qc.io import auto_detect, get_adapter
from robo_vlm_qc.models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    MacroStage,
    QcSegment,
    StageUsage,
    SuccessVerdict,
    UsageSummary,
    VisualEvidence,
    VlmRunArtifact,
)
from robo_vlm_qc.store import Repository

REV_HEADERS = {"X-Auth-Token": "rev-tok", "X-User": "reviewer_1"}
LEAD_HEADERS = {"X-Auth-Token": "lead-tok", "X-User": "lead_1"}


@pytest.fixture
def api_mock_vlm(monkeypatch: pytest.MonkeyPatch) -> MockVlmServer:
    _sanitize_test_env(monkeypatch)
    srv = MockVlmServer()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def api_env(tmp_path: Path, api_mock_vlm: MockVlmServer) -> dict[str, Any]:
    cfg_dir = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", cfg_dir)
    ws = tmp_path / "ws"
    for sub in ("artifacts", "stage_cache", "exports", "configs_override", ".history"):
        (ws / sub).mkdir(parents=True, exist_ok=True)
    write_workspace_config(
        ws,
        base_url=api_mock_vlm.base_url,
        api_key_env="MOCK_VLM_API_KEY",
    )
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    return {
        "tmp_path": tmp_path,
        "workspace": ws,
        "config_dir": cfg_dir,
        "data_root": data_root,
        "mock_vlm": api_mock_vlm,
    }


def make_client(
    ws: Path,
    cfg_dir: Path,
    *,
    allowed_data_roots: list[Path] | None = None,
    cors_origins: list[str] | None = None,
) -> TestClient:
    kwargs: dict[str, Any] = {
        "workspace": ws,
        "config_dir": cfg_dir,
        "reviewer_token": "rev-tok",
        "lead_token": "lead-tok",
        "allowed_data_roots": allowed_data_roots,
    }
    if cors_origins is not None:
        kwargs["cors_origins"] = cors_origins
    app = create_app(**kwargs)
    return TestClient(app)


def ingest_dir(
    repo: Repository,
    cfg_dir: Path,
    ws: Path,
    src_dir: Path,
    *,
    dataset: str | None = None,
) -> list[str]:
    cfg = ConfigManager(config_dir=cfg_dir, workspace_dir=ws)
    handles = auto_detect(src_dir, task_patterns=cfg.task_patterns())
    ep_ids: list[str] = []
    for h in handles:
        ad = get_adapter(h.adapter_type)
        with ad.open(h) as view:
            meta = view.meta
            ds_name = dataset or meta.dataset
            resolved_task = cfg.resolve_task(ds_name, None).task_id
            if ds_name != meta.dataset or resolved_task != meta.task_id:
                meta = meta.model_copy(
                    update={
                        "dataset": ds_name,
                        "episode_id": f"{ds_name}/{meta.stem}",
                        "task_id": resolved_task,
                    }
                )
            repo.ingest_episode(meta)
            ep_ids.append(meta.episode_id)
    return ep_ids


def seed_vlm_run(
    repo: Repository,
    cfg: ConfigManager,
    ep_id: str,
    *,
    run_id: str,
    created_at: float | None = None,
    prompt_version: str = "p1",
    label_version: str | int = "1",
    is_eval_sandbox: bool = False,
    passed: bool = True,
    verdict: str = "success",
    segments: list[QcSegment] | None = None,
) -> VlmRunArtifact:
    meta = repo.get_episode_meta(ep_id)
    segs = segments if segments is not None else [
        QcSegment(
            seg_id="seg_1",
            t_start=0.0,
            t_end=min(0.4, meta.duration_s * 0.25),
            label_id="boundary_idle",
            name="Boundary idle",
            training_role="badcase",
            splice_policy="cut_clean",
            mask_side=None,
            affected_arm="both",
            visual_evidence=VisualEvidence(
                initial_state="start",
                process_change="idle",
                end_state="end",
            ),
            vlm_raw_policy="cut_clean",
            confidence=0.9,
        )
    ]
    art = VlmRunArtifact(
        run_id=run_id,
        episode_id=ep_id,
        created_at=created_at if created_at is not None else time.time(),
        is_eval_sandbox=is_eval_sandbox,
        config_fingerprint=ConfigFingerprintSnapshot(
            config_sha256=cfg.fingerprints.config_sha256,
            model_routing=dict(cfg.settings.get("vlm", {}).get("stage_models", {})),
            prompt_bundle_version=prompt_version,
            label_taxonomy_version=str(label_version),
            task_spec_version="t1",
        ),
        macro_stages=[
            MacroStage(
                stage_id="S1",
                t_start=0.0,
                t_end=meta.duration_s,
                label="Operation",
                label_en="op",
            )
        ],
        arm_timelines=ArmTimelines(),
        qc_segments=segs,
        success=SuccessVerdict(
            verdict=verdict,  # type: ignore[arg-type]
            passed=passed,
            confidence=0.92,
            summary="seed",
        ),
        usage=UsageSummary(
            stages=[
                StageUsage(
                    stage="s3_discover",
                    model="gpt-4o",
                    total_tokens=150,
                    elapsed_s=0.1,
                )
            ],
            total_tokens=150,
            total_elapsed_s=0.1,
        ),
    )
    repo.save_vlm_run_artifact(art)
    return art


def create_video_episode_file(
    ds_dir: Path,
    filename: str = "ep_vid_001.mp4",
    *,
    duration_s: float = 2.0,
    fps: float = 15.0,
    identical_frames: bool = False,
) -> Path:
    ds_dir.mkdir(parents=True, exist_ok=True)
    n_frames = max(6, int(round(duration_s * fps)))
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        idx = 0 if identical_frames else i
        frames.append(_make_frame_bgr(160, 120, idx, n_frames, "vid"))
    out = ds_dir / filename
    _write_mp4_frames(out, frames, fps=fps)
    return out
