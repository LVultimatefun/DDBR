"""Unit and CLI tests for ingest, export, and subcommand validation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

from robo_vlm_qc.cli import main
from robo_vlm_qc.models.annotation import (
    ConfigFingerprintSnapshot,
    MacroStage,
    SuccessVerdict,
    VlmRunArtifact,
)
from robo_vlm_qc.models.review import ReviewWritePayload, WatchCoverage
from robo_vlm_qc.store import Repository


def _make_mp4(path: Path, n_frames: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120)
    )
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            frame = np.full((120, 160, 3), (i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_cli_ingest_dry_run_and_idempotent_db(tmp_path: Path) -> None:
    data_dir = tmp_path / "pick_and_place" / "success"
    _make_mp4(data_dir / "ep_01.mp4", n_frames=12)
    _make_mp4(data_dir / "ep_02.mp4", n_frames=15)
    ws = tmp_path / "ws"

    runner = CliRunner()

    # Dry-run creates no DB
    res_dry = runner.invoke(
        main,
        ["-w", str(ws), "ingest", "--data", str(tmp_path / "pick_and_place"), "--dry-run", "--json"],
    )
    assert res_dry.exit_code == 0, res_dry.output
    payload_dry = json.loads(res_dry.output)
    assert payload_dry["discovered"] == 2
    assert payload_dry["would_ingest"] == 2
    assert not (ws / "robo_vlm_qc.db").exists()

    # Real ingest
    res1 = runner.invoke(
        main,
        ["-w", str(ws), "ingest", "--data", str(tmp_path / "pick_and_place"), "--json"],
    )
    assert res1.exit_code == 0, res1.output
    p1 = json.loads(res1.output)
    assert p1["inserted"] == 2
    assert p1["episodes"][0]["task_id"] == "pick_and_place"

    # Re-run is idempotent (skipped == 2)
    res2 = runner.invoke(
        main,
        ["-w", str(ws), "ingest", "--data", str(tmp_path / "pick_and_place"), "--json"],
    )
    assert res2.exit_code == 0, res2.output
    p2 = json.loads(res2.output)
    assert p2["skipped"] == 2

    # --dataset override routes to matching task
    res3 = runner.invoke(
        main,
        [
            "-w",
            str(ws),
            "ingest",
            "--data",
            str(tmp_path / "pick_and_place"),
            "--dataset",
            "bimanual_handover",
            "--json",
        ],
    )
    assert res3.exit_code == 0, res3.output
    p3 = json.loads(res3.output)
    assert p3["inserted"] == 2
    assert all(ep["episode_id"].startswith("bimanual_handover/") for ep in p3["episodes"])
    assert all(ep["task_id"] == "bimanual_handover" for ep in p3["episodes"])

    repo = Repository(ws)
    try:
        rows = repo.list_episodes()
        assert len(rows) == 4
    finally:
        repo.close()


def test_cli_subcommands_require_arguments_or_valid_workspace(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_ws = str(tmp_path / "missing_ws")
    for args in (
        ["-w", missing_ws, "render-mp4"],
        ["-w", missing_ws, "export"],
        ["-w", missing_ws, "serve"],
        ["-w", missing_ws, "eval"],
    ):
        res = runner.invoke(main, args)
        assert res.exit_code == 2


def test_cli_export_json_dry_run_and_missing_source(tmp_path: Path) -> None:
    ws = tmp_path / "ws_cli"
    vdir = tmp_path / "vid_ds"
    _make_mp4(vdir / "ep_001.mp4", n_frames=45)

    runner = CliRunner()
    r_ing = runner.invoke(main, ["-w", str(ws), "ingest", "--data", str(vdir), "--json"])
    assert r_ing.exit_code == 0, r_ing.output
    ep_id = json.loads(r_ing.output)["episodes"][0]["episode_id"]

    repo = Repository(ws)
    try:
        art = VlmRunArtifact(
            run_id="r_cli",
            episode_id=ep_id,
            created_at=1.0,
            config_fingerprint=ConfigFingerprintSnapshot(
                model_routing={},
                prompt_bundle_version="1",
                label_taxonomy_version="1",
                task_spec_version="1",
            ),
            macro_stages=[MacroStage(stage_id="S1", t_start=0.0, t_end=1.5, label="Operation")],
            success=SuccessVerdict(passed=True, verdict="success", confidence=0.95),
        )
        repo.save_vlm_run_artifact(art)
        repo.save_review(
            ep_id,
            "lead",
            "lead",
            ReviewWritePayload(
                base_run_id="r_cli",
                action="approve",
                qc_segments=[],
                success=art.success,
                macro_stages=art.macro_stages,
                watch_coverage=WatchCoverage(
                    watched_intervals=[(0.0, 1.5)],
                    watched_s=1.5,
                    duration_s=1.5,
                    coverage_ratio=1.0,
                    confirmed=True,
                ),
            ),
        )
    finally:
        repo.close()

    dry_out = tmp_path / "dry_out"
    r_dry = runner.invoke(
        main,
        [
            "-w",
            str(ws),
            "export",
            "--format",
            "zero_copy_index",
            "--recipe",
            "bc_clean",
            "--out",
            str(dry_out),
            "--dry-run",
            "--json",
        ],
    )
    assert r_dry.exit_code == 0, r_dry.output
    dry_payload = json.loads(r_dry.output)
    assert dry_payload["dry_run"] is True
    assert dry_payload["episodes_included"] == 1
    assert not dry_out.exists()

    real_out = tmp_path / "real_zci"
    r_exp = runner.invoke(
        main,
        [
            "-w",
            str(ws),
            "export",
            "--format",
            "zero_copy_index",
            "--recipe",
            "bc_clean",
            "--out",
            str(real_out),
            "--episode",
            ep_id,
            "--json",
        ],
    )
    assert r_exp.exit_code == 0, r_exp.output
    assert (real_out / "qc_index.parquet").is_file()

    for mp4 in vdir.rglob("*.mp4"):
        mp4.unlink()
    err_out = tmp_path / "err_zci"
    r_err = runner.invoke(
        main,
        [
            "-w",
            str(ws),
            "export",
            "--format",
            "zero_copy_index",
            "--recipe",
            "bc_clean",
            "--out",
            str(err_out),
        ],
    )
    assert r_err.exit_code != 0
    assert "Traceback" not in r_err.output
