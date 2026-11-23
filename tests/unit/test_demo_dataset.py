"""Unit test verifying scripts/generate_demo_dataset.py generates ingestable episodes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from robo_vlm_qc.cli import main
from robo_vlm_qc.io import get_adapter
from robo_vlm_qc.store import Repository

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = REPO_ROOT / "scripts" / "generate_demo_dataset.py"


def test_generate_demo_dataset_runs_and_ingests_cleanly(tmp_path: Path) -> None:
    assert DEMO_SCRIPT.is_file(), f"Missing demo generator script: {DEMO_SCRIPT}"
    demo_out = tmp_path / "demo_data"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    proc = subprocess.run(
        [
            sys.executable,
            str(DEMO_SCRIPT),
            "--out",
            str(demo_out),
            "--duration-s",
            "1.0",
            "--fps",
            "15",
            "--width",
            "160",
            "--height",
            "120",
            "--json",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    gen_summary = json.loads(proc.stdout)
    assert len(gen_summary["episodes"]) == 3
    assert {e["adapter_type"] for e in gen_summary["episodes"]} == {"video_dir", "lerobot"}

    ws = tmp_path / "ws"
    runner = CliRunner()
    res = runner.invoke(main, ["-w", str(ws), "ingest", "--data", str(demo_out), "--json"])
    assert res.exit_code == 0, res.output
    ing = json.loads(res.output)
    assert ing["inserted"] == 3
    assert ing["errors_count"] == 0
    assert {ep["adapter_type"] for ep in ing["episodes"]} == {"video_dir", "lerobot"}

    repo = Repository(ws)
    try:
        rows = repo.list_episodes(limit=20)
        assert len(rows) == 3
        lr_rows = [r for r in rows if r["adapter_type"] == "lerobot"]
        assert len(lr_rows) == 1
        meta = repo.get_episode_meta(lr_rows[0]["episode_id"])
        assert meta.kinematics.has_kinematics is True
        assert meta.kinematics.state_dim == 81
        ad = get_adapter("lerobot")
        handles = ad.discover(Path(meta.source_uri))
        assert len(handles) == 1
        with ad.open(handles[0]) as view:
            assert view.kinematics is not None
            assert view.kinematics.raw_state is not None
            assert view.kinematics.raw_state.shape[1] == 81
            rgb = view.media.read_frame_rgb("main", 0.0)
            assert rgb.shape == (120, 160, 3)
    finally:
        repo.close()
