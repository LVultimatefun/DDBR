"""Tests for GET /api/ingest/browse, --allowed-data-roots containment, and serve CLI options."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner

from robo_vlm_qc.cli import main
from tests.api.conftest import (
    LEAD_HEADERS,
    REV_HEADERS,
    create_video_episode_file,
    make_client,
)
from tests.e2e.helpers import make_synthetic_h5_raw_v3, make_synthetic_lerobot


def test_ingest_browse_auth_whitelist_symlinks_and_preflight(
    api_env: dict[str, Any], tmp_path: Path
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]

    root_a = tmp_path / "allowed_a"
    root_b = tmp_path / "allowed_b"
    evil_prefix = tmp_path / "allowed_a_evil"
    forbidden = tmp_path / "forbidden_secret"
    for d in (root_a, root_b, evil_prefix, forbidden):
        d.mkdir(parents=True, exist_ok=True)

    # Seed all 3 input adapter types cleanly inside root_a
    make_synthetic_h5_raw_v3(root_a / "h5_ds", name="ep_h5.h5", duration_s=3.0, stationary=True)
    make_synthetic_lerobot(root_a / "lerobot_ds", n_episodes=2, duration_s=2.0)
    create_video_episode_file(root_a / "vid_ds", "ep_v.mp4", duration_s=2.0, fps=15.0)
    (root_a / ".hidden_dir").mkdir(exist_ok=True)

    # Seed secret file & symlink escape pointing from root_a -> forbidden
    make_synthetic_h5_raw_v3(forbidden, name="secret.h5", duration_s=2.0, stationary=True)
    escape_link = root_a / "escape_link"
    escape_link.symlink_to(forbidden, target_is_directory=True)
    broken_link = root_a / "broken_link"
    broken_link.symlink_to(tmp_path / "does_not_exist_target")

    with make_client(ws, cfg_dir, allowed_data_roots=[root_a, root_b]) as client:
        # Auth / RBAC
        assert client.get("/api/ingest/browse").status_code == 401
        assert client.get("/api/ingest/browse", headers=REV_HEADERS).status_code == 403

        # Default path -> root_a, parent_path is None at root boundary
        r_root = client.get("/api/ingest/browse", headers=LEAD_HEADERS)
        assert r_root.status_code == 200
        b_root = r_root.json()
        assert Path(b_root["current_path"]) == root_a.resolve()
        assert b_root["parent_path"] is None
        assert b_root["allowed_roots"] == [str(root_a.resolve()), str(root_b.resolve())]

        entry_names = {e["name"] for e in b_root["entries"]}
        assert "h5_ds" in entry_names
        assert "lerobot_ds" in entry_names
        assert "vid_ds" in entry_names
        # Escaping symlink, broken symlink, and dotfiles MUST be omitted
        assert "escape_link" not in entry_names
        assert "broken_link" not in entry_names
        assert ".hidden_dir" not in entry_names

        # Preflight detected_adapters reflects all 3 formats without double-counting structured MP4s
        ad_map = {d["adapter_type"]: d["episode_count"] for d in b_root["detected_adapters"]}
        assert set(ad_map.keys()) == {"h5_raw_v3", "lerobot", "video_dir"}
        assert ad_map["h5_raw_v3"] == 1
        assert ad_map["lerobot"] == 2
        assert ad_map["video_dir"] == 1

        # Subdirectory browsing preserves parent_path within root_a
        r_sub = client.get(
            f"/api/ingest/browse?path={root_a / 'h5_ds'}", headers=LEAD_HEADERS
        )
        assert r_sub.status_code == 200
        assert Path(r_sub.json()["parent_path"]) == root_a.resolve()

        # Root B works
        assert (
            client.get(f"/api/ingest/browse?path={root_b}", headers=LEAD_HEADERS).status_code
            == 200
        )

        # Out-of-root paths, prefix collision, parent traversal, symlink escape -> 403 (even if nonexistent!)
        for bad_path in (
            str(forbidden),
            str(evil_prefix),
            str(root_a / ".." / "forbidden_secret"),
            str(escape_link),
            str(tmp_path / "nonexistent_outside_dir"),
        ):
            r_bad = client.get(
                "/api/ingest/browse", params={"path": bad_path}, headers=LEAD_HEADERS
            )
            assert r_bad.status_code == 403, (bad_path, r_bad.text)

        # Inside root_a: missing path -> 404, file path -> 400
        assert (
            client.get(
                "/api/ingest/browse",
                params={"path": str(root_a / "missing_sub")},
                headers=LEAD_HEADERS,
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/ingest/browse",
                params={"path": str(root_a / "h5_ds" / "ep_h5.h5")},
                headers=LEAD_HEADERS,
            ).status_code
            == 400
        )


def test_cli_serve_allowed_data_roots_host_default_and_cors_options(
    tmp_path: Path, monkeypatch: Any
) -> None:
    runner = CliRunner()
    r_help = runner.invoke(main, ["serve", "--help"])
    assert r_help.exit_code == 0
    assert "--allowed-data-roots" in r_help.output
    assert "--cors-origin" in r_help.output
    assert "127.0.0.1" in r_help.output

    captured: dict[str, Any] = {}

    def _fake_create_app(**kwargs: Any) -> object:
        captured["create_app"] = kwargs
        return object()

    def _fake_uvicorn_run(app: Any, host: str, port: int) -> None:
        captured["uvicorn"] = {"host": host, "port": port}

    import robo_vlm_qc.api as api_mod
    import uvicorn

    monkeypatch.setattr(api_mod, "create_app", _fake_create_app)
    monkeypatch.setattr(uvicorn, "run", _fake_uvicorn_run)

    r1 = tmp_path / "root1"
    r2 = tmp_path / "root2"
    r1.mkdir()
    r2.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()

    res = runner.invoke(
        main,
        [
            "--workspace",
            str(ws),
            "serve",
            "--no-auth",
            "--allowed-data-roots",
            str(r1),
            "--allowed-data-roots",
            str(r2),
            "--cors-origin",
            "https://studio.example.com",
            "--cors-origin",
            "http://localhost:3000",
        ],
    )
    assert res.exit_code == 0, res.output
    assert captured["uvicorn"]["host"] == "127.0.0.1"
    assert captured["create_app"]["allowed_data_roots"] == [r1, r2]
    assert list(captured["create_app"]["cors_origins"]) == [
        "https://studio.example.com",
        "http://localhost:3000",
    ]
