"""Tests for .history/ config snapshot listing, diff, and atomic rollback."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from tests.api.conftest import LEAD_HEADERS, REV_HEADERS, make_client


def test_lab_history_dual_layout_filter_diff_and_rollback(api_env: dict[str, Any]) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]

    with make_client(ws, cfg_dir) as client:
        # Auth / RBAC
        assert client.get("/api/lab/history").status_code == 401
        assert client.get("/api/lab/history", headers=REV_HEADERS).status_code == 403

        # 1. Mutate labels -> creates initial base snapshot (.history/labels/) + override snapshot
        lbl_orig = client.get("/api/lab/labels", headers=LEAD_HEADERS).json()
        lbl_v2 = {**lbl_orig, "version": int(lbl_orig["version"]) + 1}
        assert client.put("/api/lab/labels", headers=LEAD_HEADERS, json=lbl_v2).status_code == 200

        lbl_v3 = {**lbl_orig, "version": int(lbl_orig["version"]) + 2}
        assert client.put("/api/lab/labels", headers=LEAD_HEADERS, json=lbl_v3).status_code == 200

        # 2. Mutate prompts (s3_discover) -> creates initial base snapshot (.history/s3_discover/)
        pr_orig = client.get("/api/lab/prompts", headers=LEAD_HEADERS).json()
        s3_orig = dict(pr_orig["s3_discover"])
        s3_v2 = {**s3_orig, "version": "p_hist_v2"}
        assert (
            client.put("/api/lab/prompts", headers=LEAD_HEADERS, json=s3_v2).status_code == 200
        )

        # 3. Mutate task spec
        tasks_orig = client.get("/api/lab/tasks", headers=LEAD_HEADERS).json()["tasks"]
        tid = next(iter(t for t in tasks_orig if t != "_default"))
        t_v2 = {**tasks_orig[tid], "goal": "Updated goal for history test"}
        assert client.put("/api/lab/tasks", headers=LEAD_HEADERS, json=t_v2).status_code == 200

        # Seed a corrupt JSON file and .config_override.lock inside .history/ (must be ignored!)
        (ws / ".history" / ".config_override.lock").touch()
        corrupt_dir = ws / ".history" / "corrupt_test"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        (corrupt_dir / "bad.json").write_text("{not_json", encoding="utf-8")

        # List all + filter by target & name (including prompt alias "s3")
        r_all = client.get("/api/lab/history", headers=LEAD_HEADERS)
        assert r_all.status_code == 200
        assert r_all.json()["total"] >= 4

        r_lbl_hist = client.get(
            "/api/lab/history?target=labels&name=labels", headers=LEAD_HEADERS
        )
        assert r_lbl_hist.status_code == 200
        lbl_items = r_lbl_hist.json()["items"]
        assert len(lbl_items) == 2  # initial_base_snapshot + v2 snapshot before v3 write
        assert all(it["target"] == "labels" for it in lbl_items)
        assert all(it["relative_path"] == "labels.yaml" for it in lbl_items)

        r_pr_hist = client.get(
            "/api/lab/history?target=prompts&name=s3", headers=LEAD_HEADERS
        )
        assert r_pr_hist.status_code == 200
        pr_items = r_pr_hist.json()["items"]
        assert len(pr_items) >= 1
        assert pr_items[0]["target"] == "prompts"
        assert pr_items[0]["name"] == "s3_discover"
        assert pr_items[0]["relative_path"] == "prompts/s3_discover.yaml"

        r_task_hist = client.get(
            f"/api/lab/history?target=tasks&name={tid}", headers=LEAD_HEADERS
        )
        assert r_task_hist.status_code == 200
        assert len(r_task_hist.json()["items"]) >= 1

        # Diff against initial base label snapshot (oldest item)
        init_lbl_snap = lbl_items[-1]
        assert "initial_base_snapshot" in init_lbl_snap["reason"]
        r_diff = client.get(
            f"/api/lab/history/{init_lbl_snap['snapshot_id']}/diff",
            headers=LEAD_HEADERS,
        )
        assert r_diff.status_code == 200
        b_diff = r_diff.json()
        assert b_diff["relative_path"] == "labels.yaml"
        assert "unified_diff" in b_diff and "@@" in b_diff["unified_diff"]
        line_types = {ln["type"] for ln in b_diff["lines"]}
        assert {"hunk", "add", "remove"}.issubset(line_types)

        # Rollback labels to initial base snapshot -> archives current v3, reloads ConfigManager
        seq_before = client.app.state.ctx.sse.latest_seq()  # type: ignore[attr-defined]
        sha_before = client.app.state.ctx.get_config().fingerprints.config_sha256  # type: ignore[attr-defined]
        r_rb = client.post(
            f"/api/lab/history/{init_lbl_snap['snapshot_id']}/rollback",
            headers=LEAD_HEADERS,
        )
        assert r_rb.status_code == 200
        b_rb = r_rb.json()
        assert b_rb["ok"] is True
        assert b_rb["relative_path"] == "labels.yaml"
        assert b_rb["config_sha256"] != sha_before

        # Active labels immediately match original version
        lbl_after = client.get("/api/lab/labels", headers=LEAD_HEADERS).json()
        assert lbl_after["version"] == lbl_orig["version"]

        # New archive entry was created with reason="rollback:<snapshot_id>"
        lbl_hist_after = client.get(
            "/api/lab/history?target=labels", headers=LEAD_HEADERS
        ).json()["items"]
        assert any(
            it["reason"] == f"rollback:{init_lbl_snap['snapshot_id']}"
            for it in lbl_hist_after
        )

        # SSE emitted config_updated
        _, events = client.app.state.ctx.sse.since(seq_before)  # type: ignore[attr-defined]
        assert any(
            ev.get("type") == "config_updated" and ev.get("reason") == "rollback"
            for ev in events
        )

        # Rollback prompt & task snapshots as well
        r_rb_pr = client.post(
            f"/api/lab/history/{pr_items[-1]['snapshot_id']}/rollback",
            headers=LEAD_HEADERS,
        )
        assert r_rb_pr.status_code == 200
        assert r_rb_pr.json()["relative_path"] == "prompts/s3_discover.yaml"
        assert (
            client.get("/api/lab/prompts", headers=LEAD_HEADERS).json()["s3_discover"]["version"]
            == s3_orig["version"]
        )

        task_items = r_task_hist.json()["items"]
        r_rb_t = client.post(
            f"/api/lab/history/{task_items[-1]['snapshot_id']}/rollback",
            headers=LEAD_HEADERS,
        )
        assert r_rb_t.status_code == 200
        assert r_rb_t.json()["relative_path"] == f"tasks/{tid}.yaml"
        assert (
            client.get("/api/lab/tasks", headers=LEAD_HEADERS).json()["tasks"][tid]["goal"]
            == tasks_orig[tid]["goal"]
        )


def test_lab_history_security_and_invalid_snapshot_validation(
    api_env: dict[str, Any], tmp_path: Path
) -> None:
    ws: Path = api_env["workspace"]
    cfg_dir: Path = api_env["config_dir"]

    # Create a valid labels override first so we can verify rollback failure leaves it untouched
    with make_client(ws, cfg_dir) as client:
        lbl_orig = client.get("/api/lab/labels", headers=LEAD_HEADERS).json()
        lbl_v2 = {**lbl_orig, "version": int(lbl_orig["version"]) + 1}
        assert client.put("/api/lab/labels", headers=LEAD_HEADERS, json=lbl_v2).status_code == 200
        override_before = (ws / "configs_override" / "labels.yaml").read_text("utf-8")

        # Bad format snapshot_id -> 400
        assert (
            client.get("/api/lab/history/bad!id/diff", headers=LEAD_HEADERS).status_code == 400
        )
        # Path traversal encoded in base64 -> 400
        trav_id = (
            base64.urlsafe_b64encode(b"../../etc/passwd.json").decode("ascii").rstrip("=")
        )
        assert (
            client.get(f"/api/lab/history/{trav_id}/diff", headers=LEAD_HEADERS).status_code
            == 400
        )

        # Non-existent valid base64 json path -> 404
        missing_id = (
            base64.urlsafe_b64encode(b"labels/does_not_exist.json")
            .decode("ascii")
            .rstrip("=")
        )
        assert (
            client.get(f"/api/lab/history/{missing_id}/diff", headers=LEAD_HEADERS).status_code
            == 404
        )

        # Symlink inside .history/ pointing outside .history/ -> 403
        outside_json = tmp_path / "outside.json"
        outside_json.write_text(
            json.dumps({"relative_path": "labels.yaml", "content": "version: 1\n"}),
            encoding="utf-8",
        )
        link_snap = ws / ".history" / "escape_snap.json"
        link_snap.symlink_to(outside_json)
        link_id = (
            base64.urlsafe_b64encode(b"escape_snap.json").decode("ascii").rstrip("=")
        )
        assert (
            client.get(f"/api/lab/history/{link_id}/diff", headers=LEAD_HEADERS).status_code
            == 403
        )
        assert (
            client.post(
                f"/api/lab/history/{link_id}/rollback", headers=LEAD_HEADERS
            ).status_code
            == 403
        )

        # Invalid Pydantic schema inside snapshot -> 422 and configs_override/labels.yaml untouched
        bad_snap_dir = ws / ".history" / "labels"
        bad_snap_dir.mkdir(parents=True, exist_ok=True)
        bad_snap = bad_snap_dir / "9999999999999_bad.json"
        bad_snap.write_text(
            json.dumps(
                {
                    "path": str(ws / "configs_override" / "labels.yaml"),
                    "relative_path": "configs_override/labels.yaml",
                    "created_at": 1700000000.0,
                    "reason": "corrupt_schema_test",
                    "encoding": "utf-8",
                    "content": "version: not_an_int\nlabels: 12345\n",
                }
            ),
            encoding="utf-8",
        )
        bad_snap_id = (
            base64.urlsafe_b64encode(b"labels/9999999999999_bad.json")
            .decode("ascii")
            .rstrip("=")
        )
        r_422 = client.post(
            f"/api/lab/history/{bad_snap_id}/rollback", headers=LEAD_HEADERS
        )
        assert r_422.status_code == 422
        assert (ws / "configs_override" / "labels.yaml").read_text("utf-8") == override_before
