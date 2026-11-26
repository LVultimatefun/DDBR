"""Click CLI for robo_vlm_qc (ingest, run-vlm, render-mp4, export, serve, eval)."""

from __future__ import annotations

import contextlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from .config import ConfigManager
from .export import PieceSlicer, get_exporter, group_kept_pieces
from .export.viz_mp4 import render_viz_mp4
from .io import auto_detect, get_adapter
from .io.base import EpisodeHandle, EpisodeView, validate_id_part
from .models.episode import EpisodeMeta
from .models.export import ExportManifest, ExportRecipe, KeepPiece
from .models.review import EpisodeStatus, ResolvedEpisode
from .store import NotFoundError, Repository
from .vlm import VlmDag
from .vlm.client import normalize_stage_name


@click.group()
@click.option(
    "--workspace",
    "-w",
    type=click.Path(path_type=Path),
    default=Path("./workspace"),
    show_default=True,
    help="Workspace directory for SQLite DB, artifacts, and stage cache.",
)
@click.option(
    "--config-dir",
    "-c",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional override path for base configs directory.",
)
@click.pass_context
def main(ctx: click.Context, workspace: Path, config_dir: Path | None) -> None:
    """robo_vlm_qc — VLM-Native embodied trajectory QC and data platform."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = workspace
    ctx.obj["config_dir"] = config_dir


@main.command("ingest")
@click.option(
    "--data",
    "data_paths",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    required=True,
    help="Read-only dataset file or directory to scan (repeatable).",
)
@click.option(
    "--adapter",
    type=click.Choice(["auto", "h5_raw_v3", "lerobot", "video_dir"]),
    default="auto",
    show_default=True,
    help="Input adapter selection.",
)
@click.option(
    "--dataset",
    "dataset_override",
    type=str,
    default=None,
    help="Optional dataset name override for all ingested episodes.",
)
@click.option(
    "--task",
    "--task-id",
    "task_override",
    type=str,
    default=None,
    help="Optional explicit task_id override for all ingested episodes (or 'auto').",
)
@click.option("--dry-run", is_flag=True, help="Discover & validate without writing DB.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
@click.pass_context
def ingest_cmd(
    ctx: click.Context,
    data_paths: tuple[Path, ...],
    adapter: str,
    dataset_override: str | None,
    task_override: str | None,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Scan strictly read-only data sources and register episodes into SQLite."""
    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]

    if dataset_override is not None:
        validate_id_part(dataset_override, "dataset")

    cfg_ws = workspace if (workspace / "configs_override").is_dir() else None
    cfg_mgr = ConfigManager(config_dir=config_dir, workspace_dir=cfg_ws)
    eff_task_override: str | None = None
    if task_override is not None:
        cleaned_task = task_override.strip()
        if cleaned_task and cleaned_task.lower() != "auto":
            if cleaned_task not in cfg_mgr.tasks:
                msg = (
                    f"Unknown --task {cleaned_task!r}. "
                    f"Configured tasks: {', '.join(sorted(cfg_mgr.tasks))}"
                )
                if as_json:
                    click.echo(
                        json.dumps(
                            {
                                "ok": False,
                                "error": msg,
                                "discovered": 0,
                                "inserted": 0,
                                "updated": 0,
                                "skipped": 0,
                                "errors": [],
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    click.echo(msg, err=True)
                sys.exit(1)
            eff_task_override = cleaned_task
    task_patterns = cfg_mgr.task_patterns()

    repo: Repository | None = None if dry_run else Repository(workspace)
    inserted = 0
    updated = 0
    skipped = 0
    would_insert = 0
    errors: list[dict[str, str]] = []
    adapter_counts: Counter[str] = Counter()
    episode_records: list[dict[str, Any]] = []

    try:
        all_handles = []
        for data_root in data_paths:
            handles = auto_detect(
                data_root,
                adapter_name=None if adapter == "auto" else adapter,
                task_patterns=task_patterns,
            )
            all_handles.extend(handles)

        if not all_handles:
            msg = "No readable episodes discovered under provided --data path(s)."
            if as_json:
                click.echo(
                    json.dumps(
                        {
                            "ok": False,
                            "error": msg,
                            "discovered": 0,
                            "inserted": 0,
                            "updated": 0,
                            "skipped": 0,
                            "errors": [],
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                click.echo(msg, err=True)
            sys.exit(1)

        for handle in all_handles:
            ad = get_adapter(handle.adapter_type)
            try:
                with ad.open(handle) as view:
                    meta = view.meta
                    ds_name = dataset_override or meta.dataset
                    if eff_task_override is not None:
                        resolved_task = eff_task_override
                    else:
                        h5_hint = meta.extra_meta.get("h5_task_name") or meta.extra_meta.get(
                            "ep_task_str"
                        )
                        resolved_task = cfg_mgr.resolve_task(
                            ds_name, str(h5_hint) if h5_hint else None
                        ).task_id
                    if ds_name != meta.dataset or resolved_task != meta.task_id:
                        meta = meta.model_copy(
                            update={
                                "dataset": ds_name,
                                "episode_id": f"{ds_name}/{meta.stem}",
                                "task_id": resolved_task,
                            }
                        )
                    adapter_counts[meta.adapter_type] += 1
                    if dry_run:
                        outcome = "dry_run"
                        would_insert += 1
                    else:
                        assert repo is not None
                        _, outcome = repo.ingest_episode(meta)
                        if outcome == "inserted":
                            inserted += 1
                        elif outcome == "updated":
                            updated += 1
                        else:
                            skipped += 1
                    episode_records.append(
                        {
                            "episode_id": meta.episode_id,
                            "adapter_type": meta.adapter_type,
                            "task_id": meta.task_id,
                            "duration_s": round(meta.duration_s, 3),
                            "fps": meta.fps,
                            "has_kinematics": meta.kinematics.has_kinematics,
                            "outcome": outcome,
                        }
                    )
            except Exception as exc:
                errors.append({"episode_id": handle.episode_id, "error": str(exc)})
    finally:
        if repo is not None:
            repo.close()

    report = {
        "ok": len(errors) == 0,
        "workspace": str(workspace),
        "dry_run": dry_run,
        "discovered": len(all_handles),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "would_ingest": would_insert,
        "errors_count": len(errors),
        "per_adapter": dict(adapter_counts),
        "episodes": episode_records,
        "errors": errors,
    }

    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        console = Console()
        table = Table(title="robo_vlm_qc ingest summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_row("Discovered", str(report["discovered"]))
        if dry_run:
            table.add_row("Dry-run valid", str(would_insert))
        else:
            table.add_row("Inserted", str(inserted))
            table.add_row("Updated", str(updated))
            table.add_row("Skipped", str(skipped))
        table.add_row("Errors", str(len(errors)))
        for ad_name, cnt in sorted(adapter_counts.items()):
            table.add_row(f"Adapter: {ad_name}", str(cnt))
        console.print(table)
        if errors:
            for err in errors:
                click.echo(f"ERROR [{err['episode_id']}]: {err['error']}", err=True)

    if errors:
        sys.exit(1)


def _open_view_from_meta(meta: EpisodeMeta, cfg_mgr: ConfigManager) -> EpisodeView:
    ad = get_adapter(meta.adapter_type)
    if meta.adapter_type == "lerobot":
        handles = ad.discover(Path(meta.source_uri))
        matched = next((h for h in handles if h.stem == meta.stem), None)
        if matched is None:
            raise FileNotFoundError(
                f"LeRobot episode {meta.episode_id!r} not found under {meta.source_uri}"
            )
        view = ad.open(matched)
        if view.meta.episode_id != meta.episode_id or view.meta.task_id != meta.task_id:
            view.meta = meta
        return view

    h5_cfg = cfg_mgr.settings.get("io", {}).get("h5", {})
    handle = EpisodeHandle(
        episode_id=meta.episode_id,
        dataset=meta.dataset,
        stem=meta.stem,
        task_id=meta.task_id,
        adapter_type=meta.adapter_type,
        source_uri=meta.source_uri,
        source_mtime=meta.source_mtime,
        metadata={
            "hand_mapping": h5_cfg.get("hand_mapping"),
            "min_invalid_gap_s": h5_cfg.get("min_invalid_gap_s", 0.5),
            "h5_task_name": meta.extra_meta.get("h5_task_name"),
            "ep_task_str": meta.extra_meta.get("ep_task_str"),
        },
    )
    view = ad.open(handle)
    if view.meta.episode_id != meta.episode_id or view.meta.task_id != meta.task_id:
        view.meta = meta
    return view


@main.command("run-vlm")
@click.option("--episode", "episodes", multiple=True, help="Target episode_id (repeatable).")
@click.option("--dataset", default=None, help="Filter by dataset.")
@click.option("--stage", default="all", help="Stage selection (all|s1..s5).")
@click.option("--fresh", is_flag=True, help="Ignore StageCache and re-run.")
@click.option("--workers", type=int, default=4, show_default=True, help="Episode worker concurrency.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
@click.pass_context
def run_vlm_cmd(
    ctx: click.Context,
    episodes: tuple[str, ...],
    dataset: str | None,
    stage: str,
    fresh: bool,
    workers: int,
    as_json: bool,
) -> None:
    """Run causal VLM DAG + guardrails on ingested episodes."""
    stage_norm = stage.strip().lower()
    _valid_stages = {
        "all",
        "s1",
        "s1_grounding",
        "s2",
        "s2_macro",
        "s3",
        "s3_discover",
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
    if stage_norm not in _valid_stages:
        raise click.UsageError(
            f"Invalid --stage {stage!r}. Expected 'all' or one of s1..s5."
        )

    fresh_stages: set[str] | None = None
    if stage_norm != "all":
        canonical_stage = normalize_stage_name(stage_norm)
        downstream_map = {
            "s1_grounding": {"s1_grounding"},
            "s2_macro": {"s2_macro", "s3_discover", "s4_verify"},
            "s3_discover": {"s3_discover", "s4_verify"},
            "s4_verify": {"s4_verify"},
            "s5_success": {"s5_success"},
        }
        fresh_stages = downstream_map.get(canonical_stage, {canonical_stage})

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    cfg_mgr = ConfigManager(config_dir=config_dir, workspace_dir=workspace)

    repo = Repository(workspace)
    dag: VlmDag | None = None
    try:
        metas: list[EpisodeMeta] = []
        if episodes:
            for ep_id in episodes:
                try:
                    metas.append(repo.get_episode_meta(ep_id))
                except NotFoundError as exc:
                    raise click.ClickException(str(exc)) from exc
        else:
            rows = repo.list_episodes(dataset=dataset, limit=10000)
            for r in rows:
                metas.append(repo.get_episode_meta(str(r["episode_id"])))

        if not metas:
            raise click.ClickException("No matching ingested episodes found in workspace.")

        dag = VlmDag(workspace=workspace, config=cfg_mgr, repo=repo)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        def _process_one(meta: EpisodeMeta) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
            assert dag is not None
            try:
                with _open_view_from_meta(meta, cfg_mgr) as view:
                    art = dag.run_episode(view, fresh=fresh, fresh_stages=fresh_stages)
                return {
                    "episode_id": meta.episode_id,
                    "run_id": art.run_id,
                    "verdict": art.success.verdict,
                    "passed": art.success.passed,
                    "qc_segments": len(art.qc_segments),
                    "total_tokens": art.usage.total_tokens,
                }, None
            except Exception as exc:
                return None, {"episode_id": meta.episode_id, "error": str(exc)}

        max_w = max(1, min(int(workers), len(metas)))
        if max_w == 1:
            outcomes = [_process_one(m) for m in metas]
        else:
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                outcomes = list(pool.map(_process_one, metas))

        for ok_res, err_res in outcomes:
            if ok_res is not None:
                results.append(ok_res)
            if err_res is not None:
                errors.append(err_res)
    finally:
        if dag is not None:
            try:
                dag.client.close()
            except Exception:
                pass
        repo.close()

    report = {
        "ok": len(errors) == 0,
        "workspace": str(workspace),
        "processed": len(results),
        "errors_count": len(errors),
        "results": results,
        "errors": errors,
    }
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        click.echo(
            f"run-vlm finished: processed={len(results)}, errors={len(errors)}"
        )
        for err in errors:
            click.echo(f"ERROR [{err['episode_id']}]: {err['error']}", err=True)

    if errors:
        sys.exit(1)


@main.command("render-mp4")
@click.option("--episode", default=None, help="Target episode_id.")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Output MP4 path or directory.")
@click.option("--all-ready", is_flag=True, help="Render all episodes with an active VLM run.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
@click.pass_context
def render_mp4_cmd(
    ctx: click.Context,
    episode: str | None,
    out: Path | None,
    all_ready: bool,
    as_json: bool,
) -> None:
    """Render 1080p Studio HUD visualization MP4 from ResolvedEpisode."""
    if not episode and not all_ready:
        raise click.UsageError("Specify --episode <episode_id> or --all-ready.")

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    cfg_mgr = ConfigManager(config_dir=config_dir, workspace_dir=workspace)
    repo = Repository(workspace)
    rendered: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    try:
        ep_ids: list[str] = []
        if episode:
            ep_ids = [episode]
        else:
            for r in repo.list_episodes(limit=10000):
                if r.get("active_run_id"):
                    ep_ids.append(str(r["episode_id"]))

        if not ep_ids:
            raise click.ClickException("No VLM-ready episodes found to render.")

        for ep_id in ep_ids:
            try:
                resolved = repo.resolve_episode(ep_id)
                if episode and out is not None:
                    out_p = Path(out)
                elif out is not None:
                    out_p = Path(out) / f"{resolved.meta.dataset}__{resolved.meta.stem}.mp4"
                else:
                    out_p = (
                        workspace
                        / "exports"
                        / "viz_mp4"
                        / f"{resolved.meta.dataset}__{resolved.meta.stem}.mp4"
                    )
                with _open_view_from_meta(resolved.meta, cfg_mgr) as view:
                    saved = render_viz_mp4(resolved, view, out_p, cfg=cfg_mgr)
                rendered.append({"episode_id": ep_id, "out_path": str(saved)})
            except Exception as exc:
                errors.append({"episode_id": ep_id, "error": str(exc)})
    finally:
        repo.close()

    report = {
        "ok": len(errors) == 0,
        "rendered_count": len(rendered),
        "rendered": rendered,
        "errors": errors,
    }
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in rendered:
            click.echo(f"Rendered {item['episode_id']} -> {item['out_path']}")
        for err in errors:
            click.echo(f"ERROR [{err['episode_id']}]: {err['error']}", err=True)

    if errors:
        sys.exit(1)


@main.command("export")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["lerobot_v3", "h5_clean", "zero_copy_index", "viz_mp4"]),
    default="lerobot_v3",
    show_default=True,
)
@click.option(
    "--recipe",
    "recipe_name",
    type=click.Choice(["bc_clean", "recovery_plus", "badcase_pairs"]),
    default="bc_clean",
    show_default=True,
)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--chunk-size", type=int, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--dataset", default=None, help="Filter by dataset.")
@click.option("--task", "task_id", default=None, help="Filter by task_id.")
@click.option("--episode", "episodes", multiple=True, help="Target episode_id (repeatable).")
@click.option(
    "--status",
    "statuses",
    type=click.Choice([s.value for s in EpisodeStatus]),
    multiple=True,
    help="Episode statuses to include (defaults to approved & gt_frozen).",
)
@click.option("--tail-frames", type=int, default=None)
@click.option(
    "--smooth-tail",
    type=click.Choice(["on_jump", "always", "never"]),
    default=None,
)
@click.option("--min-piece-s", type=float, default=None)
@click.option("--context-window-s", type=float, default=2.0, show_default=True)
@click.option(
    "--state-schema-mode",
    type=click.Choice(["auto", "preserve_native", "canonical_81d"]),
    default=None,
    help="LeRobot v3 state/action schema mode (defaults to config or auto).",
)
@click.option(
    "--split-by-subtask",
    is_flag=True,
    default=False,
    help="Split each kept piece along aligned subtask boundaries into independent episodes.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
@click.pass_context
def export_cmd(
    ctx: click.Context,
    fmt: str,
    recipe_name: str,
    out: Path,
    chunk_size: int | None,
    dry_run: bool,
    dataset: str | None,
    task_id: str | None,
    episodes: tuple[str, ...],
    statuses: tuple[str, ...],
    tail_frames: int | None,
    smooth_tail: str | None,
    min_piece_s: float | None,
    context_window_s: float,
    state_schema_mode: str | None,
    split_by_subtask: bool,
    as_json: bool,
) -> None:
    """Export curated slices using PieceSlicer and transactional exporters."""
    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    cfg_mgr = ConfigManager(config_dir=config_dir, workspace_dir=workspace)
    exp_cfg = cfg_mgr.settings.get("export", {})

    eff_chunk_size = chunk_size if chunk_size is not None else exp_cfg.get("chunk_size")
    eff_min_piece_s = (
        float(min_piece_s)
        if min_piece_s is not None
        else float(exp_cfg.get("min_piece_s", 0.1))
    )
    eff_tail_frames = (
        tail_frames if tail_frames is not None else exp_cfg.get("tail_frames")
    )
    eff_smooth_tail = (
        smooth_tail
        if smooth_tail is not None
        else str(exp_cfg.get("smooth_tail", "on_jump"))
    )
    eff_schema_mode = (
        state_schema_mode
        if state_schema_mode is not None
        else str(exp_cfg.get("state_schema_mode", "auto"))
    )
    eff_statuses = (
        [EpisodeStatus(s) for s in statuses]
        if statuses
        else [EpisodeStatus.APPROVED, EpisodeStatus.GT_FROZEN]
    )

    try:
        recipe = ExportRecipe(
            name=recipe_name,  # type: ignore[arg-type]
            format=fmt,  # type: ignore[arg-type]
            chunk_size=eff_chunk_size,
            min_piece_s=eff_min_piece_s,
            tail_frames=eff_tail_frames,
            smooth_tail=eff_smooth_tail,  # type: ignore[arg-type]
            include_statuses=eff_statuses,
            context_window_s=context_window_s,
            dataset_filter=dataset,
            task_filter=task_id,
            state_schema_mode=eff_schema_mode,  # type: ignore[arg-type]
            split_by_subtask=bool(split_by_subtask),
        )
    except Exception as exc:
        click.echo(f"Invalid export parameters: {exc}", err=True)
        sys.exit(2)

    repo = Repository(workspace)
    slicer = PieceSlicer(cfg_mgr)
    allowed_status_set = set(eff_statuses)

    resolved_list: list[ResolvedEpisode] = []
    views_by_ep: dict[str, EpisodeView] = {}
    all_pieces: list[KeepPiece] = []
    errors: list[dict[str, str]] = []
    manifest: ExportManifest | None = None

    with contextlib.ExitStack() as stack:
        try:
            candidate_rows: list[dict[str, Any]] = []
            if episodes:
                for ep_id in episodes:
                    try:
                        candidate_rows.append(repo.get_episode_row(ep_id))
                    except Exception as exc:
                        errors.append({"episode_id": ep_id, "error": str(exc)})
            else:
                candidate_rows = repo.list_episodes(
                    dataset=dataset, task_id=task_id, limit=100000
                )

            for row in candidate_rows:
                ep_id = str(row["episode_id"])
                if EpisodeStatus(str(row["status"])) not in allowed_status_set:
                    continue
                if not row.get("active_run_id"):
                    continue
                try:
                    resolved = repo.resolve_episode(ep_id)
                    view = stack.enter_context(_open_view_from_meta(resolved.meta, cfg_mgr))
                    ep_pieces = slicer.slice(resolved, view, recipe)
                    resolved_list.append(resolved)
                    views_by_ep[ep_id] = view
                    all_pieces.extend(ep_pieces)
                except Exception as exc:
                    errors.append({"episode_id": ep_id, "error": str(exc)})

            if dry_run:
                (
                    _,
                    entries,
                    ep_inc,
                    ep_skp,
                    p_exp,
                    p_drop,
                    tot_f,
                ) = group_kept_pieces(all_pieces, resolved_list, recipe)
                manifest = ExportManifest(
                    export_id="dry_run",
                    created_at=0.0,
                    format=recipe.format,
                    recipe=recipe,
                    output_dir=str(Path(out).resolve()),
                    episodes_included=ep_inc,
                    episodes_skipped=ep_skp,
                    pieces_exported=p_exp,
                    pieces_dropped=p_drop,
                    total_frames=tot_f,
                    episodes=entries,
                )
            elif errors and not resolved_list:
                pass
            else:
                base_exporter = get_exporter(fmt)
                try:
                    exporter = type(base_exporter)(cfg=cfg_mgr)
                except TypeError:
                    exporter = base_exporter
                try:
                    manifest = exporter.export(
                        all_pieces, resolved_list, views_by_ep, out, recipe
                    )
                except Exception as exc:
                    errors.append({"episode_id": "*", "error": str(exc)})
        finally:
            repo.close()

    if errors and manifest is None:
        if as_json:
            click.echo(
                json.dumps(
                    {"ok": False, "errors": errors},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for err in errors:
                click.echo(f"ERROR [{err['episode_id']}]: {err['error']}", err=True)
        sys.exit(1)

    assert manifest is not None
    if as_json:
        payload = manifest.model_dump(mode="json")
        payload["ok"] = len(errors) == 0
        payload["dry_run"] = dry_run
        payload["errors"] = errors
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        console = Console()
        table = Table(
            title=f"robo_vlm_qc export ({'dry-run' if dry_run else fmt} / {recipe_name})"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Output dir", str(manifest.output_dir))
        table.add_row("Episodes included", str(manifest.episodes_included))
        table.add_row("Episodes skipped", str(manifest.episodes_skipped))
        table.add_row("Pieces exported", str(manifest.pieces_exported))
        table.add_row("Pieces dropped", str(manifest.pieces_dropped))
        table.add_row("Total frames", str(manifest.total_frames))
        table.add_row("Artifacts", str(len(manifest.artifacts)))
        console.print(table)
        for err in errors:
            click.echo(f"ERROR [{err['episode_id']}]: {err['error']}", err=True)

    if errors:
        sys.exit(1)


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--reviewer-token", default=None)
@click.option("--lead-token", default=None)
@click.option("--no-auth", is_flag=True)
@click.option(
    "--allowed-data-roots",
    "allowed_data_roots",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    multiple=True,
    default=(),
    help="Allowed read-only root directory for Web ingest/browse (repeatable).",
)
@click.option(
    "--cors-origin",
    "cors_origins",
    type=str,
    multiple=True,
    default=(),
    help="Allowed CORS origin (repeatable; overrides ROBO_VLM_QC_CORS_ORIGINS).",
)
@click.pass_context
def serve_cmd(
    ctx: click.Context,
    host: str,
    port: int,
    reviewer_token: str | None,
    lead_token: str | None,
    no_auth: bool,
    allowed_data_roots: tuple[Path, ...],
    cors_origins: tuple[str, ...],
) -> None:
    """Start FastAPI + React Dual-Role Studio server."""
    if port < 1 or port > 65535:
        raise click.BadParameter(
            f"Port must be between 1 and 65535, got {port}",
            param_hint="'--port'",
        )

    import os

    if (
        not no_auth
        and reviewer_token is None
        and lead_token is None
        and not os.environ.get("ROBO_VLM_QC_REVIEWER_TOKEN")
        and not os.environ.get("ROBO_VLM_QC_LEAD_TOKEN")
    ):
        raise click.UsageError(
            "Provide --reviewer-token / --lead-token (or ROBO_VLM_QC_*_TOKEN env vars), "
            "or pass --no-auth for local unauthenticated access."
        )

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    if workspace.exists() and not workspace.is_dir():
        raise click.ClickException(f"Workspace path is not a directory: {workspace}")

    import uvicorn

    from .api import create_app

    app = create_app(
        workspace=workspace,
        config_dir=config_dir,
        reviewer_token=reviewer_token,
        lead_token=lead_token,
        no_auth=no_auth,
        allowed_data_roots=([Path(p) for p in allowed_data_roots] if allowed_data_roots else None),
        cors_origins=(list(cors_origins) if cors_origins else None),
        recover_stale_jobs=True,
    )
    uvicorn.run(app, host=host, port=port)


@main.command("eval")
@click.option("--dataset", default=None, help="Filter GT episodes by dataset.")
@click.option("--task", "task_id", default=None, help="Filter GT episodes by task_id.")
@click.option("--prompt-ver", default=None, help="Candidate prompt version tag for sandbox run.")
@click.option("--label-ver", default=None, help="Candidate label version tag for sandbox run.")
@click.option("--few-shot/--no-few-shot", default=True, show_default=True, help="Enable Few-Shot retrieval.")
@click.option("--episode", "episodes", multiple=True, help="Target gt_frozen episode_id (repeatable).")
@click.option("--workers", type=int, default=4, show_default=True, help="Episode eval concurrency.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON eval report.")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    dataset: str | None,
    task_id: str | None,
    prompt_ver: str | None,
    label_ver: str | None,
    few_shot: bool,
    episodes: tuple[str, ...],
    workers: int,
    as_json: bool,
) -> None:
    """Run VLM Lab GT regression eval sandbox."""
    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    db_path = workspace / "robo_vlm_qc.db"
    if not db_path.is_file():
        msg = f"Workspace database not found: {db_path}"
        if as_json:
            click.echo(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
        else:
            click.echo(msg, err=True)
        sys.exit(2)

    from .lab.eval_runner import run_gt_eval

    repo = Repository(workspace)
    try:
        report = run_gt_eval(
            repo,
            config_dir=config_dir,
            dataset=dataset,
            task_id=task_id,
            episode_ids=list(episodes) if episodes else None,
            few_shot=few_shot,
            workers=workers,
            prompt_ver=prompt_ver,
            label_ver=label_ver,
        )
    finally:
        repo.close()

    if report.metrics.episodes_evaluated == 0:
        msg = "No gt_frozen episodes found in workspace for GT regression evaluation."
        if as_json:
            payload = report.model_dump(mode="json")
            payload["ok"] = False
            payload["error"] = msg
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            click.echo(msg, err=True)
        sys.exit(2)

    if as_json:
        payload = report.model_dump(mode="json")
        payload["ok"] = True
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        m = report.metrics
        console = Console()
        table = Table(title=f"robo_vlm_qc eval ({report.eval_id})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("episodes_evaluated", str(m.episodes_evaluated))
        table.add_row("f1", f"{m.f1:.4f}")
        table.add_row("precision", f"{m.precision:.4f}")
        table.add_row("recall", f"{m.recall:.4f}")
        table.add_row("mean_iou", f"{m.mean_iou:.4f}")
        table.add_row("overcut_rate", f"{m.overcut_rate:.4f}")
        table.add_row("guardrail_downgrade_rate", f"{m.guardrail_downgrade_rate:.4f}")
        table.add_row("success_accuracy", f"{m.success_accuracy:.4f}")
        table.add_row("total_tokens", str(m.total_tokens))
        table.add_row("cache_hit_rate", f"{m.cache_hit_rate:.4f}")
        console.print(table)


@main.command("config-vlm")
@click.option(
    "--preset",
    type=click.Choice(["openai", "dashscope", "gemini", "vllm", "custom"], case_sensitive=False),
    default=None,
    help="Apply a built-in VLM preset (openai|dashscope|gemini|vllm|custom).",
)
@click.option(
    "--provider",
    type=click.Choice(["openai_compat", "gemini"], case_sensitive=False),
    default=None,
    help="VLM API provider protocol.",
)
@click.option("--base-url", default=None, help="VLM API base URL (http:// or https://).")
@click.option(
    "--api-key",
    default=None,
    help="API key to save securely into <workspace>/.env (0600); never written to YAML.",
)
@click.option(
    "--api-key-env",
    default=None,
    help="Environment variable name used to look up the API key (e.g. OPENAI_API_KEY).",
)
@click.option(
    "--model",
    default=None,
    help="Unified model name applied across all S1-S5 stages.",
)
@click.option(
    "--thinking-mode",
    type=click.Choice(["auto", "dashscope", "disabled"], case_sensitive=False),
    default=None,
    help="Reasoning/thinking request parameter mode.",
)
@click.option(
    "--test",
    "run_test",
    is_flag=True,
    default=False,
    help="Send a minimal connectivity probe to verify the configured VLM endpoint.",
)
@click.option("--json", "as_json", is_flag=True, help="Print sanitized JSON summary.")
@click.pass_context
def config_vlm_cmd(
    ctx: click.Context,
    preset: str | None,
    provider: str | None,
    base_url: str | None,
    api_key: str | None,
    api_key_env: str | None,
    model: str | None,
    thinking_mode: str | None,
    run_test: bool,
    as_json: bool,
) -> None:
    """Configure workspace VLM API provider/model/key (safe YAML + 0600 .env) and optionally probe connectivity."""
    from .api.routers.settings import (
        apply_vlm_settings_update,
        load_vlm_presets,
        probe_vlm_connection,
        read_vlm_settings_dict,
    )
    from .config import default_config_dir

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    eff_cfg_dir = Path(config_dir).resolve() if config_dir is not None else default_config_dir()

    eff_provider = provider.lower() if provider else None
    eff_base_url = base_url
    eff_api_key_env = api_key_env
    eff_thinking_mode = thinking_mode.lower() if thinking_mode else None
    eff_default_model = model
    eff_stage_models: dict[str, str] | None = None

    if preset is not None:
        preset_key = preset.strip().lower()
        presets_by_id = {p["id"]: p for p in load_vlm_presets(eff_cfg_dir)}
        p_item = presets_by_id.get(preset_key)
        if p_item is None:
            raise click.BadParameter(f"Unknown preset: {preset!r}", param_hint="'--preset'")
        if eff_provider is None:
            eff_provider = str(p_item["provider"])
        if eff_base_url is None and str(p_item.get("base_url") or "").strip():
            eff_base_url = str(p_item["base_url"])
        if eff_api_key_env is None and str(p_item.get("api_key_env") or "").strip():
            eff_api_key_env = str(p_item["api_key_env"])
        if eff_thinking_mode is None:
            eff_thinking_mode = str(p_item["thinking_mode"])
        if eff_default_model is None:
            p_sm = {
                k: str(v)
                for k, v in (p_item.get("stage_models") or {}).items()
                if str(v).strip()
            }
            if p_sm:
                eff_stage_models = p_sm

    has_updates = any(
        x is not None
        for x in (
            eff_provider,
            eff_base_url,
            eff_api_key_env,
            eff_thinking_mode,
            eff_default_model,
            eff_stage_models,
        )
    ) or bool(api_key and api_key.strip())

    try:
        if has_updates:
            cfg_mgr = apply_vlm_settings_update(
                workspace=workspace,
                config_dir=eff_cfg_dir,
                provider=eff_provider,  # type: ignore[arg-type]
                base_url=eff_base_url,
                api_key_env=eff_api_key_env,
                api_key=api_key,
                thinking_mode=eff_thinking_mode,  # type: ignore[arg-type]
                default_model=eff_default_model,
                stage_models=eff_stage_models,
                reason="cli_config_vlm",
            )
        else:
            cfg_mgr = ConfigManager(config_dir=eff_cfg_dir, workspace_dir=workspace)
    except ValueError as exc:
        if as_json:
            click.echo(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            click.echo(f"Invalid VLM configuration: {exc}", err=True)
        sys.exit(2)

    sanitized = read_vlm_settings_dict(
        cfg_mgr,
        workspace=workspace,
        config_dir=eff_cfg_dir,
    )
    test_result: dict[str, Any] | None = None
    if run_test:
        test_result = probe_vlm_connection(
            cfg=cfg_mgr,
            workspace=workspace,
            model=eff_default_model,
        )

    ok_flag = True if test_result is None else bool(test_result["ok"])
    report: dict[str, Any] = {
        "ok": ok_flag,
        "workspace": str(workspace),
        "updated": has_updates,
        "provider": sanitized["provider"],
        "base_url": sanitized["base_url"],
        "api_key_env": sanitized["api_key_env"],
        "api_key_configured": sanitized["api_key_configured"],
        "api_key_masked": sanitized["api_key_masked"],
        "thinking_mode": sanitized["thinking_mode"],
        "stage_models": sanitized["stage_models"],
    }
    if test_result is not None:
        report["test"] = test_result

    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        console = Console()
        table = Table(title="robo_vlm_qc VLM configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("provider", str(report["provider"]))
        table.add_row("base_url", str(report["base_url"]))
        table.add_row("api_key_env", str(report["api_key_env"]))
        table.add_row(
            "api_key",
            report["api_key_masked"] if report["api_key_configured"] else "(not configured)",
        )
        table.add_row("thinking_mode", str(report["thinking_mode"]))
        for st_key, st_model in report["stage_models"].items():
            table.add_row(f"model.{st_key}", str(st_model))
        if test_result is not None:
            status_str = (
                f"OK ({test_result['latency_ms']} ms)"
                if test_result["ok"]
                else f"FAILED ({test_result['detail']})"
            )
            table.add_row("connectivity_test", status_str)
        console.print(table)

    if not ok_flag:
        sys.exit(1)


@main.command("gc")
@click.option(
    "--max-cache-gb",
    type=float,
    default=None,
    help="Target maximum preview cache size in GB (defaults to io.preview_cache.max_cache_gb).",
)
@click.option(
    "--clear-all",
    is_flag=True,
    default=False,
    help="Evict all cached preview MP4 files and stale partial files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report files that would be evicted without deleting anything.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON summary.")
@click.pass_context
def gc_cmd(
    ctx: click.Context,
    max_cache_gb: float | None,
    clear_all: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Run LRU garbage collection on workspace MP4 preview cache."""
    if max_cache_gb is not None and max_cache_gb < 0:
        raise click.BadParameter(
            f"--max-cache-gb must be >= 0, got {max_cache_gb}",
            param_hint="'--max-cache-gb'",
        )

    from .io.media import PreviewCache

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    cfg_ws = workspace if (workspace / "configs_override").is_dir() else None
    cfg_mgr = ConfigManager(config_dir=config_dir, workspace_dir=cfg_ws)

    raw_cfg_gb = (
        cfg_mgr.settings.get("io", {})
        .get("preview_cache", {})
        .get("max_cache_gb", 10.0)
    )
    eff_max_gb = (
        0.0
        if clear_all
        else (
            float(max_cache_gb)
            if max_cache_gb is not None
            else (float(raw_cfg_gb) if raw_cfg_gb is not None else None)
        )
    )
    target_bytes = int(eff_max_gb * (1024**3)) if eff_max_gb is not None else None

    ws_resolved = Path(workspace).resolve()
    previews_root = ws_resolved / "artifacts" / "previews"
    if dry_run and not previews_root.is_dir():
        result: dict[str, Any] = {
            "dry_run": True,
            "cache_dir": str(previews_root),
            "max_bytes": target_bytes,
            "before_bytes": 0,
            "after_bytes": 0,
            "freed_bytes": 0,
            "before_files": 0,
            "after_files": 0,
            "evicted_count": 0,
            "evicted_files": [],
            "stale_partials_removed": 0,
            "stale_partials": [],
            "empty_dirs_removed": 0,
        }
    else:
        cache = PreviewCache(ws_resolved, max_cache_gb=eff_max_gb)
        result = cache.evict_lru(max_bytes=target_bytes, dry_run=dry_run)

    report = {
        "ok": True,
        "workspace": str(ws_resolved),
        "clear_all": bool(clear_all),
        "max_cache_gb": eff_max_gb,
        **result,
    }
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        console = Console()
        table = Table(
            title=f"robo_vlm_qc preview cache GC ({'dry-run' if dry_run else 'applied'})"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Cache directory", str(report["cache_dir"]))
        table.add_row("Files before -> after", f"{report['before_files']} -> {report['after_files']}")
        table.add_row(
            "Size before -> after (MB)",
            f"{report['before_bytes'] / (1024 ** 2):.2f} -> {report['after_bytes'] / (1024 ** 2):.2f}",
        )
        table.add_row("Freed (MB)", f"{report['freed_bytes'] / (1024 ** 2):.2f}")
        table.add_row("Evicted previews", str(report["evicted_count"]))
        table.add_row("Stale partials removed", str(report["stale_partials_removed"]))
        table.add_row("Empty dirs removed", str(report["empty_dirs_removed"]))
        console.print(table)


@main.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Print JSON diagnostics report.")
@click.pass_context
def doctor_cmd(ctx: click.Context, as_json: bool) -> None:
    """Run environment, media toolchain, workspace DB, web UI, and VLM config diagnostics."""
    import os
    import shutil
    import sqlite3
    import subprocess

    from .api.routers.settings import read_vlm_settings_dict
    from .api.state import resolve_default_web_dist
    from .config import default_config_dir
    from .io.media import _is_nvenc_available

    workspace: Path = ctx.obj["workspace"]
    config_dir: Path | None = ctx.obj["config_dir"]
    ws_resolved = Path(workspace).resolve()
    eff_cfg_dir = (
        Path(config_dir).resolve() if config_dir is not None else default_config_dir()
    )

    checks: list[dict[str, Any]] = []

    def _add_check(
        name: str,
        status: str,
        detail: str,
        *,
        critical: bool,
        meta: dict[str, Any] | None = None,
    ) -> None:
        checks.append(
            {
                "name": name,
                "status": status,
                "critical": critical,
                "detail": detail,
                "meta": meta or {},
            }
        )

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    _add_check(
        "python",
        "ok" if py_ok else "fail",
        f"Python {py_ver}" + ("" if py_ok else " (requires >= 3.10)"),
        critical=True,
        meta={"version": py_ver},
    )

    # 2. OpenCV
    try:
        import cv2

        cv_ver = str(getattr(cv2, "__version__", "unknown"))
        _add_check(
            "opencv",
            "ok",
            f"OpenCV {cv_ver}",
            critical=True,
            meta={"version": cv_ver},
        )
    except Exception as exc:
        _add_check(
            "opencv",
            "fail",
            f"OpenCV unavailable: {exc}",
            critical=True,
        )

    # 3 & 4. ffmpeg & ffprobe
    for tool_name in ("ffmpeg", "ffprobe"):
        bin_path = shutil.which(tool_name)
        if not bin_path:
            _add_check(
                tool_name,
                "fail",
                f"{tool_name} not found in PATH",
                critical=True,
            )
        else:
            ver_line = ""
            try:
                proc = subprocess.run(
                    [bin_path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout:
                    ver_line = proc.stdout.splitlines()[0].strip()
            except Exception:
                ver_line = ""
            _add_check(
                tool_name,
                "ok",
                ver_line or f"Found at {bin_path}",
                critical=True,
                meta={"path": bin_path, "version_line": ver_line},
            )

    # 5. NVENC hardware acceleration (optional)
    nvenc_ok = bool(_is_nvenc_available())
    _add_check(
        "nvenc",
        "ok" if nvenc_ok else "warn",
        "h264_nvenc available" if nvenc_ok else "h264_nvenc unavailable (will use libx264 CPU fallback)",
        critical=False,
        meta={"available": nvenc_ok},
    )

    # 6. Workspace writability & free disk space
    writable = False
    free_disk_gb: float | None = None
    ws_detail = ""
    if ws_resolved.exists() and not ws_resolved.is_dir():
        ws_detail = f"Workspace path is not a directory: {ws_resolved}"
    else:
        probe_dir = ws_resolved
        if not probe_dir.exists():
            cur = probe_dir
            while not cur.exists() and cur != cur.parent:
                cur = cur.parent
            probe_dir = cur
        if probe_dir.is_dir():
            try:
                usage = shutil.disk_usage(probe_dir)
                free_disk_gb = round(usage.free / float(1024**3), 2)
            except OSError:
                free_disk_gb = None
            if ws_resolved.is_dir():
                probe_file = ws_resolved / f".doctor_write_probe_{os.getpid()}"
                try:
                    probe_file.write_text("ok", encoding="utf-8")
                    writable = True
                except OSError as exc:
                    ws_detail = f"Workspace not writable: {exc}"
                finally:
                    with contextlib.suppress(OSError):
                        probe_file.unlink()
            else:
                writable = os.access(probe_dir, os.W_OK)
                if not writable:
                    ws_detail = f"Parent directory not writable: {probe_dir}"
        else:
            ws_detail = f"Cannot resolve writable parent for {ws_resolved}"

    if not writable:
        _add_check(
            "workspace",
            "fail",
            ws_detail or f"Workspace not writable: {ws_resolved}",
            critical=True,
            meta={
                "path": str(ws_resolved),
                "writable": False,
                "free_disk_gb": free_disk_gb,
            },
        )
    else:
        low_disk = free_disk_gb is not None and free_disk_gb < 1.0
        disk_str = f"{free_disk_gb:.2f} GB free" if free_disk_gb is not None else "disk usage unknown"
        _add_check(
            "workspace",
            "warn" if low_disk else "ok",
            f"Writable ({disk_str})",
            critical=True,
            meta={
                "path": str(ws_resolved),
                "writable": True,
                "free_disk_gb": free_disk_gb,
            },
        )

    # 7. SQLite WAL DB health & episode count
    db_path = ws_resolved / "robo_vlm_qc.db"
    db_exists = db_path.is_file()
    journal_mode: str | None = None
    episode_count: int | None = None
    if not db_exists:
        _add_check(
            "sqlite_db",
            "warn",
            f"Database not initialized yet at {db_path}",
            critical=False,
            meta={"exists": False, "path": str(db_path), "episode_count": 0},
        )
    else:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            jm_row = conn.execute("PRAGMA journal_mode;").fetchone()
            journal_mode = str(jm_row[0]).lower() if jm_row else None
            qc_row = conn.execute("PRAGMA quick_check;").fetchone()
            integrity_ok = bool(qc_row and str(qc_row[0]).lower() == "ok")
            cnt_row = conn.execute("SELECT COUNT(*) FROM episodes;").fetchone()
            episode_count = int(cnt_row[0]) if cnt_row else 0
            if not integrity_ok:
                _add_check(
                    "sqlite_db",
                    "fail",
                    f"SQLite integrity check failed: {qc_row[0] if qc_row else 'unknown'}",
                    critical=True,
                    meta={
                        "exists": True,
                        "path": str(db_path),
                        "journal_mode": journal_mode,
                        "episode_count": episode_count,
                    },
                )
            else:
                is_wal = journal_mode == "wal"
                _add_check(
                    "sqlite_db",
                    "ok" if is_wal else "warn",
                    f"Healthy ({journal_mode or 'unknown'} mode, {episode_count} episodes)",
                    critical=True,
                    meta={
                        "exists": True,
                        "path": str(db_path),
                        "journal_mode": journal_mode,
                        "episode_count": episode_count,
                    },
                )
        except Exception as exc:
            _add_check(
                "sqlite_db",
                "fail",
                f"SQLite database error: {exc}",
                critical=True,
                meta={"exists": True, "path": str(db_path)},
            )
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    # 8. web/dist availability
    web_dist = resolve_default_web_dist(None)
    web_built = (web_dist / "index.html").is_file()
    _add_check(
        "web_dist",
        "ok" if web_built else "warn",
        f"Built SPA found at {web_dist}"
        if web_built
        else f"SPA bundle not found at {web_dist} (run npm run build in web/)",
        critical=False,
        meta={"path": str(web_dist), "available": web_built},
    )

    # 9. Configured VLM provider / base_url / masked API key presence (NEVER plaintext)
    vlm_info: dict[str, Any] = {
        "provider": None,
        "base_url": None,
        "api_key_env": None,
        "api_key_configured": False,
        "api_key_masked": "",
    }
    try:
        cfg_ws = ws_resolved if (ws_resolved / "configs_override").is_dir() else None
        cfg_mgr = ConfigManager(config_dir=eff_cfg_dir, workspace_dir=cfg_ws)
        sanitized = read_vlm_settings_dict(
            cfg_mgr,
            workspace=ws_resolved,
            config_dir=eff_cfg_dir,
        )
        vlm_info = {
            "provider": sanitized["provider"],
            "base_url": sanitized["base_url"],
            "api_key_env": sanitized["api_key_env"],
            "api_key_configured": bool(sanitized["api_key_configured"]),
            "api_key_masked": str(sanitized["api_key_masked"]),
        }
        key_state = (
            f"key={vlm_info['api_key_masked']}"
            if vlm_info["api_key_configured"]
            else f"{vlm_info['api_key_env']} not set"
        )
        _add_check(
            "vlm_config",
            "ok" if vlm_info["api_key_configured"] else "warn",
            f"{vlm_info['provider']} @ {vlm_info['base_url']} ({key_state})",
            critical=False,
            meta=vlm_info,
        )
    except Exception as exc:
        _add_check(
            "vlm_config",
            "fail",
            f"Failed to load VLM configuration: {exc}",
            critical=True,
        )

    ok_count = sum(1 for c in checks if c["status"] == "ok")
    warn_count = sum(1 for c in checks if c["status"] == "warn")
    fail_count = sum(1 for c in checks if c["status"] == "fail")
    critical_ok = all(not (c["critical"] and c["status"] == "fail") for c in checks) and fail_count == 0

    report: dict[str, Any] = {
        "ok": critical_ok,
        "workspace": str(ws_resolved),
        "checks": checks,
        "summary": {
            "ok": ok_count,
            "warn": warn_count,
            "fail": fail_count,
        },
        "vlm": vlm_info,
        "workspace_info": {
            "writable": writable,
            "free_disk_gb": free_disk_gb,
            "db_exists": db_exists,
            "journal_mode": journal_mode,
            "episode_count": episode_count if episode_count is not None else 0,
            "web_dist_available": web_built,
        },
    }

    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        console = Console()
        table = Table(title="robo_vlm_qc doctor diagnostics")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        for c in checks:
            st = c["status"]
            style = "green" if st == "ok" else ("yellow" if st == "warn" else "red")
            table.add_row(c["name"], f"[{style}]{st.upper()}[/{style}]", c["detail"])
        console.print(table)

    if not critical_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
