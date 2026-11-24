"""3-level concurrent causal VLM DAG: S0 -> S1 -> (S2 || S5) -> S3 -> S4A -> S4B -> Guardrail -> Store."""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import ConfigManager, canonical_sha256
from ..guardrail import (
    apply_hallucination_check,
    apply_splice_gate,
    apply_visual_gate,
)
from ..io.base import EpisodeView
from ..models.annotation import (
    ArmTimelines,
    ConfigFingerprintSnapshot,
    GroundingResult,
    MacroStage,
    QcSegment,
    StageRefs,
    SuccessVerdict,
    VlmRunArtifact,
)
from ..models.review import EpisodeStatus
from ..models.taxonomy import FewShotItem, TaskSpec
from ..store.repository import GtFrozenError, NotFoundError, Repository
from .cache import StageCache
from .client import (
    BudgetTracker,
    VlmCancelledError,
    VlmClient,
    check_client_cancelled,
    normalize_stage_name,
)
from .prompt_builder import PromptBuilder
from .stage1_grounding import run_stage1_grounding
from .stage2_macro import run_stage2_macro
from .stage3_discover import run_stage3_discover
from .stage4_verify import run_stage4_verify
from .stage5_success import run_stage5_success


class VlmDag:
    """Orchestrates the 5-stage causal VLM pipeline, stage caching, guardrails, and persistence."""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        workspace_dir: Path | None = None,
        config: ConfigManager | None = None,
        cfg: ConfigManager | None = None,
        config_manager: ConfigManager | None = None,
        client: VlmClient | None = None,
        vlm_client: VlmClient | None = None,
        cache: StageCache | None = None,
        stage_cache: StageCache | None = None,
        repo: Repository | None = None,
        repository: Repository | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        ws = workspace or workspace_dir
        cm = config_manager or config or cfg
        if ws is None:
            if cm is not None and cm.workspace_dir is not None:
                ws = cm.workspace_dir
            else:
                ws = Path("./workspace")
        self.workspace = Path(ws).resolve()
        self.cfg = cm or ConfigManager(workspace_dir=self.workspace)
        self._cancel_event = cancel_event
        self.client = (
            client
            or vlm_client
            or VlmClient.from_config(
                self.cfg,
                workspace_dir=self.workspace,
                cancel_event=cancel_event,
            )
        )
        self.cache = cache or stage_cache or StageCache(workspace=self.workspace)
        self.repo = repo or repository or Repository(self.workspace)
        self.prompt_builder = PromptBuilder(self.cfg)

    def _resolve_task(self, view: EpisodeView, task_spec: TaskSpec | None) -> TaskSpec:
        if task_spec is not None:
            return task_spec
        if view.meta.task_id and view.meta.task_id != "_default":
            return self.cfg.get_task(view.meta.task_id)
        hint = view.meta.extra_meta.get("h5_task_name") or view.meta.extra_meta.get("ep_task_str")
        return self.cfg.resolve_task(view.meta.dataset, str(hint) if hint else None)

    def run_episode(
        self,
        view: EpisodeView | None = None,
        *,
        episode_view: EpisodeView | None = None,
        task_spec: TaskSpec | None = None,
        task: TaskSpec | None = None,
        few_shots: Sequence[FewShotItem] | None = None,
        few_shot_items: Sequence[FewShotItem] | None = None,
        fresh: bool = False,
        fresh_stages: Sequence[str] | set[str] | None = None,
        is_eval_sandbox: bool = False,
        persist: bool = True,
        cancel_event: threading.Event | None = None,
        on_status_change: Callable[[str, EpisodeStatus], None] | None = None,
        can_rollback: Callable[[str], bool] | None = None,
    ) -> VlmRunArtifact:
        ev = view or episode_view
        assert ev is not None, "EpisodeView is required"

        def _check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise VlmCancelledError(
                    f"VLM run cancelled for episode {ev.meta.episode_id!r}"
                )
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise VlmCancelledError(
                    f"VLM run cancelled for episode {ev.meta.episode_id!r}"
                )
            check_client_cancelled(self.client)

        def _emit_status(st: EpisodeStatus) -> None:
            if on_status_change is not None:
                with contextlib.suppress(Exception):
                    on_status_change(ev.meta.episode_id, st)

        t_spec = self._resolve_task(ev, task_spec or task)
        fs_stages = (
            {normalize_stage_name(str(s)) for s in fresh_stages}
            if fresh_stages
            else set()
        )
        if "s2_macro" in fs_stages:
            fs_stages |= {"s3_discover", "s4_verify"}
        if "s3_discover" in fs_stages:
            fs_stages |= {"s4_verify"}

        fs_raw = few_shots if few_shots is not None else few_shot_items
        if fs_raw is None:
            fs_list = self.repo.list_few_shots(task_id=t_spec.task_id, enabled_only=True)
        else:
            fs_list = [fs for fs in fs_raw if fs.enabled]

        if fs_list:
            few_shot_hash = canonical_sha256(
                [fs.model_dump(mode="json") for fs in fs_list]
            )
        else:
            few_shot_hash = "none"

        ep_budget = BudgetTracker(
            max_tokens=self.cfg.resolve_episode_budget_tokens(ev.duration_s),
            parent=self.client.budget,
        )

        # Manage episode status transition safely if tracked in SQLite
        transitioned_to_running = False
        prev_status: EpisodeStatus | None = None

        try:
            _check_cancelled()
            if persist and not is_eval_sandbox:
                try:
                    row = self.repo.get_episode_row(ev.meta.episode_id)
                    cur_st = EpisodeStatus(row["status"])
                    if cur_st == EpisodeStatus.GT_FROZEN:
                        raise GtFrozenError(
                            f"Episode {ev.meta.episode_id!r} is gt_frozen; cannot overwrite active VLM run"
                        )
                    if cur_st in {
                        EpisodeStatus.INGESTED,
                        EpisodeStatus.VLM_QUEUED,
                        EpisodeStatus.VLM_READY,
                    }:
                        prev_status = cur_st
                        _check_cancelled()
                        self.repo.transition_status(
                            ev.meta.episode_id, EpisodeStatus.VLM_RUNNING
                        )
                        transitioned_to_running = True
                        _emit_status(EpisodeStatus.VLM_RUNNING)
                except NotFoundError:
                    pass

            _check_cancelled()
            ep_fp = ev.meta.episode_fingerprint
            s1_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                stage_basis=self.cfg.s1_cache_basis(t_spec.task_id),
            )
            s2_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                stage_basis=self.cfg.s2_cache_basis(t_spec.task_id),
            )
            s5_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                stage_basis=self.cfg.s5_cache_basis(t_spec.task_id),
            )

            # Level 1: Stage 1 (2D Grounding)
            _check_cancelled()
            s1_cached = None if (fresh or "s1_grounding" in fs_stages) else self.cache.get(s1_key)
            if s1_cached is not None:
                grounding = GroundingResult.model_validate(s1_cached)
                ep_budget.record(
                    stage="s1_grounding",
                    model=self.client.model_for_stage("s1_grounding"),
                    cache_hit=True,
                )
            else:
                grounding = run_stage1_grounding(
                    view=ev,
                    task=t_spec,
                    client=self.client,
                    cfg=self.cfg,
                    prompt_builder=self.prompt_builder,
                    budget=ep_budget,
                    cancel_event=cancel_event or self._cancel_event,
                )
                _check_cancelled()
                self.cache.put(s1_key, grounding.model_dump(mode="json"))

            # Level 2: Stage 2 (Macro Backbone) || Stage 5 (Dense-Tail Success + 30% Control)
            _check_cancelled()

            def _exec_s2() -> tuple[list[MacroStage], ArmTimelines]:
                _check_cancelled()
                s2_cached = None if (fresh or "s2_macro" in fs_stages) else self.cache.get(s2_key)
                if s2_cached is not None:
                    stages = [
                        MacroStage.model_validate(x)
                        for x in s2_cached.get("macro_stages", [])
                    ]
                    arms = ArmTimelines.model_validate(
                        s2_cached.get("arm_timelines", {})
                    )
                    ep_budget.record(
                        stage="s2_macro",
                        model=self.client.model_for_stage("s2_macro"),
                        cache_hit=True,
                    )
                    return stages, arms
                stages, arms = run_stage2_macro(
                    view=ev,
                    task=t_spec,
                    client=self.client,
                    cfg=self.cfg,
                    prompt_builder=self.prompt_builder,
                    budget=ep_budget,
                )
                _check_cancelled()
                self.cache.put(
                    s2_key,
                    {
                        "macro_stages": [s.model_dump(mode="json") for s in stages],
                        "arm_timelines": arms.model_dump(mode="json"),
                    },
                )
                return stages, arms

            def _exec_s5() -> SuccessVerdict:
                _check_cancelled()
                s5_cached = None if (fresh or "s5_success" in fs_stages) else self.cache.get(s5_key)
                if s5_cached is not None:
                    verdict = SuccessVerdict.model_validate(s5_cached)
                    ep_budget.record(
                        stage="s5_success",
                        model=self.client.model_for_stage("s5_success"),
                        cache_hit=True,
                    )
                    return verdict
                verdict = run_stage5_success(
                    view=ev,
                    task=t_spec,
                    client=self.client,
                    cfg=self.cfg,
                    prompt_builder=self.prompt_builder,
                    budget=ep_budget,
                )
                _check_cancelled()
                self.cache.put(s5_key, verdict.model_dump(mode="json"))
                return verdict

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_s2 = pool.submit(_exec_s2)
                fut_s5 = pool.submit(_exec_s5)
                macro_stages, arm_timelines = fut_s2.result()
                success_verdict = fut_s5.result()

            _check_cancelled()
            s2_out_hash = canonical_sha256(
                {
                    "macro_stages": [s.model_dump(mode="json") for s in macro_stages],
                    "arm_timelines": arm_timelines.model_dump(mode="json"),
                }
            )
            s3_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                upstream_hash=f"{s2_key}:{s2_out_hash}",
                stage_basis=self.cfg.s3_cache_basis(t_spec.task_id, few_shot_hash),
            )

            # Level 3A: Stage 3 (Sliding QC Discovery w/ S2 Protection & Few-Shot)
            _check_cancelled()
            s3_cached = None if (fresh or "s3_discover" in fs_stages) else self.cache.get(s3_key)
            if s3_cached is not None:
                s3_cands = [
                    QcSegment.model_validate(x) for x in s3_cached.get("segments", [])
                ]
                ep_budget.record(
                    stage="s3_discover",
                    model=self.client.model_for_stage("s3_discover"),
                    cache_hit=True,
                )
            else:
                s3_cands = run_stage3_discover(
                    view=ev,
                    task=t_spec,
                    macro_stages=macro_stages,
                    few_shots=fs_list,
                    client=self.client,
                    cfg=self.cfg,
                    prompt_builder=self.prompt_builder,
                    labels=self.cfg.labels,
                    budget=ep_budget,
                )
                _check_cancelled()
                self.cache.put(
                    s3_key,
                    {"segments": [s.model_dump(mode="json") for s in s3_cands]},
                )

            _check_cancelled()
            s3_out_hash = canonical_sha256(
                [s.model_dump(mode="json") for s in s3_cands]
            )
            s4_key = StageCache.compute_key(
                episode_fingerprint=ep_fp,
                upstream_hash=f"{s2_key}:{s2_out_hash}:{s3_key}:{s3_out_hash}",
                stage_basis=self.cfg.s4_cache_basis(t_spec.task_id, few_shot_hash),
            )

            # Level 3B: Stage 4 (4A Global Merge + 4B 0.5s CoV Refinement)
            _check_cancelled()
            s4_cached = None if (fresh or "s4_verify" in fs_stages) else self.cache.get(s4_key)
            if s4_cached is not None:
                s4_segs = [
                    QcSegment.model_validate(x) for x in s4_cached.get("segments", [])
                ]
                ep_budget.record(
                    stage="s4_verify",
                    model=self.client.model_for_stage("s4_verify"),
                    cache_hit=True,
                )
            else:
                s4_segs = run_stage4_verify(
                    view=ev,
                    candidates=s3_cands,
                    task=t_spec,
                    macro_stages=macro_stages,
                    few_shots=fs_list,
                    client=self.client,
                    cfg=self.cfg,
                    prompt_builder=self.prompt_builder,
                    labels=self.cfg.labels,
                    budget=ep_budget,
                )
                _check_cancelled()
                self.cache.put(
                    s4_key,
                    {"segments": [s.model_dump(mode="json") for s in s4_segs]},
                )

            # Level 4: Detector-Free Light Guardrails
            _check_cancelled()
            if ev.kinematics is not None:
                qc_segments = apply_hallucination_check(
                    segments=s4_segs,
                    kinematics=ev.kinematics,
                    cfg=self.cfg,
                    duration_s=ev.duration_s,
                )
                qc_segments = apply_splice_gate(
                    segments=qc_segments,
                    kinematics=ev.kinematics,
                    cfg=self.cfg,
                    duration_s=ev.duration_s,
                )
            else:
                qc_segments = apply_visual_gate(
                    segments=s4_segs,
                    media=ev.media,
                    cfg=self.cfg,
                    duration_s=ev.duration_s,
                )

            _check_cancelled()
            run_id = f"run_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            fpshot = ConfigFingerprintSnapshot(
                model_routing={
                    k: str(v)
                    for k, v in self.cfg.settings.get("vlm", {})
                    .get("stage_models", {})
                    .items()
                },
                prompt_bundle_version=self.cfg.fingerprints.prompt_bundle_sha256[:12],
                label_taxonomy_version=f"v{self.cfg.labels.version}",
                task_spec_version=t_spec.version,
                few_shot_pack_version=few_shot_hash[:12] if fs_list else "none",
                config_sha256=self.cfg.fingerprints.config_sha256,
            )
            artifact = VlmRunArtifact(
                schema_version="1.0",
                run_id=run_id,
                episode_id=ev.meta.episode_id,
                created_at=time.time(),
                is_eval_sandbox=is_eval_sandbox,
                config_fingerprint=fpshot,
                stage_refs=StageRefs(
                    s1_grounding=s1_key,
                    s2_macro=s2_key,
                    s3_discover=s3_key,
                    s4_verify=s4_key,
                    s5_success=s5_key,
                ),
                grounding=grounding,
                macro_stages=macro_stages,
                arm_timelines=arm_timelines,
                qc_segments=qc_segments,
                success=success_verdict,
                usage=ep_budget.to_usage_summary(),
            )
            _check_cancelled()
            if persist:
                self.repo.save_vlm_run_artifact(
                    artifact, before_commit_check=_check_cancelled
                )
                if not is_eval_sandbox:
                    try:
                        cur_after = EpisodeStatus(
                            self.repo.get_episode_row(ev.meta.episode_id)["status"]
                        )
                        _emit_status(cur_after)
                    except NotFoundError:
                        pass
            return artifact

        except BaseException as exc:
            if persist and not is_eval_sandbox and (
                transitioned_to_running or isinstance(exc, VlmCancelledError)
            ):
                if can_rollback is None or can_rollback(ev.meta.episode_id):
                    with contextlib.suppress(Exception):
                        prev_map = (
                            {ev.meta.episode_id: prev_status}
                            if prev_status in {EpisodeStatus.INGESTED, EpisodeStatus.VLM_READY}
                            else None
                        )
                        restored = self.repo.restore_vlm_episodes_on_cancel(
                            [ev.meta.episode_id], prev_statuses=prev_map
                        )
                        for _, restored_st in restored:
                            _emit_status(restored_st)
            raise

    run = run_episode
    execute = run_episode
