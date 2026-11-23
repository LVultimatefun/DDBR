import type {
  EpisodeListItem,
  HumanReviewArtifact,
  JobStatus,
  QcSegment,
  ResolvedEpisode,
} from "../api/types";
import { defaultGuardrailReport } from "../constants/labels";

export function makeQcSegment(overrides: Partial<QcSegment> = {}): QcSegment {
  return {
    seg_id: "seg_1",
    t_start: 1.0,
    t_end: 2.5,
    label_id: "missed_grasp_retry",
    name: "Missed Grasp Retry",
    training_role: "recovery",
    splice_policy: "mask_loss",
    mask_side: "left",
    affected_arm: "left",
    visual_evidence: {
      initial_state: "Hand approaches target block",
      process_change: "First grasp misses and retracts",
      end_state: "Re-grasped securely",
      confidence_note: "",
    },
    vlm_raw_policy: "mask_loss",
    guardrail: defaultGuardrailReport(),
    is_unknown: false,
    unknown_note: "",
    suggested_label: null,
    confidence: 0.92,
    ...overrides,
  };
}

export function makeResolvedEpisode(overrides: Partial<ResolvedEpisode> = {}): ResolvedEpisode {
  const baseSeg = makeQcSegment();
  return {
    episode_id: "demo_ds/ep_001",
    meta: {
      episode_id: "demo_ds/ep_001",
      dataset: "demo_ds",
      stem: "ep_001",
      task_id: "pick_and_place",
      adapter_type: "h5_raw_v3",
      source_uri: "/ro/demo_ds/ep_001.h5",
      source_mtime: 1700000000,
      episode_fingerprint: "fp_123",
      duration_s: 10.0,
      fps: 30,
      frame_count: 300,
      cameras: [
        {
          name: "head_rgbd",
          role: "main",
          width: 1280,
          height: 720,
          fps: 30,
          source_type: "h5_jpeg",
          is_virtual_crop: false,
          source_stream: "head_rgbd",
        },
        {
          name: "torso_rgbd",
          role: "front",
          width: 1280,
          height: 720,
          fps: 30,
          source_type: "h5_jpeg",
          is_virtual_crop: false,
          source_stream: "torso_rgbd",
        },
      ],
      camera_roles: { main: "head_rgbd", front: "torso_rgbd" },
      kinematics: {
        has_kinematics: true,
        n_arms: 2,
        arm_dof: 7,
        hand_dof: 22,
        torso_dof: 3,
        has_chassis: false,
        has_tactile: true,
        tactile_dim: 30,
        state_dim: 81,
        has_action_cmd: true,
        source_hz: { robot: 30 },
      },
      recorded_success: true,
      extra_meta: {},
    },
    status: "vlm_ready",
    active_run_id: "run_001",
    active_rev_id: null,
    is_human_reviewed: false,
    qc_segments: [baseSeg],
    success: {
      passed: true,
      verdict: "success",
      confidence: 0.95,
      criteria: [
        {
          criterion: "Object placed stably inside target bin",
          met: true,
          satisfied_at_s: 8.5,
          evidence: "Object base rests flat inside the bin at 8.5s",
        },
      ],
      control_check: {
        cut_s: 3.0,
        cut_ratio: 0.3,
        control_verdict: "failure",
        passed: true,
        met: [false],
      },
      summary: "Task completed successfully",
    },
    macro_stages: [
      {
        stage_id: "M1",
        t_start: 0.0,
        t_end: 5.0,
        label: "Pick Object",
        label_en: "pick_object",
        summary: "Left hand approaches and grasps the target block",
        color_rgb: [59, 130, 246],
      },
      {
        stage_id: "M2",
        t_start: 5.0,
        t_end: 10.0,
        label: "Place in Bin",
        label_en: "place_in_bin",
        summary: "Places the block steadily into the bin",
        color_rgb: [16, 185, 129],
      },
    ],
    grounding: {
      schema_version: 1,
      ref_size: [1280, 720],
      keyframes_s: [0, 2.5, 5.0],
      object_tracks_2d: {
        block: {
          object_id: "block",
          name: "Target Block",
          tag_pos: "top",
          is_static_container: false,
          keyframes: [
            {
              row: 0,
              time_s: 0.0,
              visible: true,
              bbox_norm_1000: [200, 300, 350, 500],
              bbox_px: [256, 216, 448, 360],
              state: "stationary",
              status_kind: "neutral",
            },
          ],
        },
      },
    },
    arm_timelines: {
      left: [
        { t_start: 0, t_end: 5.0, state: "work", desc: "Grasp target block" },
        { t_start: 5.0, t_end: 10.0, state: "idle", desc: "Retract" },
      ],
      right: [{ t_start: 0, t_end: 10.0, state: "hold", desc: "Stabilize tray" }],
    },
    raw_vlm_segments: [baseSeg],
    raw_vlm_success: {
      passed: true,
      verdict: "success",
      confidence: 0.95,
      criteria: [],
      control_check: null,
      summary: "VLM raw verdict: success",
    },
    raw_vlm_macro_stages: [],
    review_diff: null,
    notes: "",
    flagged_reason: null,
    claimed_by: null,
    claim_expires_at: null,
    ...overrides,
  };
}

export function makeEpisodeListItem(overrides: Partial<EpisodeListItem> = {}): EpisodeListItem {
  return {
    episode_id: "demo_ds/ep_001",
    dataset: "demo_ds",
    task_id: "pick_and_place",
    adapter_type: "h5_raw_v3",
    duration_s: 10.0,
    fps: 30.0,
    has_kinematics: 1,
    status: "vlm_ready",
    active_run_id: "run_001",
    active_rev_id: null,
    claimed_by: null,
    claim_expires_at: null,
    resolved_passed: true,
    cut_ratio: 0.0,
    mask_ratio: 0.15,
    badcase_count: 1,
    unknown_count: 0,
    guardrail_downgraded: false,
    updated_at: 1700000100,
    ...overrides,
  };
}

export function makeReviewArtifact(
  overrides: Partial<HumanReviewArtifact> = {},
): HumanReviewArtifact {
  return {
    rev_id: "rev_001",
    episode_id: "demo_ds/ep_001",
    base_run_id: "run_001",
    author: "lead_1",
    role: "lead",
    action: "approve",
    created_at: 1700000200,
    watch_coverage: {
      watched_intervals: [[0, 10]],
      watched_s: 10,
      duration_s: 10,
      coverage_ratio: 1,
      confirmed: true,
    },
    qc_segments: [makeQcSegment()],
    success: {
      passed: false,
      verdict: "failure",
      confidence: 1.0,
      criteria: [],
      control_check: null,
      summary: "Snapshot revision",
    },
    macro_stages: [],
    notes: "Historical review snapshot",
    flagged_reason: null,
    diff_summary: {
      fp_deleted: 1,
      fn_added: 0,
      policy_flips: 0,
      label_changes: 0,
      boundary_nudges: 0,
      macro_stage_edits: 0,
      success_flipped: true,
      total_edits: 2,
    },
    ...overrides,
  };
}

export function makeJobStatus(overrides: Partial<JobStatus> = {}): JobStatus {
  return {
    job_id: "job_001",
    job_type: "batch_vlm",
    status: "running",
    progress_done: 1,
    progress_total: 4,
    summary: null,
    created_at: 1700000300,
    updated_at: 1700000305,
    ...overrides,
  };
}
