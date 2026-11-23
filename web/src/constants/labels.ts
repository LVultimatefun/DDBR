import type {
  EpisodeStatus,
  GuardrailMetrics,
  GuardrailReport,
  MaskSide,
  QcSegment,
  SplicePolicy,
  TrainingRole,
  VisualLabelDef,
} from "../api/types";

export const FORBIDDEN_LEGACY_LABELS = new Set<string>(["long_pause", "back_and_forth"]);

export const ALL_EPISODE_STATUSES: EpisodeStatus[] = [
  "ingested",
  "vlm_queued",
  "vlm_running",
  "vlm_ready",
  "claimed",
  "reviewed",
  "flagged",
  "approved",
  "gt_frozen",
];

export const EPISODE_STATUS_META: Record<EpisodeStatus, { label: string; color: string }> = {
  ingested: { label: "Ingested", color: "#64748b" },
  vlm_queued: { label: "VLM Queued", color: "#0284c7" },
  vlm_running: { label: "VLM Running", color: "#2563eb" },
  vlm_ready: { label: "Ready for Review", color: "#0891b2" },
  claimed: { label: "Claimed", color: "#7c3aed" },
  reviewed: { label: "Reviewed", color: "#d97706" },
  flagged: { label: "Flagged", color: "#dc2626" },
  approved: { label: "Approved", color: "#16a34a" },
  gt_frozen: { label: "GT Frozen", color: "#0f766e" },
};

export const DEFAULT_VISUAL_LABELS: VisualLabelDef[] = [
  {
    id: "boundary_idle",
    name: "Boundary Idle",
    category: "idle",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "cut_clean",
    collector_ui: {
      title: "Leading or Trailing Idle",
      hint: "Full-scene motionless waiting before manipulation begins or after the task has completely finished.",
    },
    vlm_semantics: {
      definition:
        "Preparation stillness at the start of a trajectory before motion begins, or trailing global stillness after the task completes and both arms retract.",
      counter_examples: [
        "Brief mid-task pause while aligning hands with a target opening",
        "One hand steadily holding an object while waiting for the other hand to act",
      ],
      visual_cues: [
        "Both arms and end-effectors have virtually no displacement; manipulated objects and containers are completely still",
        "Located at the very start or end of the trajectory with no active task progress",
      ],
    },
  },
  {
    id: "mid_idle_freeze",
    name: "Mid-Task Freeze",
    category: "idle",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "cut_clean",
    collector_ui: {
      title: "Mid-Task Stall / Freeze",
      hint: "Both arms freeze mid-task with no effective fine adjustment or forward progress.",
    },
    vlm_semantics: {
      definition:
        "Obvious abnormal stall partway through execution where both arms hover or lock still, neither advancing the subtask nor performing fine alignment.",
      counter_examples: [
        "Slow fine adjustment during tight insertion or alignment (precision_fine_adjust)",
        "One hand steadily bracing an object while the other hand operates (bimanual_assist_hold)",
      ],
      visual_cues: [
        "Consecutive frames show no visible change in wrists, fingers, or target objects",
        "Motion resumes only after the stall ends (cut eligibility is verified by the splice guardrail)",
      ],
    },
  },
  {
    id: "missed_grasp_retry",
    name: "Missed Grasp Retry",
    category: "failure_recovery",
    status: "active",
    version: 1,
    default_training_role: "recovery",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Missed Grasp or Bump Retry",
      hint: "Gripper closes on empty space or bumps the object aside, then opens and retreats to retry grasping.",
    },
    vlm_semantics: {
      definition:
        "The end-effector approaches and closes on the target object without securing it (whiffing or knocking it askew), then opens, retracts, and re-aligns to grasp again.",
      counter_examples: [
        "Direct stable approach and clean first-try grasp",
        "Minor in-hand grip adjustment without releasing and re-grasping",
      ],
      visual_cues: [
        "Target object does not move with the gripper or is nudged off-center when fingers close",
        "Gripper opens, withdraws, and reaches back toward the same object",
      ],
    },
  },
  {
    id: "object_slip_recovery",
    name: "Object Slip Recovery",
    category: "failure_recovery",
    status: "active",
    version: 1,
    default_training_role: "recovery",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Object Slip / Drop Recovery",
      hint: "A grasped object slips, tilts, or drops, and the robot quickly stabilizes or picks it back up.",
    },
    vlm_semantics: {
      definition:
        "A held or transported object visibly slips, drops, or tips over, and the robot performs a corrective recovery to re-grasp or upright it and continue the task.",
      counter_examples: [
        "Object drops causing unrecoverable task failure (label as terminal_failure)",
        "Object remains stable in the hand with no slip",
      ],
      visual_cues: [
        "Object suddenly drops, slides out of the grip, or tips over",
        "End-effector chases, uprights, or re-grasps the object and resumes task progress",
      ],
    },
  },
  {
    id: "placement_redo",
    name: "Placement / Insertion Redo",
    category: "failure_recovery",
    status: "active",
    version: 1,
    default_training_role: "recovery",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Misaligned Placement Redo",
      hint: "Placement, capping, or insertion misses or seats crookedly, requiring withdrawal to re-align.",
    },
    vlm_semantics: {
      definition:
        "An attempt to insert, cap, or place an object fails to seat properly or rests tilted, forcing a clear upward/backward withdrawal before re-aligning and trying again.",
      counter_examples: [
        "Continuous monotonic downward insertion or slight twist seating cleanly on the first try (precision_fine_adjust)",
      ],
      visual_cues: [
        "Object catches on the container rim/slot edge or rests tilted",
        "Arm clearly withdraws upward or backward before attempting placement again",
      ],
    },
  },
  {
    id: "hesitation_oscillation",
    name: "Hesitation / Oscillation",
    category: "inefficiency",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Hesitation or Back-and-Forth Jitter",
      hint: "Repeated probing, oscillating back and forth, circling in place, or twitchy jitter with no clear progress.",
    },
    vlm_semantics: {
      definition:
        "Arm or hand moves back and forth repeatedly, circles hesitantly in free space, or probes without monotonic progress toward the current subtask.",
      counter_examples: [
        "Periodic back-and-forth motion inherently required by the task (e.g., wiping or stirring)",
        "Small monotonic fine alignment right before insertion",
      ],
      visual_cues: [
        "End-effector repeatedly approaches and retreats or sways side-to-side while object state makes no progress",
        "Clearly distinct from intentional periodic task motions",
      ],
    },
  },
  {
    id: "collision_or_knock",
    name: "Collision / Knock",
    category: "failure",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Collision or Accidental Knock",
      hint: "Arm or held object hits an unrelated container, table edge, or knocks over a nearby item.",
    },
    vlm_semantics: {
      definition:
        "Robot arm, hand, or held object accidentally collides with the table, unrelated container, or fixture, visibly displacing, shaking, or toppling the struck item.",
      counter_examples: ["Intentional contact with the target object or pressing an active fixture"],
      visual_cues: [
        "Non-target container or tabletop object visibly shifts, bounces, or tips after contact",
        "Arm experiences visible obstruction or rebound at the moment of impact",
      ],
    },
  },
  {
    id: "precision_fine_adjust",
    name: "Precision Fine Adjust (Positive)",
    category: "necessary_skill",
    status: "active",
    version: 1,
    default_training_role: "necessary",
    default_splice_policy: "keep_positive",
    collector_ui: {
      title: "Precision Fine Alignment (Good Skill)",
      hint: "Slow, steady, monotonic fine adjustment for tight insertion, precision placement, or threading.",
    },
    vlm_semantics: {
      definition:
        "Slow, deliberate, monotonically converging fine adjustment for narrow-tolerance insertion, small-part alignment, or delicate manipulation; high-value positive skill.",
      counter_examples: [
        "Explicit withdrawal and retry after a failed attempt (placement_redo)",
        "Aimless back-and-forth swaying without progress (hesitation_oscillation)",
      ],
      visual_cues: [
        "Motion is slow but directionally consistent and converging as the object enters the target pose",
      ],
    },
  },
  {
    id: "bimanual_assist_hold",
    name: "Bimanual Assist Hold (Positive)",
    category: "necessary_skill",
    status: "active",
    version: 1,
    default_training_role: "necessary",
    default_splice_policy: "keep_positive",
    collector_ui: {
      title: "Stabilizing Assist Hold (Good Skill)",
      hint: "One hand steadily holds or braces a container/object while the other hand performs the operation.",
    },
    vlm_semantics: {
      definition:
        "In bimanual coordination, one hand steadily presses, braces, or supports a container or workpiece—appearing stationary while providing essential mechanical stabilization for the active hand.",
      counter_examples: [
        "Both hands completely idle and disengaged from task objects (mid_idle_freeze)",
      ],
      visual_cues: [
        "One hand firmly grips or braces the container/object while the other hand extracts, inserts, or manipulates",
      ],
    },
  },
  {
    id: "terminal_failure",
    name: "Terminal Failure",
    category: "terminal",
    status: "active",
    version: 1,
    default_training_role: "terminal_fail",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Unrecoverable Terminal Error",
      hint: "Tipping, dropping an object out of the workspace, or damage that makes task completion impossible.",
    },
    vlm_semantics: {
      definition:
        "Severe unrecovered error causing the entire episode to fail, such as dropping a key object off the table, spilling a container, jamming a tool, or ending incomplete.",
      counter_examples: [
        "Object slips and is immediately uprighted to finish the task (object_slip_recovery)",
      ],
      visual_cues: [
        "Target container overturned, key object lost out of reach, or irreversible failure end state",
      ],
    },
  },
  {
    id: "unknown_anomaly",
    name: "Unknown Anomaly (Triage)",
    category: "unknown",
    status: "active",
    version: 1,
    default_training_role: "badcase",
    default_splice_policy: "mask_loss",
    collector_ui: {
      title: "Other Unknown Anomaly",
      hint: "Clear visual stall, occlusion, or unusual behavior that does not fit any existing label.",
    },
    vlm_semantics: {
      definition:
        "Visually clear manipulation anomaly, environmental disturbance, or hardware fault that does not map to any active core label and requires triage in VLM Lab.",
      counter_examples: ["Normal slow forward task progression"],
      visual_cues: [
        "Undefined anomaly, abnormal camera occlusion, or atypical fault; describe the observation in unknown_note",
      ],
    },
  },
];

export function normalizeSegmentTriplet(
  seg: QcSegment,
  patch: Partial<QcSegment> = {},
  catalog: VisualLabelDef[] = DEFAULT_VISUAL_LABELS,
): QcSegment {
  const merged: QcSegment = { ...seg, ...patch };

  if (FORBIDDEN_LEGACY_LABELS.has(merged.label_id)) {
    throw new Error(`Legacy label_id "${merged.label_id}" is abolished in robo_vlm_qc.`);
  }

  // If label changed without explicit policy/role override, inherit label defaults
  if (patch.label_id && patch.label_id !== seg.label_id) {
    const def = catalog.find((l) => l.id === patch.label_id);
    if (def) {
      merged.name = patch.name ?? def.name;
      if (!patch.training_role) {
        merged.training_role = def.default_training_role;
      }
      if (!patch.splice_policy) {
        merged.splice_policy = def.default_splice_policy;
      }
    }
    if (patch.label_id === "unknown_anomaly") {
      merged.is_unknown = true;
    } else if (seg.label_id === "unknown_anomaly" && patch.is_unknown === undefined) {
      merged.is_unknown = false;
    }
  }

  const policyChanged = patch.splice_policy !== undefined && patch.splice_policy !== seg.splice_policy;
  const roleChanged = patch.training_role !== undefined && patch.training_role !== seg.training_role;

  // Enforce QcSegment._validate_triplet_and_span invariants
  let policy: SplicePolicy = merged.splice_policy;
  let role: TrainingRole = merged.training_role;
  let maskSide: MaskSide | null = merged.mask_side;

  if (roleChanged && !policyChanged) {
    if (role === "necessary" && policy !== "keep_positive") {
      policy = "keep_positive";
    } else if (role === "terminal_fail" && policy === "keep_positive") {
      policy = "mask_loss";
    } else if (role === "badcase" && policy === "keep_positive") {
      policy = "mask_loss";
    }
  }

  if (policy === "keep_positive") {
    maskSide = null;
    if (role !== "necessary" && role !== "recovery") {
      role = "necessary";
    }
  } else if (policy === "cut_clean") {
    maskSide = null;
    if (role !== "badcase" && role !== "terminal_fail") {
      role = "badcase";
    }
  } else if (policy === "mask_loss") {
    if (maskSide !== "left" && maskSide !== "right" && maskSide !== "both") {
      maskSide = merged.affected_arm || "both";
    }
    if (role !== "badcase" && role !== "terminal_fail" && role !== "recovery") {
      role = "badcase";
    }
  }

  merged.t_start = Math.max(0, Number.isFinite(merged.t_start) ? merged.t_start : 0);
  if (!Number.isFinite(merged.t_end) || merged.t_end <= merged.t_start) {
    merged.t_end = Number((merged.t_start + 0.1).toFixed(3));
  }

  return {
    ...merged,
    splice_policy: policy,
    training_role: role,
    mask_side: maskSide,
  };
}

export function validateSegmentTriplet(seg: QcSegment): string | null {
  if (FORBIDDEN_LEGACY_LABELS.has(seg.label_id)) {
    return `Legacy label "${seg.label_id}" has been retired`;
  }
  if (seg.t_start < 0 || seg.t_end <= seg.t_start) {
    return "Segment end time must be greater than start time";
  }
  if (seg.splice_policy === "keep_positive") {
    if (seg.mask_side !== null) return "mask_side must be null for keep_positive policy";
    if (seg.training_role !== "necessary" && seg.training_role !== "recovery") {
      return `keep_positive is incompatible with training_role "${seg.training_role}"`;
    }
  } else if (seg.splice_policy === "cut_clean") {
    if (seg.mask_side !== null) return "mask_side must be null for cut_clean policy";
    if (seg.training_role !== "badcase" && seg.training_role !== "terminal_fail") {
      return `cut_clean is incompatible with training_role "${seg.training_role}"`;
    }
  } else if (seg.splice_policy === "mask_loss") {
    if (!seg.mask_side || !["left", "right", "both"].includes(seg.mask_side)) {
      return "mask_loss requires mask_side (left/right/both)";
    }
    if (!["badcase", "terminal_fail", "recovery"].includes(seg.training_role)) {
      return `mask_loss is incompatible with training_role "${seg.training_role}"`;
    }
  }
  return null;
}

export function defaultGuardrailMetrics(): GuardrailMetrics {
  return {
    eef_pos_m: 0,
    eef_rot_rad: 0,
    hand_rad: 0,
    torso_rad: 0,
    chassis_pos_m: 0,
    chassis_yaw_rad: 0,
    other_arm_disp_m: 0,
    bimanual_coupled: false,
    visual_ssim: null,
  };
}

export function defaultGuardrailReport(): GuardrailReport {
  return {
    gate_type: "none",
    passed: true,
    downgraded: false,
    downgrade_reason: null,
    guardrail_override: false,
    override_reason: null,
    metrics: defaultGuardrailMetrics(),
    opt_t_start: null,
    opt_t_end: null,
  };
}

export function createDefaultSegment(
  tStart: number,
  durationS: number,
  labelDef: VisualLabelDef = DEFAULT_VISUAL_LABELS[2],
): QcSegment {
  const safeStart = Math.max(0, Number(tStart.toFixed(2)));
  const safeEnd = Math.min(durationS, Math.max(safeStart + 0.5, Number((safeStart + 1.0).toFixed(2))));
  const base: QcSegment = {
    seg_id: `human_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    t_start: safeStart,
    t_end: safeEnd > safeStart ? safeEnd : Number((safeStart + 0.2).toFixed(2)),
    label_id: labelDef.id,
    name: labelDef.name,
    training_role: labelDef.default_training_role,
    splice_policy: labelDef.default_splice_policy,
    mask_side: labelDef.default_splice_policy === "mask_loss" ? "both" : null,
    affected_arm: "both",
    visual_evidence: {
      initial_state: "",
      process_change: "",
      end_state: "",
      confidence_note: "Manual annotation",
    },
    vlm_raw_policy: labelDef.default_splice_policy,
    guardrail: defaultGuardrailReport(),
    is_unknown: labelDef.id === "unknown_anomaly",
    unknown_note: "",
    suggested_label: null,
    confidence: 1.0,
  };
  return normalizeSegmentTriplet(base);
}
