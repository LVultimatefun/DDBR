export type AdapterType = "h5_raw_v3" | "lerobot" | "video_dir";

export type CameraRole =
  | "main"
  | "front"
  | "left_wrist"
  | "right_wrist"
  | "wide"
  | "wide_right"
  | "secondary"
  | "aux";

export interface CameraSpec {
  name: string;
  role: CameraRole;
  width: number;
  height: number;
  fps: number;
  source_type: string;
  is_virtual_crop: boolean;
  source_stream: string | null;
}

export interface KinematicsSpec {
  has_kinematics: boolean;
  n_arms: number;
  arm_dof: number;
  hand_dof: number;
  torso_dof: number;
  has_chassis: boolean;
  has_tactile: boolean;
  tactile_dim: number;
  state_dim: number;
  has_action_cmd: boolean;
  source_hz: Record<string, number>;
}

export interface EpisodeMeta {
  episode_id: string;
  dataset: string;
  stem: string;
  task_id: string;
  adapter_type: AdapterType;
  source_uri: string;
  source_mtime: number;
  episode_fingerprint: string;
  duration_s: number;
  fps: number;
  frame_count: number;
  cameras: CameraSpec[];
  camera_roles: Record<string, string>;
  kinematics: KinematicsSpec;
  recorded_success: boolean | null;
  extra_meta: Record<string, unknown>;
}

export type GroundingStatusKind = "neutral" | "warn" | "error" | "good";

export interface GroundingKeyframe {
  row: number;
  time_s: number;
  visible: boolean;
  bbox_norm_1000: [number, number, number, number];
  bbox_px: [number, number, number, number];
  state: string;
  status_kind: GroundingStatusKind;
}

export interface GroundingTrack {
  object_id: string;
  name: string;
  tag_pos: "top" | "bottom";
  is_static_container: boolean;
  keyframes: GroundingKeyframe[];
}

export interface GroundingResult {
  schema_version: number;
  ref_size: [number, number];
  keyframes_s: number[];
  object_tracks_2d: Record<string, GroundingTrack>;
}

export interface MacroStage {
  stage_id: string;
  t_start: number;
  t_end: number;
  label: string;
  label_en: string;
  summary: string;
  color_rgb: [number, number, number];
}

export type ArmState = "idle" | "work" | "hold" | "anom" | "mask" | "good";

export interface ArmSpan {
  t_start: number;
  t_end: number;
  state: ArmState;
  desc: string;
}

export interface ArmTimelines {
  left: ArmSpan[];
  right: ArmSpan[];
}

export type TrainingRole = "necessary" | "recovery" | "badcase" | "terminal_fail";
export type SplicePolicy = "keep_positive" | "mask_loss" | "cut_clean";
export type MaskSide = "left" | "right" | "both";

export interface VisualEvidence {
  initial_state: string;
  process_change: string;
  end_state: string;
  confidence_note: string;
}

export interface GuardrailMetrics {
  eef_pos_m: number;
  eef_rot_rad: number;
  hand_rad: number;
  torso_rad: number;
  chassis_pos_m: number;
  chassis_yaw_rad: number;
  other_arm_disp_m: number;
  bimanual_coupled: boolean;
  visual_ssim: number | null;
}

export interface GuardrailReport {
  gate_type: "splice_gate" | "hallucination" | "visual_gate" | "none";
  passed: boolean;
  downgraded: boolean;
  downgrade_reason: string | null;
  guardrail_override: boolean;
  override_reason: string | null;
  metrics: GuardrailMetrics;
  opt_t_start: number | null;
  opt_t_end: number | null;
}

export interface QcSegment {
  seg_id: string;
  t_start: number;
  t_end: number;
  label_id: string;
  name: string;
  training_role: TrainingRole;
  splice_policy: SplicePolicy;
  mask_side: MaskSide | null;
  affected_arm: MaskSide;
  visual_evidence: VisualEvidence;
  vlm_raw_policy: SplicePolicy;
  guardrail: GuardrailReport;
  is_unknown: boolean;
  unknown_note: string;
  suggested_label: string | null;
  confidence: number;
}

export interface CriterionResult {
  criterion: string;
  met: boolean | null;
  satisfied_at_s: number | null;
  evidence: string;
}

export interface ControlCheckResult {
  cut_s: number;
  cut_ratio: number;
  control_verdict: "success" | "failure" | "unsure";
  passed: boolean;
  met: Array<boolean | null>;
}

export interface SuccessVerdict {
  passed: boolean | null;
  verdict: "success" | "failure" | "unsure";
  confidence: number;
  criteria: CriterionResult[];
  control_check: ControlCheckResult | null;
  summary: string;
}

export type EpisodeStatus =
  | "ingested"
  | "vlm_queued"
  | "vlm_running"
  | "vlm_ready"
  | "claimed"
  | "reviewed"
  | "flagged"
  | "approved"
  | "gt_frozen";

export type ReviewRole = "reviewer" | "lead";
export type ReviewAction = "save_draft" | "submit_review" | "flag" | "approve" | "freeze_gt";

export interface WatchCoverage {
  watched_intervals: Array<[number, number]>;
  watched_s: number;
  duration_s: number;
  coverage_ratio: number;
  confirmed: boolean;
}

export interface ReviewDiff {
  fp_deleted: number;
  fn_added: number;
  policy_flips: number;
  label_changes: number;
  boundary_nudges: number;
  macro_stage_edits: number;
  success_flipped: boolean;
  total_edits: number;
}

export interface HumanReviewArtifact {
  schema_version?: string;
  rev_id: string;
  episode_id: string;
  base_run_id: string;
  author: string;
  role: ReviewRole;
  action: ReviewAction;
  created_at: number;
  watch_coverage: WatchCoverage;
  qc_segments: QcSegment[];
  success: SuccessVerdict;
  macro_stages: MacroStage[];
  notes: string;
  flagged_reason: string | null;
  diff_summary: ReviewDiff;
}

/** Strict DTO matching ReviewWritePayload(extra="forbid"). Never include author, grounding, arm_timelines, or raw_vlm_segments. */
export interface ReviewWritePayload {
  base_run_id: string;
  action: ReviewAction;
  qc_segments: QcSegment[];
  success: SuccessVerdict;
  macro_stages: MacroStage[];
  watch_coverage: WatchCoverage;
  notes?: string;
  flagged_reason?: string | null;
}

export interface ResolvedEpisode {
  episode_id: string;
  meta: EpisodeMeta;
  status: EpisodeStatus;
  active_run_id: string | null;
  active_rev_id: string | null;
  is_human_reviewed: boolean;
  qc_segments: QcSegment[];
  success: SuccessVerdict;
  macro_stages: MacroStage[];
  grounding: GroundingResult;
  arm_timelines: ArmTimelines;
  raw_vlm_segments: QcSegment[];
  raw_vlm_success: SuccessVerdict | null;
  raw_vlm_macro_stages: MacroStage[];
  review_diff: ReviewDiff | null;
  notes: string;
  flagged_reason: string | null;
  claimed_by?: string | null;
  claim_expires_at?: number | null;
  selected_run_id?: string | null;
  is_historical_run_view?: boolean;
  guardrail_downgraded?: boolean;
}

export interface EpisodeListItem {
  episode_id: string;
  dataset: string;
  task_id: string;
  adapter_type: AdapterType;
  source_uri?: string;
  source_mtime?: number;
  camera_roles?: Record<string, string>;
  duration_s: number;
  fps: number;
  has_kinematics: boolean | number;
  status: EpisodeStatus;
  active_run_id: string | null;
  active_rev_id: string | null;
  claimed_by: string | null;
  claim_expires_at: number | null;
  resolved_passed: boolean | number | null;
  cut_ratio: number | null;
  mask_ratio: number | null;
  badcase_count: number | null;
  unknown_count: number | null;
  guardrail_downgraded?: boolean;
  updated_at: number;
}

export type LabelCategory =
  | "idle"
  | "failure_recovery"
  | "inefficiency"
  | "failure"
  | "necessary_skill"
  | "terminal"
  | "unknown";

export type LabelStatus = "active" | "proposed" | "retired";
export type DiagnosisErrorType =
  | "fp_overflag"
  | "fn_missed"
  | "policy_flip"
  | "boundary"
  | "success_flip"
  | "stage";

export interface CollectorUiHint {
  title: string;
  hint: string;
}

export interface VlmSemantics {
  definition: string;
  counter_examples: string[];
  visual_cues: string[];
}

export interface VisualLabelDef {
  id: string;
  name: string;
  category: LabelCategory;
  status: LabelStatus;
  version: number;
  default_training_role: TrainingRole;
  default_splice_policy: SplicePolicy;
  collector_ui: CollectorUiHint;
  vlm_semantics: VlmSemantics;
  merged_from?: string[];
  merged_into?: string | null;
}

export interface LabelRegistry {
  schema_version: string;
  version: number;
  categories: LabelCategory[];
  labels: VisualLabelDef[];
}

export interface FewShotCreateRequest {
  scope: string;
  task_id?: string | null;
  label_id?: string | null;
  error_type: DiagnosisErrorType;
  episode_id: string;
  t_start: number;
  t_end: number;
  vlm_wrong_json: Record<string, unknown>;
  human_right_json: Record<string, unknown>;
  rationale: string;
  capture_tile?: boolean;
}

export interface FewShotItem extends FewShotCreateRequest {
  example_id: string;
  tile_image_path: string | null;
  enabled: boolean;
  created_at: number;
}

export type ExportFormat = "lerobot_v3" | "h5_clean" | "zero_copy_index" | "viz_mp4";
export type ExportRecipeName = "bc_clean" | "recovery_plus" | "badcase_pairs";
export type SmoothTailMode = "on_jump" | "always" | "never";

export interface ExportRecipe {
  name: ExportRecipeName;
  format: ExportFormat;
  chunk_size: number | null;
  min_piece_s: number;
  tail_frames: number | null;
  smooth_tail: SmoothTailMode;
  include_statuses: EpisodeStatus[];
  context_window_s: number;
  dataset_filter: string | null;
  task_filter: string | null;
}

export interface ExportRequest {
  format: ExportFormat;
  recipe: ExportRecipeName;
  chunk_size?: number | null;
  min_piece_s?: number;
  tail_frames?: number | null;
  smooth_tail?: SmoothTailMode;
  include_statuses?: EpisodeStatus[];
  context_window_s?: number;
  dataset_filter?: string | null;
  task_filter?: string | null;
}

export interface ExportEpisodeEntry {
  episode_id: string;
  status: EpisodeStatus;
  passed: boolean | null;
  pieces_exported: number;
  pieces_dropped: number;
  own_frames: number;
  total_frames: number;
  dropped_reasons: string[];
}

export interface ExportManifest {
  export_id: string;
  created_at: number;
  format: ExportFormat;
  recipe: ExportRecipe;
  output_dir: string;
  episodes_included: number;
  episodes_skipped: number;
  pieces_exported: number;
  pieces_dropped: number;
  total_frames: number;
  episodes: ExportEpisodeEntry[];
  artifacts: string[];
  checksums: Record<string, string>;
}

export type JobType = "batch_vlm" | "eval_gt" | "export" | "render_mp4" | "ingest";
export type JobRunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobArtifactItem {
  name: string;
  path: string;
  kind: "mp4" | "export" | string;
  export_id?: string | null;
  episode_id?: string | null;
}

export interface JobSummary {
  artifacts?: JobArtifactItem[];
  results?: Array<Record<string, unknown>>;
  errors?: Array<{ episode_id?: string; error?: string }>;
  eval_run_id?: string | null;
  eval_id?: string | null;
  report?: EvalRunReport | null;
  eval_report?: EvalRunReport | null;
  metrics?: EvalMetrics | null;
  cache_hit_stages?: Record<string, number>;
  export_id?: string | null;
  manifest?: ExportManifest | null;
  discovered?: number;
  inserted?: number;
  updated?: number;
  skipped?: number;
  [key: string]: unknown;
}

export interface JobStatus {
  job_id: string;
  job_type: JobType;
  status: JobRunStatus;
  progress_done: number;
  progress_total: number;
  params?: Record<string, unknown>;
  summary?: JobSummary | null;
  created_at?: number;
  updated_at?: number;
}

export interface IngestJobParams {
  path?: string;
  data_path?: string;
  data_paths?: string[];
  adapter?: string;
  dataset?: string | null;
}

export interface CreateJobRequest {
  job_type: JobType;
  episode_ids?: string[];
  stage?: string;
  fresh?: boolean;
  concurrency?: number;
  is_eval_sandbox?: boolean;
  export_params?: ExportRequest | Record<string, unknown>;
  eval_params?: EvalRunRequest | Record<string, unknown>;
  ingest_params?: IngestJobParams | Record<string, unknown>;
}

export type PromptStageKey =
  | "s1_grounding"
  | "s2_macro"
  | "s3_discover"
  | "s4_verify"
  | "s5_success";

export interface PromptTemplate {
  stage: PromptStageKey;
  version: string;
  description?: string;
  system_template: string;
  user_template: string;
  extra_templates: Record<string, string>;
}

export interface PromptBundle {
  s1_grounding: PromptTemplate;
  s2_macro: PromptTemplate;
  s3_discover: PromptTemplate;
  s4_verify: PromptTemplate;
  s5_success: PromptTemplate;
}

export interface GroundingObjectSpec {
  id: string;
  name: string;
  desc: string;
  tag_pos: "top" | "bottom";
  is_static: boolean;
}

export interface TaskSpec {
  task_id: string;
  version: string;
  match: string[];
  aliases: string[];
  title: string;
  description: string;
  goal: string;
  criteria: string[];
  steps: string[];
  task_prompt: string;
  grounding_objects: GroundingObjectSpec[];
  hand_mapping?: Record<string, string> | null;
  camera_roles_override?: Record<string, string> | null;
}

export interface TaskListResponse {
  tasks: Record<string, TaskSpec>;
  total: number;
}

export interface DiagnosisErrorItem {
  error_id: string;
  episode_id: string;
  task_id: string;
  error_type: DiagnosisErrorType;
  label_id?: string | null;
  t_start?: number | null;
  t_end?: number | null;
  vlm_seg_id?: string | null;
  gt_seg_id?: string | null;
  vlm_value?: unknown;
  gt_value?: unknown;
  iou?: number | null;
  severity?: number;
  summary: string;
}

export interface DiagnosisCluster {
  cluster_key: string;
  error_type: DiagnosisErrorType;
  label_id?: string | null;
  task_id?: string | null;
  count: number;
  episode_ids: string[];
  sample_summary: string;
}

export interface DiagnosisReport {
  episodes_evaluated: number;
  total_errors: number;
  counts_by_type: Record<string, number>;
  counts_by_label: Record<string, number>;
  counts_by_task: Record<string, number>;
  errors: DiagnosisErrorItem[];
  top_errors: DiagnosisErrorItem[];
  clusters?: DiagnosisCluster[];
}

export interface UnknownAnomalyItem {
  episode_id: string;
  task_id?: string | null;
  dataset?: string | null;
  seg_id: string;
  t_start: number;
  t_end: number;
  label_id: string;
  affected_arm?: MaskSide;
  unknown_note: string;
  suggested_label?: string | null;
  visual_evidence?: VisualEvidence;
  splice_policy?: SplicePolicy;
  training_role?: TrainingRole;
  mask_side?: MaskSide | null;
  confidence?: number;
  status?: EpisodeStatus;
}

export interface UnknownListResponse {
  items: UnknownAnomalyItem[];
  total: number;
}

export type UnknownTriageAction = "merge" | "promote";

export interface UnknownTriageRequest {
  episode_id: string;
  seg_id: string;
  action: UnknownTriageAction;
  target_label_id?: string;
  rationale?: string;
  new_label?: VisualLabelDef;
}

export interface UnknownTriageResponse {
  ok: boolean;
  action: UnknownTriageAction;
  episode_id: string;
  seg_id: string;
  target_label_id?: string | null;
  few_shot_id?: string | null;
  label_id?: string | null;
  label_version?: number;
}

export interface FewShotListResponse {
  items: FewShotItem[];
  total: number;
}

export interface EvalRunRequest {
  dataset?: string | null;
  task_id?: string | null;
  episode_ids?: string[] | null;
  few_shot: boolean;
  top_k?: number;
  max_tokens?: number;
  workers?: number;
  candidate_prompts?: Partial<Record<PromptStageKey, PromptTemplate>> | Record<string, unknown> | null;
  candidate_labels?: LabelRegistry | Record<string, unknown> | null;
}

export interface EvalMetrics {
  episodes_evaluated: number;
  f1: number;
  precision: number;
  recall: number;
  mean_iou: number;
  overcut_rate: number;
  guardrail_downgrade_rate: number;
  success_accuracy: number;
  total_tokens: number;
  cache_hit_stages: Record<string, number>;
  cache_hit_rate: number;
}

export interface EvalPerEpisode {
  episode_id: string;
  task_id: string;
  sandbox_run_id: string;
  gt_rev_id?: string | null;
  precision: number;
  recall: number;
  f1: number;
  mean_iou: number;
  overcut_rate: number;
  guardrail_downgrade_rate: number;
  success_correct: boolean;
  pred_segments: number;
  gt_segments: number;
  matched_segments: number;
  tokens_total: number;
  cache_hit_stages: string[];
}

export interface EvalRunReport {
  eval_id: string;
  created_at: number;
  dataset?: string | null;
  task_id?: string | null;
  few_shot: boolean;
  metrics: EvalMetrics;
  per_episode: EvalPerEpisode[];
  config_snapshot: Record<string, unknown>;
  candidate_overrides?: Record<string, unknown>;
}

export interface EvalCompareDeltas {
  delta_f1: number;
  delta_precision: number;
  delta_recall: number;
  delta_mean_iou: number;
  delta_overcut_rate: number;
  delta_guardrail_downgrade_rate: number;
  delta_success_accuracy: number;
  delta_total_tokens: number;
  delta_cache_hit_rate: number;
}

export interface EvalCompareResponse {
  evals: EvalRunReport[];
  baseline?: EvalRunReport | null;
  candidate?: EvalRunReport | null;
  deltas?: EvalCompareDeltas | null;
}

export interface PromoteRequest {
  eval_id?: string | null;
  dataset?: string | null;
  candidate_prompts?: Record<string, unknown> | null;
  candidate_labels?: LabelRegistry | Record<string, unknown> | null;
  candidate_tasks?: Record<string, unknown> | null;
  reason?: string | null;
}

export interface PromoteResponse {
  ok: boolean;
  eval_id?: string | null;
  dataset?: string | null;
  promoted_files: string[];
  label_version?: number | null;
  config_sha256?: string;
}

export interface PromptAssistRequest {
  stage?: PromptStageKey;
  task_id?: string | null;
  episode_ids?: string[] | null;
  top_n?: number;
}

export interface PromptAssistResponse {
  stage: PromptStageKey | string;
  base_version: string;
  advisory_only: boolean;
  suggested_prompt_diff: string;
  proposed_system_template_patch?: string | null;
  proposed_user_template_patch?: string | null;
  rationale: string;
  task_spec_notes?: string | string[] | Record<string, unknown> | null;
}

export type EpisodeSortField =
  | "updated_at"
  | "duration_s"
  | "cut_ratio"
  | "mask_ratio"
  | "badcase_count"
  | "episode_id";

export type SortDir = "asc" | "desc";

export interface EpisodeFacets {
  datasets: string[];
  task_ids: string[];
  adapter_types: string[];
}

export interface CheckSpliceRequest {
  t_start: number;
  t_end: number;
  label_id: string;
  splice_policy: SplicePolicy;
  affected_arm: MaskSide;
}

export interface CheckSpliceMetrics {
  eef_pos_m: number | null;
  eef_rot_rad: number | null;
  hand_rad: number | null;
  torso_rad: number | null;
  other_arm_disp_m: number | null;
  bimanual_coupled: boolean;
  visual_ssim: number | null;
  visual_l1: number | null;
}

export interface CheckSpliceResponse {
  passed: boolean;
  downgraded: boolean;
  downgrade_reason: string | null;
  effective_splice_policy: SplicePolicy | string;
  opt_t_start: number | null;
  opt_t_end: number | null;
  metrics: CheckSpliceMetrics;
  compute_ms: number;
  mode: "kinematics" | "visual" | "skipped";
}

export interface WorkspaceSettings {
  min_watch_coverage: number;
  batch_vlm_concurrency: number;
}

export interface VlmRunSummary {
  run_id: string;
  created_at: string | number;
  prompt_version: string;
  label_version: string;
  few_shot_pack_id: string | null;
  tokens_total: number;
  is_eval_sandbox: boolean;
}

export interface EpisodeRunsResponse {
  episode_id: string;
  active_run_id: string | null;
  runs: VlmRunSummary[];
}

export interface IngestEntry {
  name: string;
  path: string;
  is_dir: boolean;
}

export interface IngestDetectedAdapter {
  adapter_type: string;
  episode_count: number;
}

export interface IngestBrowseResponse {
  current_path: string;
  parent_path: string | null;
  allowed_roots: string[];
  entries: IngestEntry[];
  detected_adapters: IngestDetectedAdapter[];
}

export type LabHistoryTarget = "labels" | "prompts" | "tasks" | "default";

export interface LabHistoryItem {
  snapshot_id: string;
  target: LabHistoryTarget | string;
  name: string;
  relative_path: string;
  created_at: number;
  reason: string;
  size_bytes: number;
}

export interface LabHistoryListResponse {
  items: LabHistoryItem[];
  total?: number;
}

export interface LabHistoryDiffLine {
  type: "context" | "add" | "remove" | "hunk";
  old_line: number | null;
  new_line: number | null;
  text: string;
}

export interface LabHistoryDiffResponse {
  snapshot_id: string;
  relative_path: string;
  created_at: number;
  reason: string;
  unified_diff: string;
  lines: LabHistoryDiffLine[];
}

export interface LabHistoryRollbackResponse {
  ok: boolean;
  relative_path: string;
  config_sha256: string;
}
