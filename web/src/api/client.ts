import { normalizeSegmentTriplet } from "../constants/labels";
import type {
  AssignEpisodeTaskRequest,
  AssignEpisodeTaskResponse,
  BatchAssignEpisodeTaskRequest,
  BatchAssignEpisodeTaskResponse,
  CheckSpliceRequest,
  CheckSpliceResponse,
  CreateJobRequest,
  DeleteJobsResponse,
  DiagnosisReport,
  EpisodeFacets,
  EpisodeListItem,
  EpisodeRunsResponse,
  EpisodeSortField,
  EpisodeStatus,
  EpisodeTaskCatalogResponse,
  EpisodeTaskSummary,
  EvalCompareResponse,
  EvalRunReport,
  EvalRunRequest,
  ExportManifest,
  ExportRequest,
  FewShotCreateRequest,
  FewShotItem,
  FewShotListResponse,
  HumanReviewArtifact,
  InferSubtasksRequest,
  InferSubtasksResponse,
  IngestBrowseResponse,
  JobClearScope,
  JobEstimateRequest,
  JobEstimateResponse,
  JobRunStatus,
  JobStatus,
  JobStatusCounts,
  JobType,
  LabHistoryDiffResponse,
  LabHistoryItem,
  LabHistoryListResponse,
  LabHistoryRollbackResponse,
  LabHistoryTarget,
  LabelRegistry,
  MacroStage,
  PromoteRequest,
  PromoteResponse,
  PromptAssistRequest,
  PromptAssistResponse,
  PromptBundle,
  QcSegment,
  ResolvedEpisode,
  ReviewAction,
  ReviewRole,
  ReviewWritePayload,
  SortDir,
  SuccessVerdict,
  TaskDraftRequest,
  TaskDraftResponse,
  TaskListResponse,
  TaskSpec,
  UnknownAnomalyItem,
  UnknownListResponse,
  UnknownTriageRequest,
  UnknownTriageResponse,
  VlmSettingsResponse,
  VlmSettingsUpdatePayload,
  VlmTestPayload,
  VlmTestResponse,
  WatchCoverage,
  WorkspaceSettings,
} from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface AuthCredentials {
  token: string;
  username: string;
}

let _credentialsProvider: () => AuthCredentials = () => ({ token: "", username: "reviewer" });

export function configureApiCredentialsProvider(provider: () => AuthCredentials): void {
  _credentialsProvider = provider;
}

export function encodeEpPath(epId: string): string {
  return epId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function appendToken(url: string, token: string): string {
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(token)}`;
}

export function previewVideoUrl(epId: string, camera?: string, tokenOverride?: string): string {
  const token = tokenOverride ?? _credentialsProvider().token;
  const base = `/api/episodes/${encodeEpPath(epId)}/preview`;
  const qs = camera ? `?camera=${encodeURIComponent(camera)}` : "";
  return appendToken(`${base}${qs}`, token);
}

export function frameUrl(
  epId: string,
  camera: string,
  tS: number,
  tokenOverride?: string,
): string {
  const token = tokenOverride ?? _credentialsProvider().token;
  const safeT = Math.max(0, tS);
  const url = `/api/episodes/${encodeEpPath(epId)}/frame?camera=${encodeURIComponent(camera)}&t_s=${safeT.toFixed(4)}`;
  return appendToken(url, token);
}

export function exportDownloadUrl(exportId: string, tokenOverride?: string): string {
  const token = tokenOverride ?? _credentialsProvider().token;
  return appendToken(`/api/exports/${encodeURIComponent(exportId)}/download`, token);
}

/**
 * Strictly builds ReviewWritePayload(extra="forbid"):
 * ONLY { base_run_id, action, qc_segments, success, macro_stages, watch_coverage, notes?, flagged_reason? }.
 * NEVER includes author, grounding, arm_timelines, or raw_vlm_segments.
 */
export function buildReviewWritePayload(input: {
  baseRunId: string;
  action: ReviewAction;
  qcSegments: QcSegment[];
  success: SuccessVerdict;
  macroStages: MacroStage[];
  watchCoverage: WatchCoverage;
  notes?: string;
  flaggedReason?: string | null;
}): ReviewWritePayload {
  const normalizedSegments = input.qcSegments.map((seg) => normalizeSegmentTriplet(seg));
  const payload: ReviewWritePayload = {
    base_run_id: input.baseRunId,
    action: input.action,
    qc_segments: normalizedSegments,
    success: input.success,
    macro_stages: input.macroStages,
    watch_coverage: input.watchCoverage,
  };
  if (input.notes !== undefined) {
    payload.notes = input.notes;
  }
  if (input.flaggedReason !== undefined) {
    payload.flagged_reason = input.flaggedReason;
  }
  return payload;
}

function unwrapItems<T>(data: unknown): T[] {
  if (Array.isArray(data)) return data as T[];
  if (data && typeof data === "object") {
    const record = data as Record<string, unknown>;
    if (Array.isArray(record.items)) return record.items as T[];
    if (Array.isArray(record.episodes)) return record.episodes as T[];
  }
  return [];
}

async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  overrideCreds?: Partial<AuthCredentials>,
): Promise<T> {
  const creds = _credentialsProvider();
  const token = overrideCreds?.token ?? creds.token;
  const username = overrideCreds?.username ?? creds.username;

  const headers = new Headers(init.headers);
  if (token) {
    headers.set("X-Auth-Token", token);
  }
  if (username) {
    headers.set("X-User", username);
  }
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && typeof body === "object") {
        const d = (body as Record<string, unknown>).detail;
        if (typeof d === "string") detail = d;
        else if (d !== undefined) detail = JSON.stringify(d);
      }
    } catch {
      const text = await res.text().catch(() => "");
      if (text) detail = text;
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export interface EpisodeFilters {
  status?: EpisodeStatus | string;
  dataset?: string;
  task_id?: string;
  adapter_type?: string;
  passed?: boolean | null;
  guardrail_downgraded?: boolean;
  claimed_by?: string;
  search?: string;
  sort_by?: EpisodeSortField;
  sort_dir?: SortDir;
  limit?: number;
  offset?: number;
}

export interface EpisodeListResponse {
  items: EpisodeListItem[];
  total: number;
  my_claimed_count?: number;
  status_counts?: Record<string, number>;
  facets?: EpisodeFacets;
}

export async function listEpisodes(filters: EpisodeFilters = {}): Promise<EpisodeListResponse> {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.dataset) params.set("dataset", filters.dataset);
  if (filters.task_id) params.set("task_id", filters.task_id);
  if (filters.adapter_type) params.set("adapter_type", filters.adapter_type);
  if (filters.passed !== undefined && filters.passed !== null) {
    params.set("passed", String(filters.passed));
  }
  if (filters.guardrail_downgraded) {
    params.set("guardrail_downgraded", "true");
  }
  if (filters.claimed_by) params.set("claimed_by", filters.claimed_by);
  if (filters.search) params.set("search", filters.search);
  if (filters.sort_by) params.set("sort_by", filters.sort_by);
  if (filters.sort_dir) params.set("sort_dir", filters.sort_dir);
  if (filters.limit !== undefined) params.set("limit", String(filters.limit));
  if (filters.offset !== undefined) params.set("offset", String(filters.offset));

  const qs = params.toString();
  const data = await apiFetch<unknown>(`/api/episodes${qs ? `?${qs}` : ""}`);
  const record = data && typeof data === "object" ? (data as Record<string, unknown>) : null;
  const items = unwrapItems<EpisodeListItem>(data);
  const total =
    record && typeof record.total === "number" ? (record.total as number) : items.length;
  const my_claimed_count =
    record && typeof record.my_claimed_count === "number"
      ? (record.my_claimed_count as number)
      : undefined;

  let status_counts: Record<string, number> | undefined;
  if (record && record.status_counts && typeof record.status_counts === "object") {
    status_counts = record.status_counts as Record<string, number>;
  }

  let facets: EpisodeFacets | undefined;
  if (record && record.facets && typeof record.facets === "object") {
    const rawFacets = record.facets as Record<string, unknown>;
    facets = {
      datasets: Array.isArray(rawFacets.datasets) ? (rawFacets.datasets as string[]) : [],
      task_ids: Array.isArray(rawFacets.task_ids) ? (rawFacets.task_ids as string[]) : [],
      adapter_types: Array.isArray(rawFacets.adapter_types)
        ? (rawFacets.adapter_types as string[])
        : [],
    };
  }

  return { items, total, my_claimed_count, status_counts, facets };
}

export async function getEpisode(epId: string, runId?: string): Promise<ResolvedEpisode> {
  const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiFetch<ResolvedEpisode>(`/api/episodes/${encodeEpPath(epId)}${qs}`);
}

export async function listEpisodeTasks(): Promise<EpisodeTaskCatalogResponse> {
  const data = await apiFetch<unknown>("/api/episodes/tasks");
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const rec = data as Record<string, unknown>;
    if (Array.isArray(rec.tasks)) {
      const tasks = rec.tasks as EpisodeTaskSummary[];
      const total = typeof rec.total === "number" ? rec.total : tasks.length;
      return { tasks, total };
    }
  }
  return { tasks: [], total: 0 };
}

export async function assignEpisodeTask(
  epId: string,
  body: AssignEpisodeTaskRequest,
): Promise<AssignEpisodeTaskResponse> {
  const payload: AssignEpisodeTaskRequest = {
    task_id: body.task_id,
    ...(body.allow_frozen !== undefined ? { allow_frozen: body.allow_frozen } : {}),
  };
  return apiFetch<AssignEpisodeTaskResponse>(`/api/episodes/${encodeEpPath(epId)}/task`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function batchAssignEpisodeTask(
  body: BatchAssignEpisodeTaskRequest,
): Promise<BatchAssignEpisodeTaskResponse> {
  const payload: BatchAssignEpisodeTaskRequest = {
    episode_ids: body.episode_ids,
    task_id: body.task_id,
    ...(body.allow_frozen !== undefined ? { allow_frozen: body.allow_frozen } : {}),
  };
  return apiFetch<BatchAssignEpisodeTaskResponse>("/api/episodes/batch-task", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listEpisodeRuns(epId: string): Promise<EpisodeRunsResponse> {
  return apiFetch<EpisodeRunsResponse>(`/api/episodes/${encodeEpPath(epId)}/runs`);
}

export async function checkSplice(
  epId: string,
  body: CheckSpliceRequest,
): Promise<CheckSpliceResponse> {
  return apiFetch<CheckSpliceResponse>(`/api/episodes/${encodeEpPath(epId)}/check-splice`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function inferSubtasks(
  episodeId: string,
  payload: InferSubtasksRequest,
): Promise<InferSubtasksResponse> {
  return apiFetch<InferSubtasksResponse>(
    `/api/episodes/${encodeEpPath(episodeId)}/infer-subtasks`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function getSettings(): Promise<WorkspaceSettings> {
  return apiFetch<WorkspaceSettings>("/api/settings");
}

export async function updateSettings(
  body: Partial<WorkspaceSettings>,
): Promise<WorkspaceSettings> {
  return apiFetch<WorkspaceSettings>("/api/settings", {
    method: "PUT",
    body: JSON.stringify(body),
  });
}

export const putSettings = updateSettings;

export async function getVlmSettings(): Promise<VlmSettingsResponse> {
  return apiFetch<VlmSettingsResponse>("/api/settings/vlm");
}

export async function updateVlmSettings(
  payload: VlmSettingsUpdatePayload,
): Promise<VlmSettingsResponse> {
  return apiFetch<VlmSettingsResponse>("/api/settings/vlm", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function testVlmSettings(
  payload?: VlmTestPayload,
): Promise<VlmTestResponse> {
  return apiFetch<VlmTestResponse>("/api/settings/vlm/test", {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export interface ClaimResponse {
  episode_id: string;
  claimed_by: string;
  claim_expires_at: number;
  status?: EpisodeStatus;
}

export async function claimEpisode(epId: string, user: string): Promise<ClaimResponse> {
  return apiFetch<ClaimResponse>(`/api/episodes/${encodeEpPath(epId)}/claim`, {
    method: "POST",
    body: JSON.stringify({ user }),
  });
}

export async function renewClaim(epId: string, user: string): Promise<ClaimResponse> {
  return apiFetch<ClaimResponse>(`/api/episodes/${encodeEpPath(epId)}/renew`, {
    method: "POST",
    body: JSON.stringify({ user }),
  });
}

export async function releaseClaim(epId: string, user: string): Promise<{ episode_id: string; status: EpisodeStatus }> {
  return apiFetch<{ episode_id: string; status: EpisodeStatus }>(
    `/api/episodes/${encodeEpPath(epId)}/release`,
    {
      method: "POST",
      body: JSON.stringify({ user }),
    },
  );
}

export async function listReviews(epId: string): Promise<HumanReviewArtifact[]> {
  const data = await apiFetch<unknown>(`/api/episodes/${encodeEpPath(epId)}/reviews`);
  return unwrapItems<HumanReviewArtifact>(data);
}

export async function submitReview(
  epId: string,
  payload: ReviewWritePayload,
): Promise<HumanReviewArtifact> {
  return apiFetch<HumanReviewArtifact>(`/api/episodes/${encodeEpPath(epId)}/reviews`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function createJob(body: CreateJobRequest): Promise<JobStatus> {
  return apiFetch<JobStatus>("/api/jobs", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function estimateJobCost(body: JobEstimateRequest): Promise<JobEstimateResponse> {
  return apiFetch<JobEstimateResponse>("/api/jobs/estimate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listJobs(params?: {
  status?: JobRunStatus | string;
  job_type?: JobType | string;
  limit?: number;
  offset?: number;
}): Promise<{ items: JobStatus[]; total: number; status_counts?: JobStatusCounts }> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.job_type) qs.set("job_type", params.job_type);
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  const q = qs.toString();
  const data = await apiFetch<unknown>(`/api/jobs${q ? `?${q}` : ""}`);
  const items = unwrapItems<JobStatus>(data);
  const rec = data && typeof data === "object" ? (data as Record<string, unknown>) : null;
  const total = rec && typeof rec.total === "number" ? (rec.total as number) : items.length;
  const status_counts =
    rec && rec.status_counts && typeof rec.status_counts === "object"
      ? (rec.status_counts as JobStatusCounts)
      : undefined;
  return { items, total, status_counts };
}

export async function getJob(jobId: string): Promise<JobStatus> {
  return apiFetch<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}`);
}

export async function cancelJob(jobId: string): Promise<JobStatus> {
  return apiFetch<JobStatus>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

export async function deleteJob(jobId: string): Promise<DeleteJobsResponse> {
  return apiFetch<DeleteJobsResponse>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}

export async function clearTerminalJobs(
  scope: JobClearScope = "finished",
  jobType?: JobType | string,
): Promise<DeleteJobsResponse> {
  const qs = new URLSearchParams({ scope });
  if (jobType) qs.set("job_type", jobType);
  return apiFetch<DeleteJobsResponse>(`/api/jobs?${qs.toString()}`, {
    method: "DELETE",
  });
}

export function jobArtifactUrl(jobId: string, index = 0, tokenOverride?: string): string {
  const token = tokenOverride ?? _credentialsProvider().token;
  return appendToken(`/api/jobs/${encodeURIComponent(jobId)}/artifact?index=${index}`, token);
}

export async function browseIngestDir(path?: string): Promise<IngestBrowseResponse> {
  const qs = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiFetch<IngestBrowseResponse>(`/api/ingest/browse${qs}`);
}

export const browseIngest = browseIngestDir;

export async function dryRunExport(req: ExportRequest): Promise<ExportManifest> {
  return apiFetch<ExportManifest>("/api/exports/dry-run", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function triggerExport(req: ExportRequest): Promise<ExportManifest> {
  return apiFetch<ExportManifest>("/api/exports", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function listExports(): Promise<ExportManifest[]> {
  const data = await apiFetch<unknown>("/api/exports");
  return unwrapItems<ExportManifest>(data);
}

export async function getExport(exportId: string): Promise<ExportManifest> {
  return apiFetch<ExportManifest>(`/api/exports/${encodeURIComponent(exportId)}`);
}

export interface AuthMeResponse {
  role: ReviewRole;
  user: string;
  no_auth?: boolean;
}

export async function getAuthMe(
  overrideCreds?: Partial<AuthCredentials>,
): Promise<AuthMeResponse> {
  return apiFetch<AuthMeResponse>("/api/auth/me", {}, overrideCreds);
}

export async function getLabLabels(overrideCreds?: Partial<AuthCredentials>): Promise<LabelRegistry> {
  return apiFetch<LabelRegistry>("/api/lab/labels", {}, overrideCreds);
}

export async function putLabLabels(registry: LabelRegistry): Promise<LabelRegistry> {
  return apiFetch<LabelRegistry>("/api/lab/labels", {
    method: "PUT",
    body: JSON.stringify(registry),
  });
}

export async function getSharedLabels(overrideCreds?: Partial<AuthCredentials>): Promise<LabelRegistry> {
  return apiFetch<LabelRegistry>("/api/labels", {}, overrideCreds);
}

export async function getLabPrompts(): Promise<PromptBundle> {
  return apiFetch<PromptBundle>("/api/lab/prompts");
}

export async function putLabPrompts(bundle: PromptBundle): Promise<PromptBundle> {
  return apiFetch<PromptBundle>("/api/lab/prompts", {
    method: "PUT",
    body: JSON.stringify(bundle),
  });
}

export async function getLabTasks(): Promise<TaskListResponse> {
  return apiFetch<TaskListResponse>("/api/lab/tasks");
}

export async function putLabTasks(
  payload: TaskSpec | { tasks: Record<string, TaskSpec> },
): Promise<TaskListResponse> {
  return apiFetch<TaskListResponse>("/api/lab/tasks", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function draftLabTask(body: TaskDraftRequest): Promise<TaskDraftResponse> {
  return apiFetch<TaskDraftResponse>("/api/lab/tasks/draft", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getLabDiagnosis(params?: {
  episode_ids?: string[];
  top_n?: number;
}): Promise<DiagnosisReport> {
  const qs = new URLSearchParams();
  if (params?.episode_ids) {
    for (const id of params.episode_ids) {
      const trimmed = id.trim();
      if (trimmed) qs.append("episode_ids", trimmed);
    }
  }
  if (params?.top_n !== undefined) {
    qs.set("top_n", String(params.top_n));
  }
  const q = qs.toString();
  return apiFetch<DiagnosisReport>(`/api/lab/diagnosis${q ? `?${q}` : ""}`);
}

export async function getLabUnknowns(params?: {
  dataset?: string;
  limit?: number;
}): Promise<UnknownListResponse> {
  const qs = new URLSearchParams();
  if (params?.dataset?.trim()) qs.set("dataset", params.dataset.trim());
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  const q = qs.toString();
  const data = await apiFetch<unknown>(`/api/lab/unknowns${q ? `?${q}` : ""}`);
  const items = unwrapItems<UnknownAnomalyItem>(data);
  const total =
    data && typeof data === "object" && typeof (data as Record<string, unknown>).total === "number"
      ? ((data as Record<string, unknown>).total as number)
      : items.length;
  return { items, total };
}

export async function triageUnknown(body: UnknownTriageRequest): Promise<UnknownTriageResponse> {
  return apiFetch<UnknownTriageResponse>("/api/lab/unknowns/triage", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listFewShots(params?: {
  task_id?: string;
  label_id?: string;
  scope?: string;
  enabled_only?: boolean;
}): Promise<FewShotListResponse> {
  const qs = new URLSearchParams();
  if (params?.task_id?.trim()) qs.set("task_id", params.task_id.trim());
  if (params?.label_id?.trim()) qs.set("label_id", params.label_id.trim());
  if (params?.scope?.trim()) qs.set("scope", params.scope.trim());
  if (params?.enabled_only !== undefined) qs.set("enabled_only", String(params.enabled_only));
  const q = qs.toString();
  const data = await apiFetch<unknown>(`/api/lab/few-shots${q ? `?${q}` : ""}`);
  const items = unwrapItems<FewShotItem>(data);
  const total =
    data && typeof data === "object" && typeof (data as Record<string, unknown>).total === "number"
      ? ((data as Record<string, unknown>).total as number)
      : items.length;
  return { items, total };
}

export async function createFewShot(body: FewShotCreateRequest): Promise<FewShotItem> {
  return apiFetch<FewShotItem>("/api/lab/few-shots", {
    method: "POST",
    body: JSON.stringify({ capture_tile: true, ...body }),
  });
}

export async function toggleFewShotEnabled(exampleId: string, enabled: boolean): Promise<FewShotItem> {
  return apiFetch<FewShotItem>(`/api/lab/few-shots/${encodeURIComponent(exampleId)}`, {
    method: "PATCH",
    body: JSON.stringify({ enabled }),
  });
}

export function fewShotTileUrl(exampleId: string, tokenOverride?: string): string {
  const token = tokenOverride ?? _credentialsProvider().token;
  return appendToken(`/api/lab/few-shots/${encodeURIComponent(exampleId)}/tile`, token);
}

export async function runLabEval(body: EvalRunRequest): Promise<EvalRunReport> {
  return apiFetch<EvalRunReport>("/api/lab/eval", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function compareLabEvals(params?: {
  baseline?: string;
  candidate?: string;
}): Promise<EvalCompareResponse> {
  const qs = new URLSearchParams();
  if (params?.baseline) qs.set("baseline", params.baseline);
  if (params?.candidate) qs.set("candidate", params.candidate);
  const q = qs.toString();
  return apiFetch<EvalCompareResponse>(`/api/lab/eval/compare${q ? `?${q}` : ""}`);
}

export async function promoteLabConfig(body: PromoteRequest): Promise<PromoteResponse> {
  return apiFetch<PromoteResponse>("/api/lab/promote", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function requestPromptAssist(body: PromptAssistRequest): Promise<PromptAssistResponse> {
  return apiFetch<PromptAssistResponse>("/api/lab/prompt-assist", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function listLabHistory(params?: {
  target?: LabHistoryTarget | string;
  name?: string;
}): Promise<LabHistoryListResponse> {
  const qs = new URLSearchParams();
  if (params?.target) qs.set("target", params.target);
  if (params?.name?.trim()) qs.set("name", params.name.trim());
  const q = qs.toString();
  const data = await apiFetch<unknown>(`/api/lab/history${q ? `?${q}` : ""}`);
  const items = unwrapItems<LabHistoryItem>(data);
  const total =
    data && typeof data === "object" && typeof (data as Record<string, unknown>).total === "number"
      ? ((data as Record<string, unknown>).total as number)
      : items.length;
  return { items, total };
}

export const getLabHistory = listLabHistory;

export async function getLabHistoryDiff(snapshotId: string): Promise<LabHistoryDiffResponse> {
  return apiFetch<LabHistoryDiffResponse>(
    `/api/lab/history/${encodeURIComponent(snapshotId)}/diff`,
  );
}

export async function rollbackLabHistory(
  snapshotId: string,
): Promise<LabHistoryRollbackResponse> {
  return apiFetch<LabHistoryRollbackResponse>(
    `/api/lab/history/${encodeURIComponent(snapshotId)}/rollback`,
    { method: "POST" },
  );
}
