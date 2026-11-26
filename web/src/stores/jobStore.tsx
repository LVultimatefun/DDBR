import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  cancelJob as apiCancelJob,
  clearTerminalJobs as apiClearTerminalJobs,
  deleteJob as apiDeleteJob,
  jobArtifactUrl,
  listJobs,
} from "../api/client";
import type { DeleteJobsResponse, JobClearScope, JobStatus, JobStatusCounts } from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { useAuth } from "./authStore";
import { useToast } from "./toastStore";

const EMPTY_COUNTS: JobStatusCounts = {
  queued: 0,
  running: 0,
  completed: 0,
  failed: 0,
  cancelled: 0,
};

function deriveCountsFromJobs(items: JobStatus[]): JobStatusCounts {
  const counts: JobStatusCounts = { ...EMPTY_COUNTS };
  for (const item of items) {
    if (item.status in counts) {
      counts[item.status] += 1;
    }
  }
  return counts;
}

export interface JobContextValue {
  isProvided: boolean;
  jobs: JobStatus[];
  totalJobs: number;
  statusCounts: JobStatusCounts;
  activeJobs: JobStatus[];
  drawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;
  refreshJobs: () => Promise<void>;
  trackJob: (job: JobStatus) => void;
  cancelJob: (jobId: string) => Promise<JobStatus | null>;
  deleteJob: (jobId: string) => Promise<DeleteJobsResponse | null>;
  clearJobs: (scope: JobClearScope) => Promise<DeleteJobsResponse | null>;
  artifactDownloadUrl: (jobId: string, index?: number) => string;
}

const JobContext = createContext<JobContextValue | null>(null);

function isJobActive(job: JobStatus): boolean {
  return job.status === "queued" || job.status === "running";
}

function upsertJobList(prev: JobStatus[], incoming: JobStatus): JobStatus[] {
  const idx = prev.findIndex((j) => j.job_id === incoming.job_id);
  if (idx >= 0) {
    const next = [...prev];
    next[idx] = incoming;
    return next;
  }
  return [incoming, ...prev];
}

export const JobProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isLead, token, noAuth, authProbing } = useAuth();
  const { showToast } = useToast();

  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const [totalJobs, setTotalJobs] = useState<number>(0);
  const [statusCounts, setStatusCounts] = useState<JobStatusCounts>(EMPTY_COUNTS);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const prevStatusRef = useRef<Map<string, JobStatus["status"]>>(new Map());

  const canSubscribe = isLead && (Boolean(token) || noAuth) && !authProbing;

  const notifyJobCancelled = useCallback(
    (job: JobStatus) => {
      showToast({
        message:
          job.job_type === "batch_vlm"
            ? `Job ${job.job_id} cancelled; unfinished episode statuses reverted`
            : `Job ${job.job_id} (${job.job_type}) cancelled`,
        variant: "warning",
        testId: `job-cancelled-toast-${job.job_id}`,
      });
    },
    [showToast],
  );

  const refreshJobs = useCallback(async () => {
    if (!canSubscribe) {
      setJobs([]);
      setTotalJobs(0);
      setStatusCounts(EMPTY_COUNTS);
      return;
    }
    try {
      const res = await listJobs({ limit: 20 });
      const items = Array.isArray(res.items) ? res.items : [];
      for (const item of items) {
        prevStatusRef.current.set(item.job_id, item.status);
      }
      setJobs(items);
      setTotalJobs(typeof res.total === "number" ? res.total : items.length);
      setStatusCounts(res.status_counts ?? deriveCountsFromJobs(items));
    } catch {
      // Non-fatal; global SSE or manual refresh can recover
    }
  }, [canSubscribe]);

  useEffect(() => {
    if (!canSubscribe) {
      setJobs([]);
      setTotalJobs(0);
      setStatusCounts(EMPTY_COUNTS);
      setDrawerOpen(false);
      prevStatusRef.current.clear();
      return;
    }
    void refreshJobs();
  }, [canSubscribe, refreshJobs]);

  const applyJobStatusCountDelta = useCallback(
    (prevStatus: JobStatus["status"] | undefined, nextStatus: JobStatus["status"]) => {
      if (prevStatus === nextStatus) return;
      setStatusCounts((prev) => {
        const next = { ...prev };
        if (prevStatus && prevStatus in next) {
          next[prevStatus] = Math.max(0, next[prevStatus] - 1);
        }
        if (nextStatus in next) {
          next[nextStatus] += 1;
        }
        return next;
      });
      if (!prevStatus) {
        setTotalJobs((prev) => prev + 1);
      }
    },
    [],
  );

  const trackJob = useCallback(
    (job: JobStatus) => {
      const prevStatus = prevStatusRef.current.get(job.job_id);
      prevStatusRef.current.set(job.job_id, job.status);
      setJobs((prev) => upsertJobList(prev, job));
      applyJobStatusCountDelta(prevStatus, job.status);
    },
    [applyJobStatusCountDelta],
  );

  useEventStream({
    url: canSubscribe ? "/api/events" : null,
    enabled: canSubscribe,
    onMessage: (evt: Record<string, unknown>) => {
      if (evt.type === "jobs_deleted") {
        const deletedIds = Array.isArray(evt.job_ids)
          ? evt.job_ids.map((v) => String(v))
          : [];
        if (deletedIds.length > 0) {
          const delSet = new Set(deletedIds);
          for (const id of delSet) {
            prevStatusRef.current.delete(id);
          }
          setJobs((prev) => prev.filter((j) => !delSet.has(j.job_id)));
          if (evt.status_counts && typeof evt.status_counts === "object") {
            const counts = evt.status_counts as JobStatusCounts;
            setStatusCounts(counts);
            setTotalJobs(
              counts.queued + counts.running + counts.completed + counts.failed + counts.cancelled,
            );
          } else {
            void refreshJobs();
          }
        }
        return;
      }

      if (evt.type !== "job_update" || !evt.job || typeof evt.job !== "object") {
        return;
      }
      const incoming = evt.job as JobStatus;
      if (!incoming.job_id) return;

      const prevStatus = prevStatusRef.current.get(incoming.job_id);
      prevStatusRef.current.set(incoming.job_id, incoming.status);
      setJobs((prev) => upsertJobList(prev, incoming));
      applyJobStatusCountDelta(prevStatus, incoming.status);

      const wasActive = prevStatus === "queued" || prevStatus === "running";
      if (wasActive && incoming.status === "completed") {
        showToast({
          message: `Job ${incoming.job_id} (${incoming.job_type}) completed`,
          variant: "success",
        });
      } else if (wasActive && incoming.status === "failed") {
        showToast({
          message: `Job ${incoming.job_id} (${incoming.job_type}) failed`,
          variant: "error",
        });
      } else if (wasActive && incoming.status === "cancelled") {
        notifyJobCancelled(incoming);
      }
    },
  });

  const handleCancelJob = useCallback(
    async (jobId: string): Promise<JobStatus | null> => {
      const prevStatusBeforeCall = prevStatusRef.current.get(jobId);
      try {
        const updated = await apiCancelJob(jobId);
        const latestPrev = prevStatusRef.current.get(updated.job_id) ?? prevStatusBeforeCall;
        prevStatusRef.current.set(updated.job_id, updated.status);
        setJobs((prev) => upsertJobList(prev, updated));
        applyJobStatusCountDelta(latestPrev, updated.status);
        if (updated.status === "cancelled" && latestPrev !== "cancelled") {
          notifyJobCancelled(updated);
        }
        return updated;
      } catch (err) {
        showToast({
          message: err instanceof Error ? `Failed to cancel job: ${err.message}` : "Failed to cancel job",
          variant: "error",
        });
        return null;
      }
    },
    [applyJobStatusCountDelta, notifyJobCancelled, showToast],
  );

  const handleDeleteJob = useCallback(
    async (jobId: string): Promise<DeleteJobsResponse | null> => {
      try {
        const res = await apiDeleteJob(jobId);
        prevStatusRef.current.delete(jobId);
        setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
        if (res.status_counts) {
          setStatusCounts(res.status_counts);
          setTotalJobs(
            res.status_counts.queued +
              res.status_counts.running +
              res.status_counts.completed +
              res.status_counts.failed +
              res.status_counts.cancelled,
          );
        } else {
          setTotalJobs((prev) => Math.max(0, prev - 1));
        }
        showToast({
          message: `Deleted job record ${jobId}`,
          variant: "info",
          testId: `job-deleted-toast-${jobId}`,
        });
        void refreshJobs();
        return res;
      } catch (err) {
        showToast({
          message: err instanceof Error ? `Failed to delete job: ${err.message}` : "Failed to delete job",
          variant: "error",
        });
        return null;
      }
    },
    [refreshJobs, showToast],
  );

  const handleClearJobs = useCallback(
    async (scope: JobClearScope): Promise<DeleteJobsResponse | null> => {
      try {
        const res = await apiClearTerminalJobs(scope);
        const delSet = new Set(res.job_ids);
        for (const id of delSet) {
          prevStatusRef.current.delete(id);
        }
        setJobs((prev) => prev.filter((j) => !delSet.has(j.job_id)));
        if (res.status_counts) {
          setStatusCounts(res.status_counts);
          setTotalJobs(
            res.status_counts.queued +
              res.status_counts.running +
              res.status_counts.completed +
              res.status_counts.failed +
              res.status_counts.cancelled,
          );
        }
        showToast({
          message: `Cleared ${res.deleted} job record(s)`,
          variant: "info",
          testId: `jobs-cleared-toast-${scope}`,
        });
        void refreshJobs();
        return res;
      } catch (err) {
        showToast({
          message: err instanceof Error ? `Failed to clear jobs: ${err.message}` : "Failed to clear jobs",
          variant: "error",
        });
        return null;
      }
    },
    [refreshJobs, showToast],
  );

  const artifactDownloadUrl = useCallback(
    (jobId: string, index = 0) => jobArtifactUrl(jobId, index, token),
    [token],
  );

  const activeJobs = useMemo(() => jobs.filter(isJobActive), [jobs]);

  const value = useMemo<JobContextValue>(
    () => ({
      isProvided: true,
      jobs,
      totalJobs,
      statusCounts,
      activeJobs,
      drawerOpen,
      setDrawerOpen,
      refreshJobs,
      trackJob,
      cancelJob: handleCancelJob,
      deleteJob: handleDeleteJob,
      clearJobs: handleClearJobs,
      artifactDownloadUrl,
    }),
    [
      jobs,
      totalJobs,
      statusCounts,
      activeJobs,
      drawerOpen,
      refreshJobs,
      trackJob,
      handleCancelJob,
      handleDeleteJob,
      handleClearJobs,
      artifactDownloadUrl,
    ],
  );

  return <JobContext.Provider value={value}>{children}</JobContext.Provider>;
};

const NOOP_JOB_CONTEXT: JobContextValue = {
  isProvided: false,
  jobs: [],
  totalJobs: 0,
  statusCounts: EMPTY_COUNTS,
  activeJobs: [],
  drawerOpen: false,
  setDrawerOpen: () => {},
  refreshJobs: async () => {},
  trackJob: () => {},
  cancelJob: async (jobId: string) => apiCancelJob(jobId),
  deleteJob: async (jobId: string) => apiDeleteJob(jobId),
  clearJobs: async (scope: JobClearScope) => apiClearTerminalJobs(scope),
  artifactDownloadUrl: (jobId: string, index = 0) => jobArtifactUrl(jobId, index),
};

export function useJobs(): JobContextValue {
  return useContext(JobContext) ?? NOOP_JOB_CONTEXT;
}
