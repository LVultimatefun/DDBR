import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { cancelJob as apiCancelJob, jobArtifactUrl, listJobs } from "../api/client";
import type { JobStatus } from "../api/types";
import { useEventStream } from "../api/useEventStream";
import { useAuth } from "./authStore";
import { useToast } from "./toastStore";

export interface JobContextValue {
  isProvided: boolean;
  jobs: JobStatus[];
  activeJobs: JobStatus[];
  drawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;
  refreshJobs: () => Promise<void>;
  trackJob: (job: JobStatus) => void;
  cancelJob: (jobId: string) => Promise<JobStatus | null>;
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
      return;
    }
    try {
      const res = await listJobs({ limit: 20 });
      const items = Array.isArray(res.items) ? res.items : [];
      for (const item of items) {
        prevStatusRef.current.set(item.job_id, item.status);
      }
      setJobs(items);
    } catch {
      // Non-fatal; global SSE or manual refresh can recover
    }
  }, [canSubscribe]);

  useEffect(() => {
    if (!canSubscribe) {
      setJobs([]);
      setDrawerOpen(false);
      prevStatusRef.current.clear();
      return;
    }
    void refreshJobs();
  }, [canSubscribe, refreshJobs]);

  const trackJob = useCallback((job: JobStatus) => {
    prevStatusRef.current.set(job.job_id, job.status);
    setJobs((prev) => upsertJobList(prev, job));
  }, []);

  useEventStream({
    url: canSubscribe ? "/api/events" : null,
    enabled: canSubscribe,
    onMessage: (evt: Record<string, unknown>) => {
      if (evt.type !== "job_update" || !evt.job || typeof evt.job !== "object") {
        return;
      }
      const incoming = evt.job as JobStatus;
      if (!incoming.job_id) return;

      const prevStatus = prevStatusRef.current.get(incoming.job_id);
      prevStatusRef.current.set(incoming.job_id, incoming.status);
      setJobs((prev) => upsertJobList(prev, incoming));

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
    [notifyJobCancelled, showToast],
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
      activeJobs,
      drawerOpen,
      setDrawerOpen,
      refreshJobs,
      trackJob,
      cancelJob: handleCancelJob,
      artifactDownloadUrl,
    }),
    [jobs, activeJobs, drawerOpen, refreshJobs, trackJob, handleCancelJob, artifactDownloadUrl],
  );

  return <JobContext.Provider value={value}>{children}</JobContext.Provider>;
};

const NOOP_JOB_CONTEXT: JobContextValue = {
  isProvided: false,
  jobs: [],
  activeJobs: [],
  drawerOpen: false,
  setDrawerOpen: () => {},
  refreshJobs: async () => {},
  trackJob: () => {},
  cancelJob: async (jobId: string) => apiCancelJob(jobId),
  artifactDownloadUrl: (jobId: string, index = 0) => jobArtifactUrl(jobId, index),
};

export function useJobs(): JobContextValue {
  return useContext(JobContext) ?? NOOP_JOB_CONTEXT;
}
