import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { EpisodeSortField, EpisodeStatus, SortDir } from "../api/types";
import { ALL_EPISODE_STATUSES } from "../constants/labels";

const VALID_STATUSES = new Set<string>(ALL_EPISODE_STATUSES);
const VALID_SORT_FIELDS = new Set<string>([
  "episode_id",
  "updated_at",
  "duration_s",
  "cut_ratio",
  "mask_ratio",
  "badcase_count",
]);

export type PassedFilterValue = "all" | "true" | "false";

function resolveUpdater<T>(current: T, next: T | ((prev: T) => T)): T {
  return typeof next === "function" ? (next as (prev: T) => T)(current) : next;
}

export interface QueueQueryParamsState {
  statusFilter: EpisodeStatus | "";
  setStatusFilter: (next: EpisodeStatus | "" | ((prev: EpisodeStatus | "") => EpisodeStatus | "")) => void;
  datasetFilter: string;
  setDatasetFilter: (next: string | ((prev: string) => string)) => void;
  taskFilter: string;
  setTaskFilter: (next: string | ((prev: string) => string)) => void;
  adapterFilter: string;
  setAdapterFilter: (next: string | ((prev: string) => string)) => void;
  passedFilter: PassedFilterValue;
  setPassedFilter: (next: PassedFilterValue | ((prev: PassedFilterValue) => PassedFilterValue)) => void;
  downgradedOnly: boolean;
  setDowngradedOnly: (next: boolean | ((prev: boolean) => boolean)) => void;
  searchQuery: string;
  debouncedSearch: string;
  setSearchQuery: (next: string | ((prev: string) => string)) => void;
  sortBy: EpisodeSortField;
  setSortBy: (next: EpisodeSortField | ((prev: EpisodeSortField) => EpisodeSortField)) => void;
  sortDir: SortDir;
  setSortDir: (next: SortDir | ((prev: SortDir) => SortDir)) => void;
  hasActiveFilters: boolean;
}

export function useQueueQueryParams(): QueueQueryParamsState {
  const [searchParams, setSearchParams] = useSearchParams();

  const searchKey = searchParams.toString();
  const prevSearchKeyRef = useRef(searchKey);
  const latestParamsRef = useRef<URLSearchParams>(new URLSearchParams(searchParams));

  if (prevSearchKeyRef.current !== searchKey) {
    prevSearchKeyRef.current = searchKey;
    latestParamsRef.current = new URLSearchParams(searchParams);
  }

  const mutateParams = useCallback(
    (mutator: (draft: URLSearchParams) => void) => {
      const next = new URLSearchParams(latestParamsRef.current);
      mutator(next);
      latestParamsRef.current = next;
      setSearchParams(next, { replace: true });
    },
    [setSearchParams],
  );

  const rawStatus = searchParams.get("status") ?? "";
  const statusFilter: EpisodeStatus | "" = VALID_STATUSES.has(rawStatus)
    ? (rawStatus as EpisodeStatus)
    : "";

  const datasetFilter = searchParams.get("dataset") ?? "";
  const taskFilter = searchParams.get("task_id") ?? "";
  const adapterFilter = searchParams.get("adapter_type") ?? "";

  const rawPassed = searchParams.get("passed") ?? "";
  const passedFilter: PassedFilterValue =
    rawPassed === "true" || rawPassed === "false" ? rawPassed : "all";

  const downgradedOnly = searchParams.get("guardrail_downgraded") === "true";

  const searchQuery = searchParams.get("search") ?? "";

  const rawSortBy = searchParams.get("sort_by") ?? "";
  const sortBy: EpisodeSortField = VALID_SORT_FIELDS.has(rawSortBy)
    ? (rawSortBy as EpisodeSortField)
    : "episode_id";

  const rawSortDir = searchParams.get("sort_dir") ?? "";
  const sortDir: SortDir = rawSortDir === "desc" ? "desc" : "asc";

  const [debouncedSearch, setDebouncedSearch] = useState(() => searchQuery.trim());

  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setDebouncedSearch("");
      return;
    }
    if (trimmed === debouncedSearch) return;
    const timer = window.setTimeout(() => {
      setDebouncedSearch(trimmed);
    }, 150);
    return () => window.clearTimeout(timer);
  }, [searchQuery, debouncedSearch]);

  const setStatusFilter = useCallback(
    (next: EpisodeStatus | "" | ((prev: EpisodeStatus | "") => EpisodeStatus | "")) => {
      mutateParams((draft) => {
        const curRaw = draft.get("status") ?? "";
        const cur: EpisodeStatus | "" = VALID_STATUSES.has(curRaw)
          ? (curRaw as EpisodeStatus)
          : "";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("status", resolved);
        else draft.delete("status");
      });
    },
    [mutateParams],
  );

  const setDatasetFilter = useCallback(
    (next: string | ((prev: string) => string)) => {
      mutateParams((draft) => {
        const cur = draft.get("dataset") ?? "";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("dataset", resolved);
        else draft.delete("dataset");
      });
    },
    [mutateParams],
  );

  const setTaskFilter = useCallback(
    (next: string | ((prev: string) => string)) => {
      mutateParams((draft) => {
        const cur = draft.get("task_id") ?? "";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("task_id", resolved);
        else draft.delete("task_id");
      });
    },
    [mutateParams],
  );

  const setAdapterFilter = useCallback(
    (next: string | ((prev: string) => string)) => {
      mutateParams((draft) => {
        const cur = draft.get("adapter_type") ?? "";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("adapter_type", resolved);
        else draft.delete("adapter_type");
      });
    },
    [mutateParams],
  );

  const setPassedFilter = useCallback(
    (next: PassedFilterValue | ((prev: PassedFilterValue) => PassedFilterValue)) => {
      mutateParams((draft) => {
        const curRaw = draft.get("passed") ?? "";
        const cur: PassedFilterValue =
          curRaw === "true" || curRaw === "false" ? curRaw : "all";
        const resolved = resolveUpdater(cur, next);
        if (resolved === "all") draft.delete("passed");
        else draft.set("passed", resolved);
      });
    },
    [mutateParams],
  );

  const setDowngradedOnly = useCallback(
    (next: boolean | ((prev: boolean) => boolean)) => {
      mutateParams((draft) => {
        const cur = draft.get("guardrail_downgraded") === "true";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("guardrail_downgraded", "true");
        else draft.delete("guardrail_downgraded");
      });
    },
    [mutateParams],
  );

  const setSearchQuery = useCallback(
    (next: string | ((prev: string) => string)) => {
      mutateParams((draft) => {
        const cur = draft.get("search") ?? "";
        const resolved = resolveUpdater(cur, next);
        if (resolved) draft.set("search", resolved);
        else draft.delete("search");
      });
    },
    [mutateParams],
  );

  const setSortBy = useCallback(
    (next: EpisodeSortField | ((prev: EpisodeSortField) => EpisodeSortField)) => {
      mutateParams((draft) => {
        const curRaw = draft.get("sort_by") ?? "";
        const cur: EpisodeSortField = VALID_SORT_FIELDS.has(curRaw)
          ? (curRaw as EpisodeSortField)
          : "episode_id";
        const resolved = resolveUpdater(cur, next);
        if (resolved === "episode_id") draft.delete("sort_by");
        else draft.set("sort_by", resolved);
      });
    },
    [mutateParams],
  );

  const setSortDir = useCallback(
    (next: SortDir | ((prev: SortDir) => SortDir)) => {
      mutateParams((draft) => {
        const cur: SortDir = draft.get("sort_dir") === "desc" ? "desc" : "asc";
        const resolved = resolveUpdater(cur, next);
        if (resolved === "asc") draft.delete("sort_dir");
        else draft.set("sort_dir", resolved);
      });
    },
    [mutateParams],
  );

  const hasActiveFilters = Boolean(
    statusFilter ||
      datasetFilter.trim() ||
      taskFilter.trim() ||
      adapterFilter ||
      passedFilter !== "all" ||
      downgradedOnly ||
      searchQuery.trim(),
  );

  return {
    statusFilter,
    setStatusFilter,
    datasetFilter,
    setDatasetFilter,
    taskFilter,
    setTaskFilter,
    adapterFilter,
    setAdapterFilter,
    passedFilter,
    setPassedFilter,
    downgradedOnly,
    setDowngradedOnly,
    searchQuery,
    debouncedSearch,
    setSearchQuery,
    sortBy,
    setSortBy,
    sortDir,
    setSortDir,
    hasActiveFilters,
  };
}
