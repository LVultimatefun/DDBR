import { encodeEpPath } from "../../api/client";
import type {
  DiagnosisErrorItem,
  DiagnosisErrorType,
  LabelCategory,
  PromptStageKey,
  SplicePolicy,
  TrainingRole,
} from "../../api/types";
import type { TranslateFn } from "../../stores/i18nStore";

export type LabTab = "diagnosis" | "few_shots" | "taxonomy" | "eval";
export type TaxonomySubTab = "labels" | "prompts" | "tasks";

export const DIAGNOSIS_BUCKETS: Array<{ key: DiagnosisErrorType; labelKey: string }> = [
  { key: "fp_overflag", labelKey: "lab.bucket.fp_overflag" },
  { key: "fn_missed", labelKey: "lab.bucket.fn_missed" },
  { key: "policy_flip", labelKey: "lab.bucket.policy_flip" },
  { key: "boundary", labelKey: "lab.bucket.boundary" },
  { key: "success_flip", labelKey: "lab.bucket.success_flip" },
  { key: "stage", labelKey: "lab.bucket.stage" },
];

export const PROMPT_STAGES: Array<{ key: PromptStageKey; labelKey: string }> = [
  { key: "s1_grounding", labelKey: "lab.stage.s1_grounding" },
  { key: "s2_macro", labelKey: "lab.stage.s2_macro" },
  { key: "s3_discover", labelKey: "lab.stage.s3_discover" },
  { key: "s4_verify", labelKey: "lab.stage.s4_verify" },
  { key: "s5_success", labelKey: "lab.stage.s5_success" },
];

export const LABEL_CATEGORIES: LabelCategory[] = [
  "idle",
  "failure_recovery",
  "inefficiency",
  "failure",
  "necessary_skill",
  "terminal",
  "unknown",
];

export interface TriageDraft {
  mode: "merge" | "promote";
  targetLabelId: string;
  rationale: string;
  newId: string;
  newName: string;
  newCategory: LabelCategory;
  newRole: TrainingRole;
  newPolicy: SplicePolicy;
  newUiTitle: string;
  newUiHint: string;
  newDefinition: string;
}

export interface StageLike {
  stage_id?: string;
  t_start?: number;
  t_end?: number;
  label?: string;
  label_en?: string;
}

export function safeJsonStringify(val: unknown): string {
  if (typeof val === "string") return val;
  try {
    return JSON.stringify(val ?? {}, null, 2);
  } catch {
    return "{}";
  }
}

export function parseJsonObject(text: string, invalidObjMessage: string): Record<string, unknown> {
  const trimmed = text.trim();
  if (!trimmed) return {};
  const parsed = JSON.parse(trimmed);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed as Record<string, unknown>;
  }
  throw new Error(invalidObjMessage);
}

export function fmtPct(v: number | undefined | null): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function fmtNum(v: number | undefined | null, digits = 3): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtDelta(
  v: number | undefined | null,
  opts: { pct?: boolean; lowerBetter?: boolean; digits?: number } = {},
): { text: string; cls: string } {
  if (v === undefined || v === null || Number.isNaN(v)) {
    return { text: "—", cls: "delta-neutral" };
  }
  const sign = v > 0 ? "+" : "";
  const text = opts.pct
    ? `${sign}${(v * 100).toFixed(1)}%`
    : `${sign}${v.toFixed(opts.digits ?? 3)}`;
  if (Math.abs(v) < 1e-6) return { text, cls: "delta-neutral" };
  const improved = opts.lowerBetter ? v < 0 : v > 0;
  return { text, cls: improved ? "delta-pos" : "delta-neg" };
}

export function toJsonDict(
  val: unknown,
  fallback: Record<string, unknown>,
): Record<string, unknown> {
  if (val && typeof val === "object" && !Array.isArray(val)) {
    return val as Record<string, unknown>;
  }
  if (Array.isArray(val)) {
    return { items: val };
  }
  if (val !== null && val !== undefined) {
    return { value: val };
  }
  return fallback;
}

export function fmtCompactDiagValue(val: unknown, t: TranslateFn): string {
  if (val === null || val === undefined) return "null";
  if (typeof val === "string") return val;
  if (typeof val === "number" || typeof val === "boolean") return String(val);
  if (typeof val === "object" && !Array.isArray(val)) {
    const rec = val as Record<string, unknown>;
    if (typeof rec.label_id === "string") {
      const policy = typeof rec.splice_policy === "string" ? ` [${rec.splice_policy}]` : "";
      const span =
        typeof rec.t_start === "number" && typeof rec.t_end === "number"
          ? ` (${rec.t_start.toFixed(1)}–${rec.t_end.toFixed(1)}s)`
          : "";
      return `${rec.label_id}${policy}${span}`;
    }
    if (typeof rec.verdict === "string" || typeof rec.passed === "boolean") {
      const verdict = typeof rec.verdict === "string" ? rec.verdict : "";
      const passedTxt =
        rec.passed === true
          ? t("lab.diag.verdictPass")
          : rec.passed === false
            ? t("lab.diag.verdictFail")
            : rec.passed === null
              ? t("lab.diag.verdictUnsure")
              : "";
      if (verdict && passedTxt) return `${verdict} (${passedTxt})`;
      return verdict || passedTxt || "—";
    }
  }
  try {
    const raw = JSON.stringify(val);
    return raw.length > 140 ? `${raw.slice(0, 137)}...` : raw;
  } catch {
    return String(val);
  }
}

export function summarizeStageDiff(vlmVal: unknown, gtVal: unknown, t: TranslateFn): string[] {
  const vlmArr: StageLike[] = Array.isArray(vlmVal) ? (vlmVal as StageLike[]) : [];
  const gtArr: StageLike[] = Array.isArray(gtVal) ? (gtVal as StageLike[]) : [];
  if (vlmArr.length === 0 && gtArr.length === 0) {
    return [t("lab.diag.stageStructDiffers")];
  }

  const lines: string[] = [];
  const maxLen = Math.max(vlmArr.length, gtArr.length);
  for (let i = 0; i < maxLen; i += 1) {
    const v = vlmArr[i];
    const g = gtArr[i];
    if (v && g) {
      const id = g.stage_id || v.stage_id || `S${i + 1}`;
      const vStart = typeof v.t_start === "number" ? v.t_start : null;
      const vEnd = typeof v.t_end === "number" ? v.t_end : null;
      const gStart = typeof g.t_start === "number" ? g.t_start : null;
      const gEnd = typeof g.t_end === "number" ? g.t_end : null;
      const boundaryChanged =
        vStart !== null &&
        vEnd !== null &&
        gStart !== null &&
        gEnd !== null &&
        (Math.abs(vStart - gStart) > 0.05 || Math.abs(vEnd - gEnd) > 0.05);
      const vName = v.label || v.label_en || "";
      const gName = g.label || g.label_en || "";
      const nameChanged = Boolean(vName || gName) && vName !== gName;

      if (boundaryChanged || nameChanged) {
        const parts: string[] = [];
        if (
          boundaryChanged &&
          vStart !== null &&
          vEnd !== null &&
          gStart !== null &&
          gEnd !== null
        ) {
          parts.push(
            t("lab.diag.stageBoundaryShift", {
              id,
              vStart: vStart.toFixed(1),
              vEnd: vEnd.toFixed(1),
              gStart: gStart.toFixed(1),
              gEnd: gEnd.toFixed(1),
            }),
          );
        } else {
          parts.push(`${id}`);
        }
        if (nameChanged) {
          parts.push(
            t("lab.diag.stageRenamed", {
              vName: vName || "—",
              gName: gName || "—",
            }),
          );
        }
        lines.push(parts.join(" · "));
      }
    } else if (v && !g) {
      const id = v.stage_id || `S${i + 1}`;
      const span =
        typeof v.t_start === "number" && typeof v.t_end === "number"
          ? ` (${v.t_start.toFixed(1)}–${v.t_end.toFixed(1)}s)`
          : "";
      lines.push(t("lab.diag.stageExtraVlm", { id, span }));
    } else if (!v && g) {
      const id = g.stage_id || `S${i + 1}`;
      const span =
        typeof g.t_start === "number" && typeof g.t_end === "number"
          ? ` (${g.t_start.toFixed(1)}–${g.t_end.toFixed(1)}s)`
          : "";
      lines.push(t("lab.diag.stageAddedGt", { id, span }));
    }
  }

  return lines.length > 0
    ? lines
    : [t("lab.diag.stageCountDiff", { vlmCount: vlmArr.length, gtCount: gtArr.length })];
}

export function buildDiagnosisReviewLink(err: DiagnosisErrorItem): string {
  const qs = new URLSearchParams();
  let tVal: number | null =
    typeof err.t_start === "number" && Number.isFinite(err.t_start) ? err.t_start : null;

  if (tVal === null && err.error_type === "stage") {
    const vlmArr: StageLike[] = Array.isArray(err.vlm_value) ? (err.vlm_value as StageLike[]) : [];
    const gtArr: StageLike[] = Array.isArray(err.gt_value) ? (err.gt_value as StageLike[]) : [];
    const maxLen = Math.max(vlmArr.length, gtArr.length);
    for (let i = 0; i < maxLen; i += 1) {
      const v = vlmArr[i];
      const g = gtArr[i];
      const changed =
        !v ||
        !g ||
        v.t_start !== g.t_start ||
        v.t_end !== g.t_end ||
        v.label !== g.label ||
        v.label_en !== g.label_en;
      if (changed) {
        const cand = g?.t_start ?? v?.t_start;
        if (typeof cand === "number" && Number.isFinite(cand)) {
          tVal = cand;
          break;
        }
      }
    }
  }

  if (tVal !== null) {
    qs.set("t", String(Number(tVal.toFixed(2))));
  }
  const segId = err.gt_seg_id || err.vlm_seg_id;
  if (segId) {
    qs.set("seg", segId);
  }
  qs.set("tab", "diff");
  return `/review/${encodeEpPath(err.episode_id)}?${qs.toString()}`;
}

export function fmtCacheHits(
  stages: Record<string, number> | undefined | null,
  rate: number | undefined | null,
): string {
  const entries = Object.entries(stages ?? {}).filter(
    ([, count]) => typeof count === "number" && count > 0,
  );
  const totalHits = entries.reduce((sum, [, count]) => sum + count, 0);
  const breakdown = entries
    .map(([stage, count]) => `${stage.replace(/_.*$/, "")}:${count}`)
    .join(" ");
  if (breakdown) {
    return `${totalHits} (${breakdown}, ${fmtPct(rate)})`;
  }
  return `${totalHits} (${fmtPct(rate)})`;
}
