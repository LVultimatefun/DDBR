import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_VISUAL_LABELS } from "../constants/labels";
import type { DiagnosisErrorItem } from "../api/types";
import { AuthProvider } from "../stores/authStore";
import { I18nProvider } from "../stores/i18nStore";
import { VlmLabPage } from "./VlmLabPage";

function buildMockFixtures() {
  const labelsDoc = {
    schema_version: "1.0",
    version: 1,
    categories: [
      "idle",
      "failure_recovery",
      "inefficiency",
      "failure",
      "necessary_skill",
      "terminal",
      "unknown",
    ],
    labels: DEFAULT_VISUAL_LABELS,
  };

  const promptsDoc = {
    s1_grounding: {
      stage: "s1_grounding",
      version: "1.0.0",
      description: "2D Grounding",
      system_template: "S1 system",
      user_template: "S1 user",
      extra_templates: {},
    },
    s2_macro: {
      stage: "s2_macro",
      version: "1.0.0",
      description: "Macro backbone",
      system_template: "S2 system",
      user_template: "S2 user",
      extra_templates: {},
    },
    s3_discover: {
      stage: "s3_discover",
      version: "1.0.0",
      description: "QC discovery",
      system_template: "S3 system",
      user_template: "S3 user",
      extra_templates: {},
    },
    s4_verify: {
      stage: "s4_verify",
      version: "1.0.0",
      description: "Local verify",
      system_template: "S4 system",
      user_template: "S4 user",
      extra_templates: {},
    },
    s5_success: {
      stage: "s5_success",
      version: "1.0.0",
      description: "Success check",
      system_template: "S5 system",
      user_template: "S5 user",
      extra_templates: {},
    },
  };

  const tasksDoc = {
    tasks: {
      pick_and_place: {
        task_id: "pick_and_place",
        version: "1.0.0",
        match: ["*pick*"],
        aliases: [],
        title: "Pick and Place",
        description: "Bimanual pick-and-place task",
        goal: "Pick the target block and place it stably in the bin",
        criteria: ["Block rests stably inside the target bin"],
        steps: ["Approach", "Grasp", "Place"],
        task_prompt: "Objectively describe the pick-and-place manipulation",
        grounding_objects: [],
        hand_mapping: null,
        camera_roles_override: null,
      },
    },
    total: 1,
  };

  const diagnosisDoc = {
    episodes_evaluated: 3,
    total_errors: 4,
    counts_by_type: {
      fp_overflag: 2,
      fn_missed: 1,
      policy_flip: 1,
      boundary: 0,
      success_flip: 0,
      stage: 0,
    },
    counts_by_label: { hesitation_oscillation: 2 },
    counts_by_task: { pick_and_place: 4 },
    errors: [
      {
        error_id: "demo_lab/ep_001::fp::seg_vlm_1",
        episode_id: "demo_lab/ep_001",
        task_id: "pick_and_place",
        error_type: "fp_overflag",
        label_id: "hesitation_oscillation",
        t_start: 1.0,
        t_end: 2.5,
        vlm_seg_id: "seg_vlm_1",
        gt_seg_id: null,
        vlm_value: {
          label_id: "hesitation_oscillation",
          splice_policy: "mask_loss",
          t_start: 1.0,
          t_end: 2.5,
        },
        gt_value: null,
        iou: null,
        severity: 1.5,
        summary: "Monotonic fine adjustment misclassified as oscillation",
      },
    ],
    top_errors: [
      {
        error_id: "demo_lab/ep_001::fp::seg_vlm_1",
        episode_id: "demo_lab/ep_001",
        task_id: "pick_and_place",
        error_type: "fp_overflag",
        label_id: "hesitation_oscillation",
        t_start: 1.0,
        t_end: 2.5,
        vlm_seg_id: "seg_vlm_1",
        gt_seg_id: null,
        vlm_value: {
          label_id: "hesitation_oscillation",
          splice_policy: "mask_loss",
          t_start: 1.0,
          t_end: 2.5,
        },
        gt_value: null,
        iou: null,
        severity: 1.5,
        summary: "Monotonic fine adjustment misclassified as oscillation",
      },
    ],
    clusters: [
      {
        cluster_key: "fp_overflag:hesitation_oscillation:pick_and_place",
        error_type: "fp_overflag",
        label_id: "hesitation_oscillation",
        task_id: "pick_and_place",
        count: 2,
        episode_ids: ["demo_lab/ep_001"],
        sample_summary: "Monotonic fine adjustment misclassified as oscillation",
      },
    ],
  };

  const unknownsDoc = {
    items: [
      {
        episode_id: "demo_lab/ep_002",
        dataset: "demo_lab",
        task_id: "pick_and_place",
        status: "vlm_ready",
        seg_id: "s_unk_1",
        t_start: 0.5,
        t_end: 1.2,
        label_id: "unknown_anomaly",
        affected_arm: "right",
        unknown_note: "Brief micro-jitter on approach",
        suggested_label: null,
        visual_evidence: {
          initial_state: "Gripper approaches handle",
          process_change: "Slight jitter then continues",
          end_state: "Stable grasp restored",
          confidence_note: "unknown",
        },
        confidence: 0.55,
      },
      {
        episode_id: "demo_lab/ep_002",
        dataset: "demo_lab",
        task_id: "pick_and_place",
        status: "vlm_ready",
        seg_id: "s_unk_2",
        t_start: 2.0,
        t_end: 2.8,
        label_id: "unknown_anomaly",
        affected_arm: "left",
        unknown_note: "Cable snagged on table corner",
        suggested_label: "cable_snag",
        visual_evidence: {
          initial_state: "Left arm moving",
          process_change: "Cable catches corner",
          end_state: "Arm stalls under tension",
          confidence_note: "new anomaly",
        },
        confidence: 0.72,
      },
    ],
    total: 2,
  };

  const fewShotItems = [
    {
      example_id: "fs_1",
      scope: "task:pick_and_place",
      task_id: "pick_and_place",
      label_id: "precision_fine_adjust",
      error_type: "fp_overflag",
      episode_id: "demo_lab/ep_001",
      t_start: 1.0,
      t_end: 2.5,
      tile_image_path: "artifacts/few_shots/fs_1.jpg",
      vlm_wrong_json: { label_id: "hesitation_oscillation" },
      human_right_json: { label_id: "precision_fine_adjust" },
      rationale: "Steady monotonic fine alignment is not hesitation oscillation",
      enabled: true,
      created_at: 1700000100,
    },
  ];

  const evalBaseline = {
    eval_id: "eval_base_001",
    created_at: 1700000100,
    dataset: "demo_lab",
    task_id: null,
    few_shot: false,
    metrics: {
      episodes_evaluated: 3,
      f1: 0.65,
      precision: 0.6,
      recall: 0.71,
      mean_iou: 0.58,
      overcut_rate: 0.25,
      guardrail_downgrade_rate: 0.15,
      success_accuracy: 0.67,
      total_tokens: 12000,
      cache_hit_stages: {},
      cache_hit_rate: 0.0,
    },
    per_episode: [
      {
        episode_id: "demo_lab/ep_001",
        task_id: "pick_and_place",
        sandbox_run_id: "run_sbx_base_1",
        gt_rev_id: "rev_1",
        precision: 0.6,
        recall: 0.71,
        f1: 0.65,
        mean_iou: 0.58,
        overcut_rate: 0.25,
        guardrail_downgrade_rate: 0.15,
        success_correct: false,
        pred_segments: 2,
        gt_segments: 2,
        matched_segments: 1,
        tokens_total: 12000,
        cache_hit_stages: [],
      },
    ],
    config_snapshot: { config_sha256: "abc", label_version: 1, prompt_bundle_sha256: "def" },
    candidate_overrides: { prompts: {}, labels: {} },
  };

  const evalCandidate = {
    eval_id: "eval_cand_002",
    created_at: 1700000200,
    dataset: "demo_lab",
    task_id: null,
    few_shot: true,
    metrics: {
      episodes_evaluated: 3,
      f1: 0.88,
      precision: 0.9,
      recall: 0.86,
      mean_iou: 0.81,
      overcut_rate: 0.05,
      guardrail_downgrade_rate: 0.05,
      success_accuracy: 1.0,
      total_tokens: 4200,
      cache_hit_stages: { s1_grounding: 1, s2_macro: 1, s5_success: 1 },
      cache_hit_rate: 0.6,
    },
    per_episode: [
      {
        episode_id: "demo_lab/ep_001",
        task_id: "pick_and_place",
        sandbox_run_id: "run_sbx_cand_1",
        gt_rev_id: "rev_1",
        precision: 0.9,
        recall: 0.86,
        f1: 0.88,
        mean_iou: 0.81,
        overcut_rate: 0.05,
        guardrail_downgrade_rate: 0.05,
        success_correct: true,
        pred_segments: 1,
        gt_segments: 1,
        matched_segments: 1,
        tokens_total: 4200,
        cache_hit_stages: ["s1_grounding", "s2_macro", "s5_success"],
      },
    ],
    config_snapshot: { config_sha256: "abc2", label_version: 1, prompt_bundle_sha256: "def2" },
    candidate_overrides: { prompts: {}, labels: {} },
  };

  return {
    labelsDoc,
    promptsDoc,
    tasksDoc,
    diagnosisDoc,
    unknownsDoc,
    fewShotItems,
    evalBaseline,
    evalCandidate,
  };
}

describe("VlmLabPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.removeItem("robo_vlm_qc.locale");
  });

  it("blocks Reviewer role with 403 guard without calling lab endpoints", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    expect(screen.getByTestId("lab-forbidden")).toBeInTheDocument();
    expect(screen.getByText(/403 — Forbidden/)).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("supports diagnosis 6-bucket KPI, quick Few-Shot, and unknown triage (merge & promote)", async () => {
    const fx = buildMockFixtures();
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];

    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = init?.method ?? "GET";
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body });

      if (url.startsWith("/api/lab/labels") && method === "GET") {
        return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
      }
      if (url.startsWith("/api/lab/prompts") && method === "GET") {
        return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
      }
      if (url.startsWith("/api/lab/tasks") && method === "GET") {
        return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
      }
      if (url.startsWith("/api/lab/diagnosis")) {
        return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
      }
      if (url.startsWith("/api/lab/unknowns") && method === "GET") {
        return new Response(JSON.stringify(fx.unknownsDoc), { status: 200 });
      }
      if (url.endsWith("/api/lab/few-shots") && method === "POST") {
        return new Response(
          JSON.stringify({ ...fx.fewShotItems[0], example_id: "fs_quick_99" }),
          { status: 201 },
        );
      }
      if (url.endsWith("/api/lab/unknowns/triage") && method === "POST") {
        if (body.action === "merge") {
          return new Response(
            JSON.stringify({
              ok: true,
              action: "merge",
              episode_id: body.episode_id,
              seg_id: body.seg_id,
              target_label_id: body.target_label_id,
              few_shot_id: "fs_merged_1",
              label_version: 1,
            }),
            { status: 200 },
          );
        }
        return new Response(
          JSON.stringify({
            ok: true,
            action: "promote",
            episode_id: body.episode_id,
            seg_id: body.seg_id,
            label_id: body.new_label.id,
            label_version: 2,
          }),
          { status: 200 },
        );
      }
      return new Response(JSON.stringify({}), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    // 6-bucket KPI & Top-N error row
    await waitFor(() =>
      expect(screen.getByTestId("kpi-bucket-fp_overflag")).toHaveTextContent("2"),
    );
    expect(screen.getByTestId("kpi-bucket-fn_missed")).toHaveTextContent("1");
    expect(screen.getByTestId("kpi-bucket-policy_flip")).toHaveTextContent("1");
    const diagTable = screen.getByTestId("diagnosis-errors-table");
    expect(diagTable).toHaveTextContent("demo_lab/ep_001");
    expect(diagTable).toHaveTextContent("Monotonic fine adjustment misclassified as oscillation");
    expect(diagTable).toHaveTextContent("hesitation_oscillation");
    expect(diagTable).toHaveTextContent("VLM:");
    expect(diagTable).toHaveTextContent("→ GT: null");

    // Quick add Few-Shot from diagnosis row
    fireEvent.click(screen.getByTestId("quick-fewshot-btn-0"));
    expect(screen.getByTestId("quick-fewshot-form")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("submit-quick-fewshot-btn"));

    await waitFor(() =>
      expect(screen.getByTestId("lab-info-banner")).toHaveTextContent("fs_quick_99"),
    );
    const fsPost = calls.find((c) => c.url.endsWith("/api/lab/few-shots") && c.method === "POST");
    expect(fsPost?.body).toMatchObject({
      episode_id: "demo_lab/ep_001",
      error_type: "fp_overflag",
      capture_tile: true,
      vlm_wrong_json: {
        label_id: "hesitation_oscillation",
        splice_policy: "mask_loss",
        t_start: 1.0,
        t_end: 2.5,
      },
      human_right_json: {
        corrected: "human_gt",
      },
    });

    // Unknown triage: merge s_unk_1
    await waitFor(() => expect(screen.getByTestId("unknown-item-s_unk_1")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("triage-merge-btn-s_unk_1"));
    await waitFor(() =>
      expect(screen.getByTestId("lab-info-banner")).toHaveTextContent("fs_merged_1"),
    );
    expect(
      calls.some(
        (c) =>
          c.url.endsWith("/api/lab/unknowns/triage") &&
          (c.body as Record<string, unknown>)?.action === "merge",
      ),
    ).toBe(true);

    // Unknown triage: promote s_unk_2
    fireEvent.click(screen.getByTestId("triage-mode-promote-s_unk_2"));
    fireEvent.change(screen.getByLabelText("New label ID-s_unk_2"), {
      target: { value: "cable_snag" },
    });
    fireEvent.change(screen.getByLabelText("New label name-s_unk_2"), {
      target: { value: "Cable Snag" },
    });
    fireEvent.click(screen.getByTestId("triage-promote-btn-s_unk_2"));

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/unknowns/triage") &&
            (c.body as Record<string, unknown>)?.action === "promote",
        ),
      ).toBe(true),
    );
  });

  it("manages Few-Shot list, 2x2 tile URL with token, toggle enabled, and manual creation", async () => {
    const fx = buildMockFixtures();
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });

        if (url.startsWith("/api/lab/labels")) {
          return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts")) {
          return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/tasks")) {
          return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.startsWith("/api/lab/few-shots/fs_1") && method === "PATCH") {
          return new Response(
            JSON.stringify({ ...fx.fewShotItems[0], enabled: body.enabled }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/few-shots") && method === "GET") {
          return new Response(
            JSON.stringify({ items: fx.fewShotItems, total: fx.fewShotItems.length }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/lab/few-shots") && method === "POST") {
          const created = {
            ...fx.fewShotItems[0],
            example_id: "fs_manual_2",
            episode_id: body.episode_id,
            rationale: body.rationale,
          };
          fx.fewShotItems.push(created);
          return new Response(JSON.stringify(created), { status: 201 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-few-shots"));
    await waitFor(() => expect(screen.getByTestId("fewshot-card-fs_1")).toBeInTheDocument());

    const img = screen.getByAltText("2x2 tile fs_1") as HTMLImageElement;
    expect(img.getAttribute("src")).toBe("/api/lab/few-shots/fs_1/tile?token=lead-token");

    // Toggle enabled
    const toggle = screen.getByTestId("toggle-fewshot-fs_1") as HTMLInputElement;
    expect(toggle.checked).toBe(true);
    fireEvent.click(toggle);
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/few-shots/fs_1") &&
            c.method === "PATCH" &&
            (c.body as Record<string, unknown>)?.enabled === false,
        ),
      ).toBe(true),
    );

    // Manual creation
    fireEvent.click(screen.getByTestId("open-manual-fewshot-btn"));
    const form = screen.getByTestId("manual-fewshot-form");
    const inputs = form.querySelectorAll("input, textarea");
    fireEvent.change(inputs[0], { target: { value: "demo_lab/ep_009" } });
    fireEvent.change(inputs[inputs.length - 1], {
      target: { value: "Manually recorded visual criterion" },
    });
    fireEvent.submit(form);

    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/few-shots") &&
            c.method === "POST" &&
            (c.body as Record<string, unknown>)?.episode_id === "demo_lab/ep_009",
        ),
      ).toBe(true),
    );
  });

  it("edits and saves Labels, Prompts (+ advisory prompt-assist), and TaskSpecs", async () => {
    const fx = buildMockFixtures();
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });

        if (url.startsWith("/api/lab/labels") && method === "GET") {
          return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/labels") && method === "PUT") {
          return new Response(JSON.stringify(body), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts") && method === "GET") {
          return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts") && method === "PUT") {
          return new Response(JSON.stringify(body), { status: 200 });
        }
        if (url.startsWith("/api/lab/tasks") && method === "GET") {
          return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/tasks") && method === "PUT") {
          return new Response(
            JSON.stringify({ tasks: { [body.task_id]: body }, total: 1 }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/lab/prompt-assist") && method === "POST") {
          return new Response(
            JSON.stringify({
              stage: "s3_discover",
              base_version: "1.0.0",
              advisory_only: true,
              suggested_prompt_diff: "+ Distinguish steady monotonic alignment from oscillation",
              proposed_system_template_patch: "S3 system updated draft",
              proposed_user_template_patch: "S3 user updated draft",
              rationale: "Exempt slow monotonic precision adjustment from fp_overflag",
              task_spec_notes: "Keep consistent with Precision-First guidelines",
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    await waitFor(() => expect(screen.getByTestId("labels-editor")).toBeInTheDocument());

    // Save Labels
    fireEvent.click(screen.getByText("Bump Version (+1)"));
    fireEvent.click(screen.getByTestId("save-labels-btn"));
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/labels") &&
            c.method === "PUT" &&
            (c.body as Record<string, unknown>)?.version === 2,
        ),
      ).toBe(true),
    );

    // Prompts + advisory prompt-assist (must NOT auto-save)
    fireEvent.click(screen.getByTestId("taxonomy-subtab-prompts"));
    await waitFor(() => expect(screen.getByTestId("prompts-editor")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("prompt-assist-btn"));

    await waitFor(() => {
      const preview = screen.getByTestId("prompt-assist-preview");
      expect(preview).toHaveTextContent("Distinguish steady monotonic alignment from oscillation");
      expect(preview).toHaveTextContent("v1.0.0");
      expect(preview).not.toHaveTextContent("vundefined");
    });
    expect(calls.some((c) => c.url.endsWith("/api/lab/prompts") && c.method === "PUT")).toBe(
      false,
    );

    fireEvent.click(screen.getByTestId("apply-assist-draft-btn"));
    fireEvent.click(screen.getByTestId("save-prompts-btn"));
    await waitFor(() =>
      expect(calls.some((c) => c.url.endsWith("/api/lab/prompts") && c.method === "PUT")).toBe(
        true,
      ),
    );
    const promptsPut = calls.find((c) => c.url.endsWith("/api/lab/prompts") && c.method === "PUT");
    expect((promptsPut?.body as Record<string, Record<string, unknown>>)?.s3_discover).toMatchObject({
      system_template: "S3 system updated draft",
      user_template: "S3 user updated draft",
    });

    // TaskSpecs
    fireEvent.click(screen.getByTestId("taxonomy-subtab-tasks"));
    await waitFor(() => expect(screen.getByTestId("tasks-editor")).toBeInTheDocument());
    fireEvent.click(screen.getByTestId("save-task-btn"));
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/tasks") &&
            c.method === "PUT" &&
            (c.body as Record<string, unknown>)?.task_id === "pick_and_place",
        ),
      ).toBe(true),
    );
  });

  it("runs GT sandbox eval, compares baseline vs candidate metrics, and promotes config", async () => {
    const fx = buildMockFixtures();
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });

        if (url.startsWith("/api/lab/labels")) {
          return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts")) {
          return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/tasks")) {
          return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        if (url.endsWith("/api/jobs") && method === "POST") {
          return new Response(
            JSON.stringify({
              job_id: "job_eval_1",
              job_type: "eval_gt",
              status: "running",
              progress_done: 1,
              progress_total: 3,
            }),
            { status: 201 },
          );
        }
        if (url.endsWith("/api/jobs/job_eval_1") && method === "GET") {
          return new Response(
            JSON.stringify({
              job_id: "job_eval_1",
              job_type: "eval_gt",
              status: "completed",
              progress_done: 3,
              progress_total: 3,
              summary: {
                eval_run_id: "eval_cand_002",
                eval_report: fx.evalCandidate,
              },
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/eval/compare")) {
          return new Response(
            JSON.stringify({
              evals: [fx.evalCandidate, fx.evalBaseline],
              baseline: fx.evalBaseline,
              candidate: fx.evalCandidate,
              deltas: {
                delta_f1: 0.23,
                delta_precision: 0.3,
                delta_recall: 0.15,
                delta_mean_iou: 0.23,
                delta_overcut_rate: -0.2,
                delta_guardrail_downgrade_rate: -0.1,
                delta_success_accuracy: 0.33,
                delta_total_tokens: -7800,
                delta_cache_hit_rate: 0.6,
              },
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/lab/promote") && method === "POST") {
          return new Response(
            JSON.stringify({
              ok: true,
              eval_id: body.eval_id,
              promoted_files: ["configs_override/prompts/s3_discover.yaml"],
              label_version: 1,
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-eval"));
    await waitFor(() => expect(screen.getByTestId("eval-compare-table")).toBeInTheDocument());

    fireEvent.click(screen.getByTestId("run-eval-btn"));
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/jobs") &&
            c.method === "POST" &&
            (c.body as Record<string, unknown>)?.job_type === "eval_gt" &&
            ((c.body as Record<string, Record<string, unknown>>)?.eval_params?.few_shot === true),
        ),
      ).toBe(true),
    );

    await waitFor(() =>
      expect(screen.getByTestId("lab-info-banner")).toHaveTextContent("3 GT episodes"),
    );
    expect(screen.getByTestId("lab-info-banner")).not.toHaveTextContent("undefined");

    const table = screen.getByTestId("eval-compare-table");
    expect(table).toHaveTextContent("0.880");
    expect(table).toHaveTextContent("+0.230");
    expect(table).toHaveTextContent("-20.0%");
    expect(table).toHaveTextContent("-10.0%");
    expect(table).toHaveTextContent("-7800");
    expect(table).toHaveTextContent("+60.0%");
    expect(table).toHaveTextContent("3 (s1:1 s2:1 s5:1, 60.0%)");
    expect(table).not.toHaveTextContent("[object Object]");

    // Per-episode success_correct rendered accurately
    const evalPanel = screen.getByTestId("lab-panel-eval");
    expect(evalPanel).toHaveTextContent("Match");

    fireEvent.click(screen.getByTestId("promote-btn"));
    await waitFor(() =>
      expect(
        calls.some(
          (c) =>
            c.url.endsWith("/api/lab/promote") &&
            c.method === "POST" &&
            (c.body as Record<string, unknown>)?.eval_id === "eval_cand_002",
        ),
      ).toBe(true),
    );
    await waitFor(() =>
      expect(screen.getByTestId("lab-info-banner")).toHaveTextContent("Promoted config to active workspace"),
    );
  });

  it("renders compact stage diffs with collapsible raw JSON and valid deep-links without null params", async () => {
    const fx = buildMockFixtures();
    const extraErrors: DiagnosisErrorItem[] = [
      {
        error_id: "demo_lab/ep_001::stage::macro",
        episode_id: "demo_lab/ep_001",
        task_id: "pick_and_place",
        error_type: "stage",
        label_id: null,
        t_start: null,
        t_end: null,
        vlm_seg_id: null,
        gt_seg_id: null,
        vlm_value: [
          { stage_id: "S5", t_start: 101.0, t_end: 129.0, label: "Old Place" },
        ],
        gt_value: [
          { stage_id: "S5", t_start: 101.0, t_end: 130.5, label: "Place in Bin" },
        ],
        iou: null,
        severity: 1.5,
        summary: "Macro stage boundary and label mismatch",
      },
      {
        error_id: "demo_lab/ep_002::success_flip",
        episode_id: "demo_lab/ep_002",
        task_id: "pick_and_place",
        error_type: "success_flip",
        label_id: null,
        t_start: null,
        t_end: null,
        vlm_seg_id: null,
        gt_seg_id: null,
        vlm_value: { passed: true, verdict: "success" },
        gt_value: { passed: false, verdict: "failure" },
        iou: null,
        severity: 2.0,
        summary: "VLM false positive on success verdict",
      },
    ];
    const diagPayload = {
      ...fx.diagnosisDoc,
      top_errors: [...(fx.diagnosisDoc.top_errors as DiagnosisErrorItem[]), ...extraErrors],
    };

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.startsWith("/api/lab/labels")) return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        if (url.startsWith("/api/lab/prompts")) return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        if (url.startsWith("/api/lab/tasks")) return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        if (url.startsWith("/api/lab/diagnosis")) return new Response(JSON.stringify(diagPayload), { status: 200 });
        if (url.startsWith("/api/lab/unknowns")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    const table = await screen.findByTestId("diagnosis-errors-table");
    await waitFor(() => expect(table).toHaveTextContent("S5 boundary shift: 101.0–129.0s → 101.0–130.5s"));
    expect(table).toHaveTextContent("renamed (Old Place → Place in Bin)");
    expect(table).toHaveTextContent("hesitation_oscillation [mask_loss] (1.0–2.5s)");
    expect(table).toHaveTextContent("success (pass) → GT: failure (fail)");
    expect(table).toHaveTextContent("View raw JSON");

    const links = Array.from(table.querySelectorAll("a")) as HTMLAnchorElement[];
    expect(links.length).toBe(3);
    expect(links[0].getAttribute("href")).toBe("/review/demo_lab/ep_001?t=1&seg=seg_vlm_1&tab=diff");
    expect(links[1].getAttribute("href")).toBe("/review/demo_lab/ep_001?t=101&tab=diff");
    expect(links[2].getAttribute("href")).toBe("/review/demo_lab/ep_002?tab=diff");
    for (const lnk of links) {
      expect(lnk.getAttribute("href")).not.toContain("null");
    }
  });

  it("opens .history/ drawer, renders red/green diff lines, and rolls back snapshot with config reload", async () => {
    const fx = buildMockFixtures();
    let labelLoadCount = 0;
    let promptLoadCount = 0;
    let taskLoadCount = 0;
    const rollbackCalls: string[] = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";

        if (url.startsWith("/api/lab/labels")) {
          labelLoadCount += 1;
          return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts")) {
          promptLoadCount += 1;
          return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/tasks")) {
          taskLoadCount += 1;
          return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/diagnosis")) return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        if (url.startsWith("/api/lab/unknowns")) return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });

        if (url.startsWith("/api/lab/history/snap_1/diff")) {
          return new Response(
            JSON.stringify({
              snapshot_id: "snap_1",
              relative_path: "configs_override/qc_labels.yaml",
              created_at: 1700000100,
              reason: "manual_save",
              unified_diff: "--- a\n+++ b\n-version: 1\n+version: 2",
              lines: [
                { type: "hunk", old_line: null, new_line: null, text: "@@ -1,2 +1,2 @@" },
                { type: "remove", old_line: 1, new_line: null, text: "version: 1" },
                { type: "add", old_line: null, new_line: 1, text: "version: 2" },
              ],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/lab/history/snap_1/rollback") && method === "POST") {
          rollbackCalls.push("snap_1");
          return new Response(
            JSON.stringify({
              ok: true,
              relative_path: "configs_override/qc_labels.yaml",
              config_sha256: "deadbeef12345678",
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/history")) {
          return new Response(
            JSON.stringify({
              items: [
                {
                  snapshot_id: "snap_1",
                  target: "labels",
                  name: "qc_labels.yaml",
                  relative_path: "configs_override/qc_labels.yaml",
                  created_at: 1700000100,
                  reason: "manual_save",
                  size_bytes: 512,
                },
              ],
            }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    fireEvent.click(await screen.findByTestId("open-history-drawer-btn"));
    expect(screen.getByTestId("lab-history-drawer")).toBeInTheDocument();

    fireEvent.click(await screen.findByTestId("history-snapshot-snap_1"));
    const diffViewer = await screen.findByTestId("history-diff-viewer");
    expect(diffViewer).toHaveTextContent("-version: 1");
    expect(diffViewer).toHaveTextContent("+version: 2");
    expect(diffViewer.querySelector(".diff-line-remove")).not.toBeNull();
    expect(diffViewer.querySelector(".diff-line-add")).not.toBeNull();

    const prevLoads = labelLoadCount;
    fireEvent.click(screen.getByTestId("history-rollback-btn"));

    await waitFor(() => expect(rollbackCalls).toEqual(["snap_1"]));
    await waitFor(() => expect(labelLoadCount).toBeGreaterThan(prevLoads));
    expect(promptLoadCount).toBeGreaterThanOrEqual(2);
    expect(taskLoadCount).toBeGreaterThanOrEqual(2);
    expect(screen.getByTestId("lab-info-banner")).toHaveTextContent("Safely rolled back");
  });

  it("renders Chinese headings, tabs, KPIs, and history drawer cleanly when locale is zh", async () => {
    const fx = buildMockFixtures();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : input.toString();
        if (url.startsWith("/api/lab/labels")) return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        if (url.startsWith("/api/lab/prompts")) return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        if (url.startsWith("/api/lab/tasks")) return new Response(JSON.stringify(fx.tasksDoc), { status: 200 });
        if (url.startsWith("/api/lab/diagnosis")) return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        if (url.startsWith("/api/lab/unknowns")) return new Response(JSON.stringify(fx.unknownsDoc), { status: 200 });
        if (url.startsWith("/api/lab/few-shots")) {
          return new Response(
            JSON.stringify({ items: fx.fewShotItems, total: fx.fewShotItems.length }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/eval/compare")) {
          return new Response(
            JSON.stringify({
              evals: [fx.evalCandidate, fx.evalBaseline],
              baseline: fx.evalBaseline,
              candidate: fx.evalCandidate,
              deltas: null,
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/history")) {
          return new Response(JSON.stringify({ items: [] }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    const { unmount } = render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "rev-token", username: "rev_1", role: "reviewer" }}>
          <MemoryRouter>
            <VlmLabPage />
          </MemoryRouter>
        </AuthProvider>
      </I18nProvider>,
    );

    expect(screen.getByTestId("lab-forbidden")).toHaveTextContent("403 — 无访问权限");
    expect(screen.getByTestId("lab-forbidden")).toHaveTextContent("VLM 实验室仅限组长（Lead）使用");
    unmount();

    render(
      <I18nProvider initialLocale="zh">
        <AuthProvider initialAuth={{ token: "lead-token", username: "lead_1", role: "lead" }}>
          <MemoryRouter>
            <VlmLabPage />
          </MemoryRouter>
        </AuthProvider>
      </I18nProvider>,
    );

    // Main heading & tabs
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("VLM 实验室闭环迭代工作台（组长）");
    expect(screen.getByTestId("tab-diagnosis")).toHaveTextContent("误差诊断与未知分流");
    expect(screen.getByTestId("tab-few-shots")).toHaveTextContent("Few-Shot 错题本");
    expect(screen.getByTestId("tab-taxonomy")).toHaveTextContent("标签 / 提示词 / 任务规范");
    expect(screen.getByTestId("tab-eval")).toHaveTextContent("GT 评测沙箱");

    // Tab 1: Diagnosis & Unknown Triage
    await waitFor(() =>
      expect(screen.getByTestId("diagnosis-kpi-strip")).toHaveTextContent("已评测 GT 片段"),
    );
    expect(screen.getByTestId("kpi-bucket-fp_overflag")).toHaveTextContent("误报过检");
    expect(screen.getByTestId("unknown-triage-panel")).toHaveTextContent("未知异常分流池");

    // Tab 2: Few-Shot Mistake Book
    fireEvent.click(screen.getByTestId("tab-few-shots"));
    await waitFor(() =>
      expect(screen.getByTestId("lab-panel-few-shots")).toHaveTextContent("视觉 / 文本 Few-Shot 错题本"),
    );
    expect(screen.getByTestId("open-manual-fewshot-btn")).toHaveTextContent("新增 Few-Shot 案例");

    // Tab 3: Taxonomy + History Drawer
    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    expect(screen.getByTestId("taxonomy-subtab-labels")).toHaveTextContent("视觉标签 (labels.yaml)");
    expect(screen.getByTestId("taxonomy-subtab-prompts")).toHaveTextContent("各阶段提示词 (S1–S5)");
    expect(screen.getByTestId("taxonomy-subtab-tasks")).toHaveTextContent("任务规范 (TaskSpec)");

    fireEvent.click(screen.getByTestId("open-history-drawer-btn"));
    const drawer = await screen.findByTestId("lab-history-drawer");
    expect(drawer).toHaveTextContent(".history/ 配置快照与回滚");
    expect(drawer).toHaveTextContent("历史归档快照 (0)");
    fireEvent.click(screen.getByTestId("close-history-drawer-btn"));

    // Tab 4: GT Eval Sandbox
    fireEvent.click(screen.getByTestId("tab-eval"));
    await waitFor(() =>
      expect(screen.getByTestId("eval-compare-panel")).toHaveTextContent("版本指标对比与一键发布"),
    );
    expect(screen.getByTestId("eval-runner-controls")).toHaveTextContent("GT 回归评测沙箱");
    expect(screen.getByTestId("promote-btn")).toHaveTextContent("一键发布为生效配置");
  });

  it("launches TaskDraftWizardModal from Tasks tab with workspace episodes, drafts via AI, saves TaskSpec, and refreshes tasks", async () => {
    const fx = buildMockFixtures();
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];
    const tasksState: Record<string, Record<string, unknown>> = { ...fx.tasksDoc.tasks };

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });

        if (url.startsWith("/api/lab/labels")) {
          return new Response(JSON.stringify(fx.labelsDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/prompts")) {
          return new Response(JSON.stringify(fx.promptsDoc), { status: 200 });
        }
        if (url.endsWith("/api/lab/tasks/draft") && method === "POST") {
          return new Response(
            JSON.stringify({
              advisory_only: true,
              task_spec: {
                task_id: body.task_id,
                version: "1.0.0",
                match: ["*wipe_table*"],
                aliases: [],
                title: "Wipe Table",
                description: "Wipe tabletop clean with sponge",
                goal: "Remove spills from tabletop",
                criteria: ["Tabletop is visibly clean"],
                steps: ["Grasp sponge", "Wipe surface", "Set sponge down"],
                task_prompt: "Evaluate tabletop wiping motion.",
                grounding_objects: [
                  {
                    id: "sponge",
                    name: "Yellow Sponge",
                    desc: "Cleaning sponge",
                    tag_pos: "top",
                    is_static: false,
                  },
                ],
                hand_mapping: null,
                camera_roles_override: null,
              },
              used_sample_episode_id: body.sample_episode_id ?? null,
              sampled_frames_count: body.sample_episode_id ? 5 : 0,
              rationale: "Drafted wiping TaskSpec from sample episode.",
              warnings: [],
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/tasks") && method === "PUT") {
          const spec = body as Record<string, unknown>;
          const tid = String(spec.task_id);
          tasksState[tid] = spec;
          return new Response(
            JSON.stringify({ tasks: tasksState, total: Object.keys(tasksState).length }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/tasks") && method === "GET") {
          return new Response(
            JSON.stringify({ tasks: tasksState, total: Object.keys(tasksState).length }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/episodes")) {
          return new Response(
            JSON.stringify({
              items: [{ episode_id: "demo_ds/ep_wipe_1", dataset: "demo_ds", task_id: "_default" }],
              total: 1,
            }),
            { status: 200 },
          );
        }
        if (url.startsWith("/api/lab/diagnosis")) {
          return new Response(JSON.stringify(fx.diagnosisDoc), { status: 200 });
        }
        if (url.startsWith("/api/lab/unknowns")) {
          return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    render(
      <AuthProvider initialAuth={{ token: "lead-tok", username: "lead_1", role: "lead" }}>
        <MemoryRouter>
          <VlmLabPage />
        </MemoryRouter>
      </AuthProvider>,
    );

    fireEvent.click(screen.getByTestId("tab-taxonomy"));
    fireEvent.click(await screen.findByTestId("taxonomy-subtab-tasks"));
    fireEvent.click(await screen.findByTestId("lab-new-task-ai-btn"));

    expect(await screen.findByTestId("task-draft-wizard-modal")).toBeInTheDocument();

    await waitFor(() =>
      expect(
        Array.from(
          (screen.getByTestId("wizard-sample-episode-select") as HTMLSelectElement).options,
        ).some((o) => o.value === "demo_ds/ep_wipe_1"),
      ).toBe(true),
    );

    fireEvent.change(screen.getByTestId("wizard-task-id-input"), {
      target: { value: "wipe_table" },
    });
    fireEvent.change(screen.getByTestId("wizard-description-input"), {
      target: { value: "Use sponge to wipe table clean" },
    });
    fireEvent.change(screen.getByTestId("wizard-sample-episode-select"), {
      target: { value: "demo_ds/ep_wipe_1" },
    });
    fireEvent.click(screen.getByTestId("wizard-generate-draft-btn"));

    expect(await screen.findByTestId("wizard-sampled-frames-badge")).toHaveTextContent(
      "Sampled 5 frames from demo_ds/ep_wipe_1",
    );
    fireEvent.click(screen.getByTestId("wizard-save-task-btn"));

    await waitFor(() => expect(screen.queryByTestId("task-draft-wizard-modal")).toBeNull());
    await waitFor(() =>
      expect((screen.getByLabelText("Select TaskSpec") as HTMLSelectElement).value).toBe(
        "wipe_table",
      ),
    );
    expect(
      calls.some((c) => c.url.endsWith("/api/lab/tasks/draft") && c.method === "POST"),
    ).toBe(true);
    expect(calls.some((c) => c.url.endsWith("/api/lab/tasks") && c.method === "PUT")).toBe(true);
  });
});
