import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { TaskSpec } from "../api/types";
import { I18nProvider } from "../stores/i18nStore";
import { TaskDraftWizardModal } from "./TaskDraftWizardModal";

function makeDraftedSpec(overrides: Partial<TaskSpec> = {}): TaskSpec {
  return {
    task_id: "fold_towel",
    version: "1.0.0",
    match: ["*fold_towel*", "*towel*"],
    aliases: ["towel_folding"],
    title: "Fold Towel",
    description: "Bimanually fold a flat towel in half and align its edges.",
    goal: "Fold the towel neatly in half on the tabletop.",
    criteria: [
      "Towel is folded cleanly in half",
      "Both hands release the towel without dragging",
    ],
    steps: ["Grasp opposite corners", "Fold towel across centerline", "Smooth and release"],
    task_prompt: "Evaluate bimanual towel folding quality and smoothness.",
    grounding_objects: [
      {
        id: "towel",
        name: "Blue Towel",
        desc: "Soft rectangular fabric towel on tabletop",
        tag_pos: "top",
        is_static: false,
      },
    ],
    hand_mapping: null,
    camera_roles_override: null,
    ...overrides,
  };
}

describe("TaskDraftWizardModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("validates task_id slug (disallows _default), generates AI draft with sample episode, allows editing, and saves via PUT /api/lab/tasks", async () => {
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];
    const drafted = makeDraftedSpec();

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });

        if (url.endsWith("/api/lab/tasks/draft") && method === "POST") {
          return new Response(
            JSON.stringify({
              advisory_only: true,
              task_spec: drafted,
              used_sample_episode_id: "ds/ep_unrec_1",
              sampled_frames_count: 5,
              rationale: "Identified towel and workspace surface from 5 sampled frames.",
              warnings: ["Verify towel color if dataset contains multiple colors."],
            }),
            { status: 200 },
          );
        }
        if (url.endsWith("/api/lab/tasks") && method === "PUT") {
          const spec = body as TaskSpec;
          return new Response(
            JSON.stringify({ tasks: { [spec.task_id]: spec }, total: 1 }),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
      }),
    );

    const onClose = vi.fn();
    const onTaskSaved = vi.fn();

    render(
      <I18nProvider initialLocale="en">
        <TaskDraftWizardModal
          open={true}
          onClose={onClose}
          onTaskSaved={onTaskSaved}
          initialSampleEpisodeId="ds/ep_unrec_1"
          sampleEpisodes={[
            { episode_id: "ds/ep_unrec_1", task_id: "_default", dataset: "ds" },
            { episode_id: "ds/ep_other", task_id: "pick_and_place", dataset: "ds" },
          ]}
        />
      </I18nProvider>,
    );

    // Sample episode is pre-filled from initialSampleEpisodeId
    const sampleSelect = screen.getByTestId("wizard-sample-episode-select") as HTMLSelectElement;
    expect(sampleSelect.value).toBe("ds/ep_unrec_1");

    // Reject _default task_id without calling backend
    fireEvent.change(screen.getByTestId("wizard-task-id-input"), { target: { value: "_default" } });
    fireEvent.change(screen.getByTestId("wizard-description-input"), {
      target: { value: "\u6298\u53e0\u6bdb\u5dfe\u5e76\u653e\u6574\u9f50" },
    });
    fireEvent.click(screen.getByTestId("wizard-generate-draft-btn"));
    expect(screen.getByTestId("wizard-error-banner")).toHaveTextContent("_default");
    expect(calls.some((c) => c.url.endsWith("/api/lab/tasks/draft"))).toBe(false);

    // Reject invalid slug with spaces
    fireEvent.change(screen.getByTestId("wizard-task-id-input"), { target: { value: "bad slug!" } });
    fireEvent.click(screen.getByTestId("wizard-generate-draft-btn"));
    expect(screen.getByTestId("wizard-error-banner")).toBeInTheDocument();

    // Valid slug + Chinese description -> POST /api/lab/tasks/draft
    fireEvent.change(screen.getByTestId("wizard-task-id-input"), { target: { value: "fold_towel" } });
    fireEvent.change(screen.getByTestId("wizard-title-input"), { target: { value: "Fold Towel" } });
    fireEvent.click(screen.getByTestId("wizard-generate-draft-btn"));

    await waitFor(() => expect(screen.getByTestId("wizard-draft-meta")).toBeInTheDocument());
    expect(screen.getByTestId("wizard-advisory-badge")).toHaveTextContent("Advisory Draft");
    expect(screen.getByTestId("wizard-sampled-frames-badge")).toHaveTextContent(
      "Sampled 5 frames from ds/ep_unrec_1",
    );
    expect(screen.getByTestId("wizard-draft-rationale")).toHaveTextContent(
      "Identified towel and workspace surface",
    );
    expect(screen.getByTestId("wizard-draft-warnings")).toHaveTextContent(
      "Verify towel color",
    );

    const draftCall = calls.find((c) => c.url.endsWith("/api/lab/tasks/draft"));
    expect(draftCall?.body).toEqual({
      task_id: "fold_towel",
      description: "\u6298\u53e0\u6bdb\u5dfe\u5e76\u653e\u6574\u9f50",
      title: "Fold Towel",
      sample_episode_id: "ds/ep_unrec_1",
      num_frames: 5,
    });

    // Review step edits: update goal, add a grounding object, edit match/aliases
    fireEvent.change(screen.getByTestId("wizard-field-goal"), {
      target: { value: "Fold the blue towel neatly in half and center it." },
    });
    fireEvent.click(screen.getByTestId("wizard-add-object-btn"));
    fireEvent.change(screen.getByTestId("wizard-obj-id-1"), { target: { value: "tray" } });
    fireEvent.change(screen.getByTestId("wizard-obj-name-1"), { target: { value: "Target Tray" } });
    fireEvent.change(screen.getByTestId("wizard-obj-desc-1"), {
      target: { value: "Flat plastic workspace tray" },
    });
    fireEvent.change(screen.getByTestId("wizard-obj-tag-1"), { target: { value: "bottom" } });
    fireEvent.click(screen.getByTestId("wizard-obj-static-1"));
    fireEvent.change(screen.getByTestId("wizard-field-match"), {
      target: { value: "*fold_towel*, *linen*" },
    });
    fireEvent.change(screen.getByTestId("wizard-field-aliases"), {
      target: { value: "towel_fold, fold_cloth" },
    });

    // Save via PUT /api/lab/tasks
    fireEvent.click(screen.getByTestId("wizard-save-task-btn"));

    await waitFor(() => expect(onTaskSaved).toHaveBeenCalledTimes(1));
    expect(onClose).toHaveBeenCalledTimes(1);

    const putCall = calls.find((c) => c.url.endsWith("/api/lab/tasks") && c.method === "PUT");
    expect(putCall?.body).toMatchObject({
      task_id: "fold_towel",
      goal: "Fold the blue towel neatly in half and center it.",
      match: ["*fold_towel*", "*linen*"],
      aliases: ["towel_fold", "fold_cloth"],
      grounding_objects: [
        {
          id: "towel",
          name: "Blue Towel",
          tag_pos: "top",
          is_static: false,
        },
        {
          id: "tray",
          name: "Target Tray",
          desc: "Flat plastic workspace tray",
          tag_pos: "bottom",
          is_static: true,
        },
      ],
    });
  });

  it("supports Start Blank Draft without calling AI draft endpoint and renders Chinese labels in zh locale", async () => {
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? JSON.parse(String(init.body)) : undefined;
        calls.push({ url, method, body });
        if (url.endsWith("/api/lab/tasks") && method === "PUT") {
          const spec = body as TaskSpec;
          return new Response(JSON.stringify({ tasks: { [spec.task_id]: spec }, total: 1 }), {
            status: 200,
          });
        }
        return new Response(JSON.stringify({ items: [], total: 0 }), { status: 200 });
      }),
    );

    const onTaskSaved = vi.fn();
    const onClose = vi.fn();

    render(
      <I18nProvider initialLocale="zh">
        <TaskDraftWizardModal
          open={true}
          onClose={onClose}
          onTaskSaved={onTaskSaved}
          sampleEpisodes={[]}
        />
      </I18nProvider>,
    );

    expect(screen.getByTestId("wizard-generate-draft-btn")).toHaveTextContent(
      "AI \u751f\u6210\u8349\u7a3f",
    );
    expect(screen.getByTestId("wizard-blank-draft-btn")).toHaveTextContent(
      "\u4ece\u7a7a\u767d\u8349\u7a3f\u5f00\u59cb",
    );

    fireEvent.change(screen.getByTestId("wizard-task-id-input"), {
      target: { value: "pour_water" },
    });
    fireEvent.click(screen.getByTestId("wizard-blank-draft-btn"));

    await waitFor(() => expect(screen.getByTestId("wizard-save-task-btn")).toBeInTheDocument());
    expect(calls.some((c) => c.url.endsWith("/api/lab/tasks/draft"))).toBe(false);

    fireEvent.change(screen.getByTestId("wizard-field-goal"), {
      target: { value: "Pour water from pitcher into cup" },
    });
    fireEvent.click(screen.getByTestId("wizard-save-task-btn"));

    await waitFor(() => expect(onTaskSaved).toHaveBeenCalledTimes(1));
    expect(
      calls.some(
        (c) =>
          c.url.endsWith("/api/lab/tasks") &&
          c.method === "PUT" &&
          (c.body as TaskSpec)?.task_id === "pour_water",
      ),
    ).toBe(true);
  });
});
