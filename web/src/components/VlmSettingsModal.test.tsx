import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { VlmSettingsResponse } from "../api/types";
import { I18nProvider } from "../stores/i18nStore";
import { VlmSettingsModal } from "./VlmSettingsModal";

function makeVlmSettings(overrides: Partial<VlmSettingsResponse> = {}): VlmSettingsResponse {
  return {
    provider: "openai_compat",
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env: "DASHSCOPE_API_KEY",
    api_key_configured: true,
    api_key_masked: "sk-***abcd",
    thinking_mode: "dashscope",
    stage_models: {
      s1_grounding: "qwen-vl-max",
      s2_macro: "qwen-vl-max",
      s3_discover: "qwen-vl-max",
      s4_verify: "qwen-vl-max",
      s5_success: "qwen-vl-max",
    },
    presets: [
      {
        id: "openai",
        label: "OpenAI",
        provider: "openai_compat",
        base_url: "https://api.openai.com/v1",
        api_key_env: "OPENAI_API_KEY",
        thinking_mode: "disabled",
        default_model: "gpt-4o",
        stage_models: {
          s1_grounding: "gpt-4o",
          s2_macro: "gpt-4o",
          s3_discover: "gpt-4o",
          s4_verify: "gpt-4o",
          s5_success: "gpt-4o",
        },
      },
      {
        id: "dashscope",
        label: "DashScope (Qwen-VL)",
        provider: "openai_compat",
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: "DASHSCOPE_API_KEY",
        thinking_mode: "dashscope",
        default_model: "qwen-vl-max",
        stage_models: {
          s1_grounding: "qwen-vl-max",
          s2_macro: "qwen-vl-max",
          s3_discover: "qwen-vl-max",
          s4_verify: "qwen-vl-max",
          s5_success: "qwen-vl-max",
        },
      },
      {
        id: "gemini",
        label: "Gemini",
        provider: "gemini",
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        api_key_env: "GEMINI_API_KEY",
        thinking_mode: "auto",
        default_model: "gemini-2.5-pro",
        stage_models: {
          s1_grounding: "gemini-2.5-pro",
          s2_macro: "gemini-2.5-pro",
          s3_discover: "gemini-2.5-pro",
          s4_verify: "gemini-2.5-pro",
          s5_success: "gemini-2.5-pro",
        },
      },
      {
        id: "vllm",
        label: "Local vLLM / SGLang",
        provider: "openai_compat",
        base_url: "http://127.0.0.1:8001/v1",
        api_key_env: "VLLM_API_KEY",
        thinking_mode: "disabled",
        default_model: "Qwen/Qwen2.5-VL-72B-Instruct",
        stage_models: {
          s1_grounding: "Qwen/Qwen2.5-VL-72B-Instruct",
          s2_macro: "Qwen/Qwen2.5-VL-72B-Instruct",
          s3_discover: "Qwen/Qwen2.5-VL-72B-Instruct",
          s4_verify: "Qwen/Qwen2.5-VL-72B-Instruct",
          s5_success: "Qwen/Qwen2.5-VL-72B-Instruct",
        },
      },
      {
        id: "custom",
        label: "Custom",
        provider: "openai_compat",
        base_url: "",
        api_key_env: "VLM_API_KEY",
        thinking_mode: "auto",
        stage_models: {},
      },
    ],
    ...overrides,
  };
}

describe("VlmSettingsModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("loads masked config, applies presets, tests connection with live latency, and saves without leaking blank key", async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    const initial = makeVlmSettings();

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : undefined;
        calls.push({ url, method, body });

        if (url === "/api/settings/vlm" && method === "GET") {
          return new Response(JSON.stringify(initial), { status: 200 });
        }
        if (url === "/api/settings/vlm/test" && method === "POST") {
          return new Response(
            JSON.stringify({
              ok: true,
              latency_ms: 142.4,
              model: String(body?.model ?? "gemini-2.5-pro"),
              detail: "Probe succeeded",
            }),
            { status: 200 },
          );
        }
        if (url === "/api/settings/vlm" && method === "PUT") {
          return new Response(
            JSON.stringify(
              makeVlmSettings({
                provider: "gemini",
                base_url: String(body?.base_url ?? initial.base_url),
                api_key_env: String(body?.api_key_env ?? "GEMINI_API_KEY"),
                thinking_mode: "auto",
                stage_models: (body?.stage_models as Record<string, string>) ?? initial.stage_models,
              }),
            ),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    const onClose = vi.fn();
    const onSaved = vi.fn();
    render(<VlmSettingsModal open onClose={onClose} onSaved={onSaved} />);

    expect(await screen.findByTestId("vlm-api-key-badge")).toHaveTextContent(
      "Configured (sk-***abcd)",
    );
    const keyInput = screen.getByTestId("vlm-api-key-input") as HTMLInputElement;
    expect(keyInput.value).toBe("");
    expect(keyInput.type).toBe("password");

    // All 5 preset cards render
    for (const id of ["openai", "dashscope", "gemini", "vllm", "custom"]) {
      expect(screen.getByTestId(`vlm-preset-${id}`)).toBeInTheDocument();
    }

    // One-click switch to Gemini preset
    fireEvent.click(screen.getByTestId("vlm-preset-gemini"));
    expect((screen.getByTestId("vlm-provider-select") as HTMLSelectElement).value).toBe("gemini");
    expect((screen.getByTestId("vlm-base-url-input") as HTMLInputElement).value).toBe(
      "https://generativelanguage.googleapis.com/v1beta",
    );
    expect((screen.getByTestId("vlm-global-model-input") as HTMLInputElement).value).toBe(
      "gemini-2.5-pro",
    );

    // Test connection with blank api_key (omits api_key so backend uses saved key)
    fireEvent.click(screen.getByTestId("vlm-test-btn"));
    const testBanner = await screen.findByTestId("vlm-test-result");
    expect(testBanner).toHaveTextContent("Connection OK");
    expect(testBanner).toHaveTextContent("142 ms");
    expect(testBanner).toHaveTextContent("Probe succeeded");

    const testCall = calls.find((c) => c.url === "/api/settings/vlm/test");
    expect(testCall?.body).toEqual({
      provider: "gemini",
      base_url: "https://generativelanguage.googleapis.com/v1beta",
      api_key_env: "GEMINI_API_KEY",
      thinking_mode: "auto",
      model: "gemini-2.5-pro",
    });
    expect("api_key" in (testCall?.body ?? {})).toBe(false);

    // Save config without typing a new api_key -> omits api_key from PUT payload
    fireEvent.click(screen.getByTestId("vlm-save-btn"));
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
    expect(onClose).toHaveBeenCalledTimes(1);

    const putCall = calls.find((c) => c.url === "/api/settings/vlm" && c.method === "PUT");
    expect(putCall?.body).toMatchObject({
      provider: "gemini",
      base_url: "https://generativelanguage.googleapis.com/v1beta",
      api_key_env: "GEMINI_API_KEY",
      thinking_mode: "auto",
      default_model: "gemini-2.5-pro",
      stage_models: {
        s1_grounding: "gemini-2.5-pro",
        s2_macro: "gemini-2.5-pro",
        s3_discover: "gemini-2.5-pro",
        s4_verify: "gemini-2.5-pro",
        s5_success: "gemini-2.5-pro",
      },
    });
    expect("api_key" in (putCall?.body ?? {})).toBe(false);
  });

  it("supports per-stage S1-S5 routing overrides, new API key submission, and displays failed test details", async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = typeof input === "string" ? input : input.toString();
        const method = init?.method ?? "GET";
        const body = init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : undefined;
        calls.push({ url, method, body });

        if (url === "/api/settings/vlm" && method === "GET") {
          return new Response(
            JSON.stringify(makeVlmSettings({ api_key_configured: false, api_key_masked: "" })),
            { status: 200 },
          );
        }
        if (url === "/api/settings/vlm/test" && method === "POST") {
          return new Response(
            JSON.stringify({
              ok: false,
              latency_ms: 85,
              model: "gpt-4o",
              detail: "401 Unauthorized",
            }),
            { status: 200 },
          );
        }
        if (url === "/api/settings/vlm" && method === "PUT") {
          return new Response(
            JSON.stringify(
              makeVlmSettings({
                api_key_configured: true,
                api_key_masked: "sk-***9999",
                stage_models: (body?.stage_models as Record<string, string>) ?? {},
              }),
            ),
            { status: 200 },
          );
        }
        return new Response(JSON.stringify({}), { status: 200 });
      }),
    );

    const onClose = vi.fn();
    render(<VlmSettingsModal open onClose={onClose} />);

    expect(await screen.findByTestId("vlm-api-key-badge")).toHaveTextContent("Not Configured");

    // Click OpenAI preset, type API key, expand per-stage routing and customize S5
    fireEvent.click(screen.getByTestId("vlm-preset-openai"));
    fireEvent.change(screen.getByTestId("vlm-api-key-input"), {
      target: { value: "sk-secret-key-9999" },
    });

    fireEvent.click(screen.getByTestId("vlm-toggle-stage-routing-btn"));
    expect(screen.getByTestId("vlm-stage-routing-panel")).toBeInTheDocument();
    fireEvent.change(screen.getByTestId("vlm-stage-model-s5_success"), {
      target: { value: "gpt-4o-mini" },
    });

    // Test connection surfaces error detail cleanly
    fireEvent.click(screen.getByTestId("vlm-test-btn"));
    const failBanner = await screen.findByTestId("vlm-test-result");
    expect(failBanner).toHaveTextContent("Connection Failed");
    expect(failBanner).toHaveTextContent("401 Unauthorized");

    const testCall = calls.find((c) => c.url === "/api/settings/vlm/test");
    expect(testCall?.body?.api_key).toBe("sk-secret-key-9999");

    // Save sends heterogeneous stage_models and api_key
    fireEvent.click(screen.getByTestId("vlm-save-btn"));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));

    const putCall = calls.find((c) => c.url === "/api/settings/vlm" && c.method === "PUT");
    expect(putCall?.body).toMatchObject({
      provider: "openai_compat",
      api_key: "sk-secret-key-9999",
      stage_models: {
        s1_grounding: "gpt-4o",
        s2_macro: "gpt-4o",
        s3_discover: "gpt-4o",
        s4_verify: "gpt-4o",
        s5_success: "gpt-4o-mini",
      },
    });
    expect("default_model" in (putCall?.body ?? {})).toBe(false);
  });

  it("renders Chinese VLM modal labels and badges when locale is zh", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify(makeVlmSettings()), { status: 200 })),
    );

    render(
      <I18nProvider initialLocale="zh">
        <VlmSettingsModal open onClose={vi.fn()} />
      </I18nProvider>,
    );

    expect(await screen.findByText("VLM 模型与 API 配置")).toBeInTheDocument();
    expect(screen.getByTestId("vlm-api-key-badge")).toHaveTextContent("已配置 (sk-***abcd)");
    expect(screen.getByTestId("vlm-preset-vllm")).toHaveTextContent("本地 vLLM / SGLang");
    expect(screen.getByTestId("vlm-test-btn")).toHaveTextContent("测试连通性");
    expect(screen.getByTestId("vlm-save-btn")).toHaveTextContent("保存配置");
  });
});
