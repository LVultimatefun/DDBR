import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  Eye,
  EyeOff,
  X,
  Zap,
} from "lucide-react";
import { getVlmSettings, testVlmSettings, updateVlmSettings } from "../api/client";
import type {
  PromptStageKey,
  VlmPresetItem,
  VlmProvider,
  VlmSettingsResponse,
  VlmSettingsUpdatePayload,
  VlmTestPayload,
  VlmTestResponse,
  VlmThinkingMode,
} from "../api/types";
import { useI18n } from "../stores/i18nStore";
import { useToast } from "../stores/toastStore";

export interface VlmSettingsModalProps {
  open: boolean;
  onClose: () => void;
  onSaved?: (settings: VlmSettingsResponse) => void;
}

const STAGE_KEYS: Array<{ key: PromptStageKey; labelKey: string }> = [
  { key: "s1_grounding", labelKey: "vlmSettings.stage.s1_grounding" },
  { key: "s2_macro", labelKey: "vlmSettings.stage.s2_macro" },
  { key: "s3_discover", labelKey: "vlmSettings.stage.s3_discover" },
  { key: "s4_verify", labelKey: "vlmSettings.stage.s4_verify" },
  { key: "s5_success", labelKey: "vlmSettings.stage.s5_success" },
];

const KNOWN_PRESET_IDS = new Set(["openai", "dashscope", "gemini", "vllm", "custom"]);

const FALLBACK_PRESETS: VlmPresetItem[] = [
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
    default_model: "",
    stage_models: {},
  },
];

function emptyStageModels(fill = ""): Record<PromptStageKey, string> {
  return {
    s1_grounding: fill,
    s2_macro: fill,
    s3_discover: fill,
    s4_verify: fill,
    s5_success: fill,
  };
}

function normalizeStageModels(
  raw: Record<string, string> | undefined,
  fallback = "",
): Record<PromptStageKey, string> {
  const firstNonEmpty =
    fallback ||
    STAGE_KEYS.map((s) => (raw?.[s.key] ?? "").trim()).find(Boolean) ||
    "";
  return {
    s1_grounding: (raw?.s1_grounding ?? "").trim() || firstNonEmpty,
    s2_macro: (raw?.s2_macro ?? "").trim() || firstNonEmpty,
    s3_discover: (raw?.s3_discover ?? "").trim() || firstNonEmpty,
    s4_verify: (raw?.s4_verify ?? "").trim() || firstNonEmpty,
    s5_success: (raw?.s5_success ?? "").trim() || firstNonEmpty,
  };
}

function detectUniformModel(models: Record<PromptStageKey, string>): {
  uniform: boolean;
  primary: string;
} {
  const vals = STAGE_KEYS.map((s) => (models[s.key] ?? "").trim());
  const nonEmpty = vals.filter(Boolean);
  if (nonEmpty.length === 0) return { uniform: true, primary: "" };
  const first = nonEmpty[0];
  const uniform = vals.every((v) => v === first);
  return { uniform, primary: first };
}

function normalizeUrl(url: string): string {
  return url.trim().replace(/\/+$/, "").toLowerCase();
}

function matchPresetId(
  presets: VlmPresetItem[],
  provider: VlmProvider,
  baseUrl: string,
): string {
  const target = normalizeUrl(baseUrl);
  for (const p of presets) {
    if (p.id === "custom") continue;
    if (p.provider === provider && p.base_url && normalizeUrl(p.base_url) === target) {
      return p.id;
    }
  }
  return "custom";
}

export const VlmSettingsModal: React.FC<VlmSettingsModalProps> = ({
  open,
  onClose,
  onSaved,
}) => {
  const toast = useToast();
  const { locale, t } = useI18n();
  const tRef = useRef(t);
  tRef.current = t;

  const getPresetTitle = (preset: VlmPresetItem): string => {
    if (KNOWN_PRESET_IDS.has(preset.id)) {
      return t(`vlmSettings.preset.${preset.id}.title`);
    }
    return preset.label || preset.name || preset.id;
  };

  const getPresetSubtitle = (preset: VlmPresetItem): string => {
    if (locale === "zh" && KNOWN_PRESET_IDS.has(preset.id)) {
      return t(`vlmSettings.preset.${preset.id}.sub`);
    }
    if (preset.description) return preset.description;
    if (KNOWN_PRESET_IDS.has(preset.id)) {
      return t(`vlmSettings.preset.${preset.id}.sub`);
    }
    return preset.default_model || preset.stage_models?.s1_grounding || preset.provider;
  };

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [presets, setPresets] = useState<VlmPresetItem[]>(FALLBACK_PRESETS);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("custom");
  const [provider, setProvider] = useState<VlmProvider>("openai_compat");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState("OPENAI_API_KEY");
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyConfigured, setApiKeyConfigured] = useState(false);
  const [apiKeyMasked, setApiKeyMasked] = useState("");
  const [thinkingMode, setThinkingMode] = useState<VlmThinkingMode>("auto");
  const [globalModel, setGlobalModel] = useState("");
  const [stageModels, setStageModels] = useState<Record<PromptStageKey, string>>(
    emptyStageModels(),
  );
  const [routingExpanded, setRoutingExpanded] = useState(false);
  const [testResult, setTestResult] = useState<VlmTestResponse | null>(null);

  const hydrateFromResponse = useCallback((res: VlmSettingsResponse) => {
    const resolvedPresets =
      Array.isArray(res.presets) && res.presets.length > 0 ? res.presets : FALLBACK_PRESETS;
    const normalizedStages = normalizeStageModels(res.stage_models);
    const { uniform, primary } = detectUniformModel(normalizedStages);

    setPresets(resolvedPresets);
    setProvider(res.provider);
    setBaseUrl(res.base_url ?? "");
    setApiKeyEnv(res.api_key_env ?? "OPENAI_API_KEY");
    setApiKey("");
    setShowApiKey(false);
    setApiKeyConfigured(Boolean(res.api_key_configured));
    setApiKeyMasked(res.api_key_masked ?? "");
    setThinkingMode(res.thinking_mode ?? "auto");
    setStageModels(normalizedStages);
    setGlobalModel(primary);
    setRoutingExpanded(!uniform);
    setSelectedPresetId(matchPresetId(resolvedPresets, res.provider, res.base_url ?? ""));
  }, []);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    setTestResult(null);
    try {
      const res = await getVlmSettings();
      hydrateFromResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : tRef.current("vlmSettings.errLoad"));
    } finally {
      setLoading(false);
    }
  }, [hydrateFromResponse]);

  useEffect(() => {
    if (!open) return;
    void loadSettings();
  }, [open, loadSettings]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  const isUniform = useMemo(() => detectUniformModel(stageModels).uniform, [stageModels]);

  if (!open) return null;

  const handleSelectPreset = (preset: VlmPresetItem) => {
    setSelectedPresetId(preset.id);
    setError(null);
    setTestResult(null);
    if (preset.id === "custom") {
      if (!apiKeyEnv.trim()) {
        setApiKeyEnv(preset.api_key_env || "VLM_API_KEY");
      }
      return;
    }
    setProvider(preset.provider);
    setBaseUrl(preset.base_url);
    setApiKeyEnv(preset.api_key_env);
    setThinkingMode(preset.thinking_mode);
    const defaultMod =
      (preset.default_model ?? "").trim() ||
      (preset.stage_models?.s1_grounding ?? "").trim() ||
      Object.values(preset.stage_models ?? {}).find((v) => Boolean(v?.trim())) ||
      "";
    const nextStages = normalizeStageModels(preset.stage_models, defaultMod);
    setStageModels(nextStages);
    setGlobalModel(detectUniformModel(nextStages).primary || defaultMod);
  };

  const handleGlobalModelChange = (val: string) => {
    setGlobalModel(val);
    setStageModels(emptyStageModels(val));
  };

  const handleStageModelChange = (key: PromptStageKey, val: string) => {
    setStageModels((prev) => {
      const next = { ...prev, [key]: val };
      const { uniform, primary } = detectUniformModel(next);
      if (uniform) {
        setGlobalModel(primary);
      }
      return next;
    });
  };

  const handleTestConnection = async () => {
    const trimmedUrl = baseUrl.trim();
    if (!trimmedUrl) {
      setError(t("vlmSettings.errBaseUrlBeforeTest"));
      return;
    }
    const probeModel = globalModel.trim() || stageModels.s1_grounding.trim();
    const payload: VlmTestPayload = {
      provider,
      base_url: trimmedUrl,
      api_key_env: apiKeyEnv.trim() || undefined,
      thinking_mode: thinkingMode,
      model: probeModel || undefined,
    };
    if (apiKey.trim()) {
      payload.api_key = apiKey.trim();
    }

    setTesting(true);
    setError(null);
    setTestResult(null);
    try {
      const res = await testVlmSettings(payload);
      setTestResult(res);
    } catch (err) {
      setTestResult({
        ok: false,
        latency_ms: 0,
        model: probeModel || "unknown",
        detail: err instanceof Error ? err.message : t("vlmSettings.errTestFailed"),
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedUrl = baseUrl.trim();
    if (!trimmedUrl) {
      setError(t("vlmSettings.errBaseUrlRequired"));
      return;
    }

    const fallbackModel = globalModel.trim();
    const resolvedStageModels: Record<string, string> = {};
    for (const { key } of STAGE_KEYS) {
      const m = (stageModels[key] ?? "").trim() || fallbackModel;
      if (m) {
        resolvedStageModels[key] = m;
      }
    }
    if (Object.keys(resolvedStageModels).length === 0) {
      setError(t("vlmSettings.errModelRequired"));
      return;
    }

    const payload: VlmSettingsUpdatePayload = {
      provider,
      base_url: trimmedUrl,
      api_key_env: apiKeyEnv.trim() || undefined,
      thinking_mode: thinkingMode,
      stage_models: resolvedStageModels,
    };
    if (isUniform && fallbackModel) {
      payload.default_model = fallbackModel;
    }
    if (apiKey.trim()) {
      payload.api_key = apiKey.trim();
    }

    setSaving(true);
    setError(null);
    try {
      const updated = await updateVlmSettings(payload);
      hydrateFromResponse(updated);
      toast.showToast({
        message: t("vlmSettings.toastSaved", {
          provider: updated.provider,
          model: updated.stage_models?.s1_grounding || fallbackModel,
        }),
        variant: "success",
        testId: "vlm-settings-toast",
      });
      onSaved?.(updated);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : t("vlmSettings.errSaveFailed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="modal-backdrop"
      data-testid="vlm-settings-modal"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <form
        className="modal-card modal-card-wide vlm-modal-card"
        onSubmit={(e) => void handleSave(e)}
      >
        <div className="section-header-row">
          <h3 style={{ margin: 0, display: "inline-flex", alignItems: "center", gap: 6 }}>
            <Cpu size={17} /> {t("vlmSettings.title")}
          </h3>
          <button
            type="button"
            className="mini-btn"
            onClick={onClose}
            aria-label={t("vlmSettings.closeAria")}
            data-testid="close-vlm-settings-btn"
          >
            <X size={14} /> {t("common.close")}
          </button>
        </div>

        <p className="muted small" style={{ margin: 0 }}>
          {t("vlmSettings.desc")}
        </p>

        {loading ? (
          <div className="muted small" style={{ padding: "16px 0" }}>
            {t("vlmSettings.loading")}
          </div>
        ) : (
          <>
            {/* One-click Preset Cards */}
            <div className="field-group">
              <span>{t("vlmSettings.quickPresets")}</span>
              <div className="vlm-preset-grid" data-testid="vlm-preset-grid">
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    className={`vlm-preset-card ${selectedPresetId === preset.id ? "selected" : ""}`}
                    data-testid={`vlm-preset-${preset.id}`}
                    onClick={() => handleSelectPreset(preset)}
                  >
                    <span className="vlm-preset-title">{getPresetTitle(preset)}</span>
                    <span className="vlm-preset-sub">{getPresetSubtitle(preset)}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Provider & Thinking Mode */}
            <div className="two-col">
              <label className="field-group">
                <span>{t("vlmSettings.provider")}</span>
                <select
                  value={provider}
                  data-testid="vlm-provider-select"
                  onChange={(e) => {
                    setProvider(e.target.value as VlmProvider);
                    setSelectedPresetId("custom");
                  }}
                >
                  <option value="openai_compat">{t("vlmSettings.providerOpenaiCompat")}</option>
                  <option value="gemini">{t("vlmSettings.providerGemini")}</option>
                </select>
              </label>

              <label className="field-group">
                <span>{t("vlmSettings.thinkingMode")}</span>
                <select
                  value={thinkingMode}
                  data-testid="vlm-thinking-mode-select"
                  onChange={(e) => setThinkingMode(e.target.value as VlmThinkingMode)}
                >
                  <option value="auto">{t("vlmSettings.thinkingAuto")}</option>
                  <option value="dashscope">{t("vlmSettings.thinkingDashscope")}</option>
                  <option value="disabled">{t("vlmSettings.thinkingDisabled")}</option>
                </select>
              </label>
            </div>

            {/* Base URL */}
            <label className="field-group">
              <span>{t("vlmSettings.baseUrl")}</span>
              <input
                type="text"
                value={baseUrl}
                data-testid="vlm-base-url-input"
                onChange={(e) => {
                  setBaseUrl(e.target.value);
                  setSelectedPresetId(matchPresetId(presets, provider, e.target.value));
                }}
                placeholder={t("vlmSettings.baseUrlPlaceholder")}
              />
            </label>

            {/* API Key + Env Var */}
            <div className="two-col">
              <div className="field-group">
                <div className="vlm-key-header">
                  <span>{t("vlmSettings.apiKey")}</span>
                  <span
                    className={`badge-pill ${apiKeyConfigured ? "ok" : "warn"}`}
                    data-testid="vlm-api-key-badge"
                  >
                    {apiKeyConfigured
                      ? t("vlmSettings.configuredBadge", { masked: apiKeyMasked || "***" })
                      : t("vlmSettings.notConfiguredBadge")}
                  </span>
                </div>
                <div className="input-with-btn">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={apiKey}
                    data-testid="vlm-api-key-input"
                    autoComplete="new-password"
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={
                      apiKeyConfigured
                        ? t("vlmSettings.apiKeyPlaceholderConfigured")
                        : t("vlmSettings.apiKeyPlaceholderEmpty")
                    }
                  />
                  <button
                    type="button"
                    className="secondary-btn compact"
                    onClick={() => setShowApiKey((v) => !v)}
                    aria-label={
                      showApiKey ? t("vlmSettings.hideApiKeyAria") : t("vlmSettings.showApiKeyAria")
                    }
                  >
                    {showApiKey ? <EyeOff size={13} /> : <Eye size={13} />}
                  </button>
                </div>
              </div>

              <label className="field-group">
                <span>{t("vlmSettings.apiKeyEnv")}</span>
                <input
                  type="text"
                  className="mono"
                  value={apiKeyEnv}
                  data-testid="vlm-api-key-env-input"
                  onChange={(e) => setApiKeyEnv(e.target.value)}
                  placeholder="OPENAI_API_KEY"
                />
              </label>
            </div>

            {/* Unified Global Model + Collapsible Per-Stage Routing */}
            <div className="field-group">
              <span>{t("vlmSettings.globalModel")}</span>
              <input
                type="text"
                value={globalModel}
                data-testid="vlm-global-model-input"
                onChange={(e) => handleGlobalModelChange(e.target.value)}
                placeholder={t("vlmSettings.globalModelPlaceholder")}
              />
            </div>

            <div className="vlm-stage-routing-box">
              <button
                type="button"
                className="vlm-stage-routing-toggle"
                data-testid="vlm-toggle-stage-routing-btn"
                aria-expanded={routingExpanded}
                onClick={() => setRoutingExpanded((v) => !v)}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  {routingExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                  {t("vlmSettings.perStageRouting")}
                </span>
                <span className="muted small">
                  {isUniform
                    ? t("vlmSettings.routingUniform")
                    : t("vlmSettings.routingHeterogeneous")}
                </span>
              </button>

              {routingExpanded && (
                <div className="vlm-stage-routing-grid" data-testid="vlm-stage-routing-panel">
                  {STAGE_KEYS.map(({ key, labelKey }) => (
                    <label key={key} className="field-group">
                      <span>{t(labelKey)}</span>
                      <input
                        type="text"
                        value={stageModels[key]}
                        data-testid={`vlm-stage-model-${key}`}
                        onChange={(e) => handleStageModelChange(key, e.target.value)}
                        placeholder={globalModel || t("vlmSettings.stageModelPlaceholder")}
                      />
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Live Connection Test Result */}
            {testResult && (
              <div
                className={`vlm-test-result ${testResult.ok ? "ok" : "fail"}`}
                data-testid="vlm-test-result"
              >
                <div className="vlm-test-result-head">
                  {testResult.ok ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
                  <strong>
                    {testResult.ok ? t("vlmSettings.connectionOk") : t("vlmSettings.connectionFailed")}
                  </strong>
                  <span className="mono small">
                    {testResult.model} · {Math.round(testResult.latency_ms)} ms
                  </span>
                </div>
                {testResult.detail && <div className="small">{testResult.detail}</div>}
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}

            <div className="vlm-modal-footer">
              <button
                type="button"
                className="secondary-btn compact"
                disabled={testing || saving}
                data-testid="vlm-test-btn"
                onClick={() => void handleTestConnection()}
              >
                <Zap size={13} /> {testing ? t("vlmSettings.testing") : t("vlmSettings.testBtn")}
              </button>

              <div className="modal-actions">
                <button type="button" className="secondary-btn compact" onClick={onClose}>
                  {t("common.cancel")}
                </button>
                <button
                  type="submit"
                  className="primary-btn compact"
                  disabled={saving || testing}
                  data-testid="vlm-save-btn"
                >
                  {saving ? t("common.saving") : t("vlmSettings.saveBtn")}
                </button>
              </div>
            </div>
          </>
        )}
      </form>
    </div>
  );
};
