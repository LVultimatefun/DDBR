import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { commonEn, commonZh } from "../i18n/common";
import { I18nProvider, useI18n } from "./i18nStore";

const Probe: React.FC = () => {
  const { locale, setLocale, toggleLocale, t } = useI18n();
  return (
    <div>
      <span data-testid="probe-locale">{locale}</span>
      <span data-testid="probe-title">{t("nav.workQueue")}</span>
      <span data-testid="probe-interp">{t("queue.selectedCount", { count: 7 })}</span>
      <span data-testid="probe-missing">{t("nonexistent.key.xyz")}</span>
      <button type="button" data-testid="probe-toggle" onClick={toggleLocale}>
        toggle
      </button>
      <button type="button" data-testid="probe-set-zh" onClick={() => setLocale("zh")}>
        zh
      </button>
    </div>
  );
};

describe("i18nStore", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.lang = "";
  });

  it("defaults to en, toggles/persists zh to localStorage and document.documentElement.lang, and interpolates vars", () => {
    render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    );

    expect(screen.getByTestId("probe-locale")).toHaveTextContent("en");
    expect(screen.getByTestId("probe-title")).toHaveTextContent("Work Queue");
    expect(screen.getByTestId("probe-interp")).toHaveTextContent("Selected: 7");
    expect(screen.getByTestId("probe-missing")).toHaveTextContent("nonexistent.key.xyz");
    expect(document.documentElement.lang).toBe("en");
    expect(window.localStorage.getItem("robo_vlm_qc.locale")).toBe("en");

    fireEvent.click(screen.getByTestId("probe-toggle"));
    expect(screen.getByTestId("probe-locale")).toHaveTextContent("zh");
    expect(screen.getByTestId("probe-title")).toHaveTextContent("工作队列");
    expect(screen.getByTestId("probe-interp")).toHaveTextContent("已选：7");
    expect(document.documentElement.lang).toBe("zh-CN");
    expect(window.localStorage.getItem("robo_vlm_qc.locale")).toBe("zh");
  });

  it("hydrates locale from localStorage and respects initialLocale override", () => {
    window.localStorage.setItem("robo_vlm_qc.locale", "zh");
    const { unmount } = render(
      <I18nProvider>
        <Probe />
      </I18nProvider>,
    );
    expect(screen.getByTestId("probe-locale")).toHaveTextContent("zh");
    expect(screen.getByTestId("probe-title")).toHaveTextContent("工作队列");
    unmount();

    render(
      <I18nProvider initialLocale="en">
        <Probe />
      </I18nProvider>,
    );
    expect(screen.getByTestId("probe-locale")).toHaveTextContent("en");
    expect(screen.getByTestId("probe-title")).toHaveTextContent("Work Queue");
  });

  it("returns safe default English fallback when useI18n is called outside I18nProvider", () => {
    window.localStorage.setItem("robo_vlm_qc.locale", "zh");
    render(<Probe />);
    expect(screen.getByTestId("probe-locale")).toHaveTextContent("en");
    expect(screen.getByTestId("probe-title")).toHaveTextContent("Work Queue");
    expect(screen.getByTestId("probe-interp")).toHaveTextContent("Selected: 7");
  });

  it("has complete non-empty Chinese translations for every commonEn key", () => {
    const keys = Object.keys(commonEn) as Array<keyof typeof commonEn>;
    expect(keys.length).toBeGreaterThan(50);
    for (const k of keys) {
      expect(typeof commonZh[k]).toBe("string");
      expect(commonZh[k].trim().length).toBeGreaterThan(0);
    }
  });
});
