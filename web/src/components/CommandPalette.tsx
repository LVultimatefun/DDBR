import React, { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  Cpu,
  Film,
  KeyRound,
  Languages,
  Layers,
  ListChecks,
  Moon,
  PackageCheck,
  Search,
  Sun,
} from "lucide-react";
import { encodeEpPath, listEpisodes } from "../api/client";
import type { EpisodeListItem } from "../api/types";
import { getEpisodeStatusLabel } from "../constants/labels";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import { useJobs } from "../stores/jobStore";
import { useTheme } from "../stores/themeStore";

export interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  onOpenVlmSettings: () => void;
}

interface PaletteCommandItem {
  id: string;
  group: "navigation" | "actions" | "episodes";
  label: string;
  subtitle?: string;
  keywords: string;
  icon: React.ReactNode;
  run: () => void;
}

function fuzzyScore(query: string, text: string): number {
  const q = query.trim().toLowerCase();
  if (!q) return 1;
  const target = text.toLowerCase();
  const idx = target.indexOf(q);
  if (idx >= 0) {
    return 100 - Math.min(idx, 50) + (q.length / Math.max(target.length, 1)) * 20;
  }
  let qi = 0;
  let streak = 0;
  let score = 0;
  for (let ti = 0; ti < target.length && qi < q.length; ti++) {
    if (target[ti] === q[qi]) {
      qi += 1;
      streak += 1;
      score += 2 + streak;
    } else {
      streak = 0;
    }
  }
  return qi === q.length ? score : 0;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  open,
  onClose,
  onOpenVlmSettings,
}) => {
  const navigate = useNavigate();
  const { isLead, setAuthPromptOpen } = useAuth();
  const { locale, toggleLocale, t } = useI18n();
  const { theme, toggleTheme } = useTheme();
  const { drawerOpen, setDrawerOpen } = useJobs();

  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const [episodes, setEpisodes] = useState<EpisodeListItem[]>([]);
  const [loadingEpisodes, setLoadingEpisodes] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIndex(0);
    window.setTimeout(() => inputRef.current?.focus(), 0);

    let cancelled = false;
    setLoadingEpisodes(true);
    void listEpisodes({ limit: 50 })
      .then((res) => {
        if (!cancelled) setEpisodes(res.items ?? []);
      })
      .catch(() => {
        if (!cancelled) setEpisodes([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingEpisodes(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open]);

  const allItems = useMemo<PaletteCommandItem[]>(() => {
    const items: PaletteCommandItem[] = [
      {
        id: "nav-queue",
        group: "navigation",
        label: t("cmd.goQueue"),
        keywords: "queue work episodes list home /",
        icon: <ListChecks size={14} />,
        run: () => navigate("/"),
      },
    ];

    if (isLead) {
      items.push(
        {
          id: "nav-export",
          group: "navigation",
          label: t("cmd.goExport"),
          keywords: "export studio lerobot h5 mp4 dataset",
          icon: <PackageCheck size={14} />,
          run: () => navigate("/export"),
        },
        {
          id: "nav-lab",
          group: "navigation",
          label: t("cmd.goLab"),
          keywords: "vlm lab prompts labels eval diagnosis",
          icon: <Layers size={14} />,
          run: () => navigate("/lab"),
        },
        {
          id: "act-vlm-settings",
          group: "actions",
          label: t("cmd.openVlmSettings"),
          keywords: "vlm model api key provider cost tier preset settings",
          icon: <Cpu size={14} />,
          run: () => onOpenVlmSettings(),
        },
        {
          id: "act-job-drawer",
          group: "actions",
          label: t("cmd.toggleJobDrawer"),
          keywords: "jobs async drawer queue progress batch",
          icon: <Activity size={14} />,
          run: () => setDrawerOpen(!drawerOpen),
        },
      );
    }

    items.push(
      {
        id: "act-locale",
        group: "actions",
        label: t("cmd.switchLocale"),
        keywords: "language locale english chinese en zh i18n",
        icon: <Languages size={14} />,
        run: () => toggleLocale(),
      },
      {
        id: "act-theme",
        group: "actions",
        label: t("cmd.toggleTheme"),
        keywords: "theme dark light mode appearance color",
        icon: theme === "dark" ? <Sun size={14} /> : <Moon size={14} />,
        run: () => toggleTheme(),
      },
      {
        id: "act-auth",
        group: "actions",
        label: t("cmd.openAuth"),
        keywords: "auth token role lead reviewer credentials login",
        icon: <KeyRound size={14} />,
        run: () => setAuthPromptOpen(true),
      },
    );

    for (const ep of episodes) {
      items.push({
        id: `ep-${ep.episode_id}`,
        group: "episodes",
        label: ep.episode_id,
        subtitle: `${ep.task_id} · ${getEpisodeStatusLabel(ep.status, locale)}`,
        keywords: `${ep.episode_id} ${ep.dataset} ${ep.task_id} ${ep.status} episode review`,
        icon: <Film size={14} />,
        run: () => navigate(`/review/${encodeEpPath(ep.episode_id)}`),
      });
    }

    return items;
  }, [
    isLead,
    t,
    navigate,
    onOpenVlmSettings,
    drawerOpen,
    setDrawerOpen,
    toggleLocale,
    theme,
    toggleTheme,
    setAuthPromptOpen,
    episodes,
    locale,
  ]);

  const filtered = useMemo(() => {
    if (!query.trim()) return allItems;
    return allItems
      .map((item) => ({
        item,
        score: fuzzyScore(query, `${item.label} ${item.subtitle ?? ""} ${item.keywords}`),
      }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score)
      .map((entry) => entry.item);
  }, [allItems, query]);

  useEffect(() => {
    setActiveIndex(0);
  }, [query, filtered.length]);

  const executeItem = (item: PaletteCommandItem) => {
    onClose();
    item.run();
  };

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        onClose();
      } else if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveIndex((prev) => (filtered.length === 0 ? 0 : (prev + 1) % filtered.length));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveIndex((prev) =>
          filtered.length === 0 ? 0 : (prev - 1 + filtered.length) % filtered.length,
        );
      } else if (e.key === "Enter") {
        if (filtered[activeIndex]) {
          e.preventDefault();
          e.stopPropagation();
          executeItem(filtered[activeIndex]);
        }
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [open, filtered, activeIndex, onClose]);

  if (!open) return null;

  const grouped: Array<{ key: PaletteCommandItem["group"]; title: string; items: Array<{ item: PaletteCommandItem; flatIndex: number }> }> = [];
  const groupTitles: Record<PaletteCommandItem["group"], string> = {
    navigation: t("cmd.groupNavigation"),
    actions: t("cmd.groupActions"),
    episodes: t("cmd.groupEpisodes"),
  };

  const order: Array<PaletteCommandItem["group"]> = ["navigation", "actions", "episodes"];
  const indexMap = new Map<string, number>();
  filtered.forEach((item, idx) => indexMap.set(item.id, idx));

  for (const grp of order) {
    const grpItems = filtered
      .filter((i) => i.group === grp)
      .map((item) => ({ item, flatIndex: indexMap.get(item.id) ?? 0 }));
    if (grpItems.length > 0) {
      grouped.push({ key: grp, title: groupTitles[grp], items: grpItems });
    }
  }

  return (
    <div
      className="modal-backdrop cmd-palette-backdrop"
      data-testid="command-palette"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="cmd-palette-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={t("cmd.dialogAria")}
      >
        <div className="cmd-palette-search">
          <Search size={16} className="muted" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            data-testid="cmd-palette-input"
            placeholder={t("cmd.placeholder")}
            aria-label={t("cmd.placeholder")}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd className="cmd-kbd">Esc</kbd>
        </div>

        <div className="cmd-palette-list" role="listbox">
          {grouped.length === 0 ? (
            <div className="cmd-palette-empty">{t("cmd.empty")}</div>
          ) : (
            grouped.map((section) => (
              <div key={section.key} className="cmd-palette-group">
                <div className="cmd-palette-group-title">{section.title}</div>
                {section.items.map(({ item, flatIndex }) => {
                  const isActive = flatIndex === activeIndex;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      className={`cmd-palette-item ${isActive ? "active" : ""}`}
                      data-testid={`cmd-palette-item-${item.id}`}
                      onMouseEnter={() => setActiveIndex(flatIndex)}
                      onClick={() => executeItem(item)}
                    >
                      <span className="cmd-palette-item-icon">{item.icon}</span>
                      <span className="cmd-palette-item-label">{item.label}</span>
                      {item.subtitle && (
                        <span className="cmd-palette-item-sub muted small">{item.subtitle}</span>
                      )}
                    </button>
                  );
                })}
              </div>
            ))
          )}
          {loadingEpisodes && (
            <div className="cmd-palette-loading small muted">{t("cmd.loadingEpisodes")}</div>
          )}
        </div>

        <div className="cmd-palette-footer small muted">
          <span>{t("cmd.hintNavigate")}</span>
          <span>{t("cmd.hintSelect")}</span>
          <span>{t("cmd.hintClose")}</span>
        </div>
      </div>
    </div>
  );
};
