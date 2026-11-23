import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Eye,
  EyeOff,
  LayoutGrid,
  Maximize2,
  Pause,
  PictureInPicture2,
  Play,
  Repeat,
  Scissors,
  SkipBack,
  SkipForward,
} from "lucide-react";
import { frameUrl, previewVideoUrl } from "../api/client";
import type {
  CameraSpec,
  GroundingKeyframe,
  GroundingResult,
  GroundingStatusKind,
  QcSegment,
} from "../api/types";
import { useAuth } from "../stores/authStore";

export type PlayerViewMode = "single" | "pip" | "grid";

export interface StudioPlayerHandle {
  togglePlay: () => void;
  stepTime: (deltaS: number) => void;
  seekTo: (tS: number) => void;
  isPlaying: () => boolean;
}

export interface StudioPlayerProps {
  episodeId: string;
  cameras: CameraSpec[];
  durationS: number;
  fps: number;
  grounding: GroundingResult;
  currentTime: number;
  onTimeUpdate: (tS: number) => void;
  onWatchInterval?: (tStart: number, tEnd: number) => void;
  qcSegments?: QcSegment[];
  selectedSegment?: QcSegment | null;
  cutSkipEnabled?: boolean;
  onCutSkipChange?: (enabled: boolean) => void;
  loopSegmentEnabled?: boolean;
  onLoopSegmentChange?: (enabled: boolean) => void;
  disableInternalShortcuts?: boolean;
  controllerRef?: React.MutableRefObject<StudioPlayerHandle | null>;
}

const STATUS_COLORS: Record<GroundingStatusKind, string> = {
  neutral: "#38bdf8",
  good: "#22c55e",
  warn: "#f59e0b",
  error: "#ef4444",
};

const PLAYBACK_RATES = [0.5, 1, 1.5, 2, 3] as const;

function pickNearestKeyframe(keyframes: GroundingKeyframe[], tS: number): GroundingKeyframe | null {
  if (!keyframes || keyframes.length === 0) return null;
  let best = keyframes[0];
  let bestDiff = Math.abs(best.time_s - tS);
  for (let i = 1; i < keyframes.length; i += 1) {
    const diff = Math.abs(keyframes[i].time_s - tS);
    if (diff < bestDiff) {
      best = keyframes[i];
      bestDiff = diff;
    }
  }
  return best;
}

export const StudioPlayer: React.FC<StudioPlayerProps> = ({
  episodeId,
  cameras,
  durationS,
  fps,
  grounding,
  currentTime,
  onTimeUpdate,
  onWatchInterval,
  qcSegments = [],
  selectedSegment = null,
  cutSkipEnabled: cutSkipProp,
  onCutSkipChange,
  loopSegmentEnabled: loopProp,
  onLoopSegmentChange,
  disableInternalShortcuts = false,
  controllerRef,
}) => {
  const { token } = useAuth();
  const primaryCamera = useMemo(() => {
    const mainCam = cameras.find((c) => c.role === "main" || c.role === "front");
    return mainCam?.name ?? cameras[0]?.name ?? "main";
  }, [cameras]);

  const [selectedCamera, setSelectedCamera] = useState<string>(primaryCamera);
  const [viewMode, setViewMode] = useState<PlayerViewMode>("single");
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState<number>(1);
  const [showGrounding, setShowGrounding] = useState<boolean>(true);
  const [hdFrameMode, setHdFrameMode] = useState<boolean>(false);
  const [videoFailed, setVideoFailed] = useState<boolean>(false);
  const [failedSecondary, setFailedSecondary] = useState<Record<string, boolean>>({});
  const [internalCutSkip, setInternalCutSkip] = useState<boolean>(false);
  const [internalLoop, setInternalLoop] = useState<boolean>(false);
  const [cutSkipFlash, setCutSkipFlash] = useState<string | null>(null);

  const cutSkip = cutSkipProp !== undefined ? cutSkipProp : internalCutSkip;
  const setCutSkip = useCallback(
    (next: boolean) => {
      if (onCutSkipChange) onCutSkipChange(next);
      else setInternalCutSkip(next);
    },
    [onCutSkipChange],
  );

  const loopSegment = loopProp !== undefined ? loopProp : internalLoop;
  const setLoopSegment = useCallback(
    (next: boolean) => {
      if (onLoopSegmentChange) onLoopSegmentChange(next);
      else setInternalLoop(next);
    },
    [onLoopSegmentChange],
  );

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const secondaryVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const primaryPaneRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrackedTimeRef = useRef<number>(currentTime);
  const currentTimeRef = useRef<number>(currentTime);
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevHdTimeRef = useRef<number>(currentTime);

  currentTimeRef.current = currentTime;

  useEffect(() => {
    if (!cameras.some((c) => c.name === selectedCamera)) {
      setSelectedCamera(primaryCamera);
    }
  }, [cameras, primaryCamera, selectedCamera]);

  useEffect(() => {
    setVideoFailed(false);
    setFailedSecondary({});
  }, [episodeId, selectedCamera]);

  useEffect(() => {
    return () => {
      if (flashTimerRef.current !== null) {
        clearTimeout(flashTimerRef.current);
      }
    };
  }, []);

  const secondaryCameras = useMemo(
    () => cameras.filter((c) => c.name !== selectedCamera),
    [cameras, selectedCamera],
  );

  const primaryCameraSpec = useMemo(
    () => cameras.find((c) => c.name === selectedCamera) ?? cameras[0] ?? null,
    [cameras, selectedCamera],
  );

  const pipCamera = secondaryCameras[0] ?? null;
  const gridSecondaryCameras = useMemo(() => secondaryCameras.slice(0, 3), [secondaryCameras]);

  // If cameras drops to 1, fall back cleanly to single view
  useEffect(() => {
    if (cameras.length < 2 && viewMode !== "single") {
      setViewMode("single");
    }
  }, [cameras.length, viewMode]);

  const loopBounds = useMemo(() => {
    if (!selectedSegment) return null;
    const start = Math.max(0, Number((selectedSegment.t_start - 0.5).toFixed(3)));
    const end = Math.min(durationS, Number((selectedSegment.t_end + 0.5).toFixed(3)));
    if (end <= start) return null;
    return { start, end };
  }, [selectedSegment, durationS]);

  // Auto-disable loop if selection is cleared
  useEffect(() => {
    if (!selectedSegment && loopSegment) {
      setLoopSegment(false);
    }
  }, [selectedSegment, loopSegment, setLoopSegment]);

  const videoSrc = useMemo(
    () => previewVideoUrl(episodeId, selectedCamera, token),
    [episodeId, selectedCamera, token],
  );

  const hdFrameSrc = useMemo(
    () => frameUrl(episodeId, selectedCamera, currentTime, token),
    [episodeId, selectedCamera, currentTime, token],
  );

  const triggerCutSkipFlash = useCallback((seg: QcSegment) => {
    const span = Math.max(0.1, seg.t_end - seg.t_start).toFixed(1);
    setCutSkipFlash(`✂ Skipped cut_clean (${span}s)`);
    if (flashTimerRef.current !== null) {
      clearTimeout(flashTimerRef.current);
    }
    flashTimerRef.current = setTimeout(() => {
      setCutSkipFlash(null);
      flashTimerRef.current = null;
    }, 1500);
  }, []);

  const syncSecondaryVideos = useCallback(
    (targetT: number, shouldPlay: boolean | null, rate: number) => {
      const clamped = Math.max(0, Math.min(durationS, targetT));
      for (const [camName, secVid] of Object.entries(secondaryVideoRefs.current)) {
        if (!secVid || camName === selectedCamera) continue;
        secVid.playbackRate = rate;
        if (Math.abs(secVid.currentTime - clamped) > 0.12) {
          try {
            secVid.currentTime = clamped;
          } catch {
            // Ignore seek errors when metadata not yet loaded
          }
        }
        if (shouldPlay === true && secVid.paused) {
          void Promise.resolve(secVid.play?.()).catch(() => {});
        } else if (shouldPlay === false && !secVid.paused) {
          secVid.pause?.();
        }
      }
    },
    [durationS, selectedCamera],
  );

  const seekAllVideos = useCallback(
    (tS: number) => {
      const clamped = Math.max(0, Math.min(durationS, Number(tS.toFixed(4))));
      const vid = videoRef.current;
      if (vid && Math.abs(vid.currentTime - clamped) > 0.04) {
        try {
          vid.currentTime = clamped;
        } catch {
          // Ignore seek errors when metadata not yet loaded
        }
      }
      syncSecondaryVideos(clamped, null, playbackRate);
      lastTrackedTimeRef.current = clamped;
      onTimeUpdate(clamped);
    },
    [durationS, onTimeUpdate, playbackRate, syncSecondaryVideos],
  );

  // Sync external seek into primary & secondary <video> elements
  useEffect(() => {
    const clamped = Math.max(0, Math.min(durationS, currentTime));
    const vid = videoRef.current;
    if (vid && Math.abs(vid.currentTime - clamped) > 0.08) {
      try {
        vid.currentTime = clamped;
      } catch {
        // Ignore seek errors when metadata not yet loaded
      }
    }
    syncSecondaryVideos(clamped, null, playbackRate);
    lastTrackedTimeRef.current = clamped;
  }, [currentTime, durationS, playbackRate, syncSecondaryVideos]);

  // Accumulate watch coverage cleanly when scrubbing/inspecting in HD-frame mode
  useEffect(() => {
    if (!hdFrameMode) {
      prevHdTimeRef.current = currentTime;
      return;
    }
    const prevT = prevHdTimeRef.current;
    prevHdTimeRef.current = currentTime;
    if (onWatchInterval && currentTime !== prevT && Math.abs(currentTime - prevT) <= 2.05) {
      onWatchInterval(Math.min(prevT, currentTime), Math.max(prevT, currentTime));
    }
  }, [currentTime, hdFrameMode, onWatchInterval]);

  useEffect(() => {
    const vid = videoRef.current;
    if (vid) {
      vid.playbackRate = playbackRate;
    }
    syncSecondaryVideos(currentTimeRef.current, null, playbackRate);
  }, [playbackRate, syncSecondaryVideos]);

  // Keep newly mounted secondary videos synced when switching to PiP or Grid view
  useEffect(() => {
    if (viewMode === "single" || hdFrameMode) return;
    syncSecondaryVideos(currentTimeRef.current, playing, playbackRate);
  }, [viewMode, hdFrameMode, playing, playbackRate, syncSecondaryVideos]);

  const togglePlay = useCallback(() => {
    const vid = videoRef.current;
    if (hdFrameMode || !vid || videoFailed) {
      setPlaying((prev) => {
        const next = !prev;
        if (next && loopSegment && loopBounds) {
          const cur = currentTimeRef.current;
          if (cur < loopBounds.start || cur >= loopBounds.end) {
            onTimeUpdate(loopBounds.start);
          }
        }
        return next;
      });
      return;
    }
    if (vid.paused) {
      if (loopSegment && loopBounds) {
        if (vid.currentTime < loopBounds.start || vid.currentTime >= loopBounds.end) {
          try {
            vid.currentTime = loopBounds.start;
          } catch {
            // Ignore seek before metadata
          }
          lastTrackedTimeRef.current = loopBounds.start;
          onTimeUpdate(loopBounds.start);
        }
      }
      void Promise.resolve(vid.play?.()).catch(() => {
        setPlaying(false);
      });
      syncSecondaryVideos(vid.currentTime, true, playbackRate);
      setPlaying(true);
    } else {
      vid.pause?.();
      syncSecondaryVideos(vid.currentTime, false, playbackRate);
      setPlaying(false);
    }
  }, [
    hdFrameMode,
    videoFailed,
    loopSegment,
    loopBounds,
    onTimeUpdate,
    syncSecondaryVideos,
    playbackRate,
  ]);

  const stepTime = useCallback(
    (deltaS: number) => {
      const prevT = currentTimeRef.current;
      const next = Math.max(0, Math.min(durationS, Number((prevT + deltaS).toFixed(4))));
      if (videoRef.current && !videoRef.current.paused) {
        videoRef.current.pause();
        setPlaying(false);
      }
      syncSecondaryVideos(next, false, playbackRate);
      if (onWatchInterval && next !== prevT && Math.abs(next - prevT) <= 2.05) {
        onWatchInterval(Math.min(prevT, next), Math.max(prevT, next));
      }
      lastTrackedTimeRef.current = next;
      onTimeUpdate(next);
    },
    [durationS, onTimeUpdate, onWatchInterval, playbackRate, syncSecondaryVideos],
  );

  // Expose imperative controls to ReviewStudioPage unified keyboard handler
  useEffect(() => {
    if (!controllerRef) return;
    controllerRef.current = {
      togglePlay,
      stepTime,
      seekTo: seekAllVideos,
      isPlaying: () => playing,
    };
    return () => {
      if (controllerRef.current) {
        controllerRef.current = null;
      }
    };
  }, [controllerRef, playing, seekAllVideos, stepTime, togglePlay]);

  // Fallback internal shortcuts when StudioPlayer is mounted standalone (e.g. unit tests)
  useEffect(() => {
    if (disableInternalShortcuts) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepTime(-1 / Math.max(1, fps));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        stepTime(1 / Math.max(1, fps));
      } else if (e.key.toLowerCase() === "j") {
        e.preventDefault();
        stepTime(e.shiftKey ? -2.0 : -0.5);
      } else if (e.key.toLowerCase() === "l") {
        e.preventDefault();
        stepTime(e.shiftKey ? 2.0 : 0.5);
      } else if (e.key.toLowerCase() === "k" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setCutSkip(!cutSkip);
      } else if (e.key.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey && selectedSegment) {
        e.preventDefault();
        setLoopSegment(!loopSegment);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    disableInternalShortcuts,
    fps,
    stepTime,
    togglePlay,
    cutSkip,
    setCutSkip,
    loopSegment,
    setLoopSegment,
    selectedSegment,
  ]);

  const applyPlaybackPolicies = useCallback(
    (nextT: number): number => {
      let targetT = nextT;
      if (cutSkip && qcSegments.length > 0) {
        let skippedSeg: QcSegment | null = null;
        for (let iter = 0; iter < 5; iter += 1) {
          const hitCut = qcSegments.find(
            (s) =>
              s.splice_policy === "cut_clean" &&
              targetT >= s.t_start - 0.03 &&
              targetT < s.t_end - 0.04,
          );
          if (!hitCut) break;
          targetT = Math.min(durationS, Number((hitCut.t_end + 0.02).toFixed(3)));
          skippedSeg = hitCut;
        }
        if (skippedSeg) {
          triggerCutSkipFlash(skippedSeg);
        }
      }

      if (loopSegment && loopBounds && targetT >= loopBounds.end) {
        targetT = loopBounds.start;
      }
      return targetT;
    },
    [cutSkip, durationS, loopBounds, loopSegment, qcSegments, triggerCutSkipFlash],
  );

  // Playback timer in HD-frame mode or when preview video failed
  useEffect(() => {
    if (!playing || (!hdFrameMode && !videoFailed)) return;
    const tickMs = 100;
    const timer = window.setInterval(() => {
      const prevT = currentTimeRef.current;
      const rawNext = Math.min(durationS, Number((prevT + (tickMs / 1000) * playbackRate).toFixed(3)));
      if (onWatchInterval && rawNext > prevT) {
        onWatchInterval(prevT, rawNext);
      }
      const nextT = applyPlaybackPolicies(rawNext);
      lastTrackedTimeRef.current = nextT;
      onTimeUpdate(nextT);
      if (nextT >= durationS && !loopSegment) {
        setPlaying(false);
      }
    }, tickMs);
    return () => window.clearInterval(timer);
  }, [
    playing,
    hdFrameMode,
    videoFailed,
    durationS,
    playbackRate,
    onWatchInterval,
    applyPlaybackPolicies,
    onTimeUpdate,
    loopSegment,
  ]);

  // Draw read-only 2D grounding canvas strictly aligned to primary camera pane
  const drawGrounding = useCallback(() => {
    const canvas = canvasRef.current;
    const pane = primaryPaneRef.current;
    if (!canvas || !pane) return;
    const width = pane.clientWidth || 640;
    const height = pane.clientHeight || 360;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    if (!showGrounding || !grounding?.object_tracks_2d) return;

    const tracks = Object.values(grounding.object_tracks_2d);
    for (const track of tracks) {
      const kf = pickNearestKeyframe(track.keyframes, currentTime);
      if (!kf || !kf.visible) continue;
      const [nx1, ny1, nx2, ny2] = kf.bbox_norm_1000;
      const x1 = (nx1 / 1000) * width;
      const y1 = (ny1 / 1000) * height;
      const bw = Math.max(2, ((nx2 - nx1) / 1000) * width);
      const bh = Math.max(2, ((ny2 - ny1) / 1000) * height);

      const color = STATUS_COLORS[kf.status_kind] ?? STATUS_COLORS.neutral;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(track.is_static_container ? [6, 4] : []);
      ctx.strokeRect(x1, y1, bw, bh);

      const label = `${track.name || track.object_id}${kf.state ? ` · ${kf.state}` : ""}`;
      ctx.font = "12px system-ui, sans-serif";
      const textW = ctx.measureText(label).width + 10;
      const labelY =
        track.tag_pos === "bottom" ? Math.min(height - 18, y1 + bh + 2) : Math.max(0, y1 - 20);
      ctx.fillStyle = "rgba(15, 23, 42, 0.82)";
      ctx.fillRect(x1, labelY, textW, 18);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, x1 + 5, labelY + 13);
      ctx.restore();
    }
  }, [currentTime, grounding, showGrounding]);

  useEffect(() => {
    drawGrounding();
  }, [drawGrounding, viewMode]);

  useEffect(() => {
    const el = primaryPaneRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => drawGrounding());
    ro.observe(el);
    return () => ro.disconnect();
  }, [drawGrounding, viewMode]);

  const handleVideoTimeUpdate = () => {
    const vid = videoRef.current;
    if (!vid) return;
    const rawT = vid.currentTime;
    const prevT = lastTrackedTimeRef.current;
    const maxStep = Math.max(0.6, 0.75 * playbackRate);
    if (onWatchInterval && rawT > prevT && rawT - prevT <= maxStep) {
      onWatchInterval(prevT, rawT);
    }

    const adjustedT = applyPlaybackPolicies(rawT);
    if (Math.abs(adjustedT - rawT) > 0.02) {
      try {
        vid.currentTime = adjustedT;
      } catch {
        // Ignore seek error
      }
      syncSecondaryVideos(adjustedT, null, playbackRate);
      lastTrackedTimeRef.current = adjustedT;
      onTimeUpdate(adjustedT);
      return;
    }

    syncSecondaryVideos(rawT, null, playbackRate);
    lastTrackedTimeRef.current = rawT;
    onTimeUpdate(rawT);
  };

  const activeMaskSegment = useMemo(() => {
    if (!cutSkip) return null;
    return (
      qcSegments.find(
        (s) =>
          s.splice_policy === "mask_loss" &&
          currentTime >= s.t_start &&
          currentTime <= s.t_end,
      ) ?? null
    );
  }, [cutSkip, qcSegments, currentTime]);

  const renderSecondaryMedia = (cam: CameraSpec, showLabel: boolean) => {
    const isFailed = Boolean(failedSecondary[cam.name]);
    const src = previewVideoUrl(episodeId, cam.name, token);
    const imgSrc = frameUrl(episodeId, cam.name, currentTime, token);
    return (
      <>
        {!hdFrameMode && !isFailed ? (
          <video
            ref={(el) => {
              secondaryVideoRefs.current[cam.name] = el;
            }}
            className="player-media"
            data-testid={`secondary-video-${cam.name}`}
            src={src}
            muted
            playsInline
            preload="metadata"
            onError={() =>
              setFailedSecondary((prev) => ({
                ...prev,
                [cam.name]: true,
              }))
            }
          />
        ) : (
          <img
            className="player-media"
            data-testid={`secondary-frame-${cam.name}`}
            src={imgSrc}
            alt={`${episodeId} ${cam.name} @ ${currentTime.toFixed(2)}s`}
          />
        )}
        {showLabel && (
          <span className="cam-overlay-label">
            {cam.name} ({cam.role})
          </span>
        )}
      </>
    );
  };

  const trackCount = Object.keys(grounding?.object_tracks_2d ?? {}).length;
  const hasMultiCam = cameras.length >= 2;
  const isGridMode = viewMode === "grid" && hasMultiCam;
  const isPipMode = viewMode === "pip" && hasMultiCam && Boolean(pipCamera);

  return (
    <div className="studio-player-card" data-testid="studio-player">
      <div className="player-topbar">
        <div className="camera-tabs" role="tablist" aria-label="Camera views">
          {cameras.map((cam) => (
            <button
              key={cam.name}
              type="button"
              role="tab"
              aria-selected={selectedCamera === cam.name}
              className={`cam-tab ${selectedCamera === cam.name ? "active" : ""}`}
              onClick={() => setSelectedCamera(cam.name)}
            >
              <Camera size={13} />
              <span>{cam.name}</span>
              <span className="cam-role-chip">{cam.role}</span>
            </button>
          ))}
        </div>

        <div className="player-toggles">
          {/* View Mode Switcher: Single / PiP / 2x2 Grid */}
          <div className="view-mode-switcher" role="group" aria-label="Multi-camera view mode" data-testid="view-mode-switcher">
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "single" ? "active" : ""}`}
              onClick={() => setViewMode("single")}
              data-testid="view-mode-single"
              title="Single camera full view"
            >
              <Maximize2 size={12} /> Single
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "pip" ? "active" : ""}`}
              disabled={!hasMultiCam}
              onClick={() => setViewMode("pip")}
              data-testid="view-mode-pip"
              title={hasMultiCam ? "Primary + secondary Picture-in-Picture (PiP)" : "Requires at least 2 cameras"}
            >
              <PictureInPicture2 size={12} /> PiP
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "grid" ? "active" : ""}`}
              disabled={!hasMultiCam}
              onClick={() => setViewMode("grid")}
              data-testid="view-mode-grid"
              title={hasMultiCam ? "Multi-camera grid (2x2)" : "Requires at least 2 cameras"}
            >
              <LayoutGrid size={12} /> 2×2
            </button>
          </div>

          <button
            type="button"
            className={`pill-btn ${cutSkip ? "active" : ""}`}
            onClick={() => setCutSkip(!cutSkip)}
            title="Cut-Skip Preview (K): automatically skips cut_clean segments and highlights mask_loss intervals"
            data-testid="toggle-cut-skip"
          >
            <Scissors size={13} />
            <span>Cut-Skip (K)</span>
          </button>

          <button
            type="button"
            className={`pill-btn ${showGrounding ? "active" : ""}`}
            onClick={() => setShowGrounding((v) => !v)}
            title="2D Grounding boxes are read-only visual aids"
            data-testid="toggle-grounding"
          >
            {showGrounding ? <Eye size={13} /> : <EyeOff size={13} />}
            <span>2D Boxes ({trackCount})</span>
          </button>

          <button
            type="button"
            className={`pill-btn ${hdFrameMode ? "active" : ""}`}
            data-testid="toggle-hd-frame"
            onClick={() => {
              if (!hdFrameMode && videoRef.current && !videoRef.current.paused) {
                videoRef.current.pause();
                syncSecondaryVideos(currentTimeRef.current, false, playbackRate);
                setPlaying(false);
              }
              setHdFrameMode((v) => !v);
            }}
          >
            HD Frame
          </button>
        </div>
      </div>

      <div
        className={`player-viewport ${isGridMode ? "grid-mode" : ""} ${isGridMode && gridSecondaryCameras.length >= 2 ? "grid-2x2" : ""} ${isPipMode ? "pip-mode" : ""}`}
        data-testid="player-viewport"
        data-view-mode={viewMode}
      >
        {/* Primary camera pane — 2D Grounding canvas is strictly scoped & aligned here */}
        <div className="primary-cam-pane" ref={primaryPaneRef} data-testid="primary-cam-pane">
          {!hdFrameMode && !videoFailed ? (
            <video
              ref={videoRef}
              className="player-media"
              data-testid="primary-video"
              src={videoSrc}
              playsInline
              preload="metadata"
              onTimeUpdate={handleVideoTimeUpdate}
              onPlay={() => {
                setPlaying(true);
                syncSecondaryVideos(videoRef.current?.currentTime ?? currentTimeRef.current, true, playbackRate);
              }}
              onPause={() => {
                setPlaying(false);
                syncSecondaryVideos(videoRef.current?.currentTime ?? currentTimeRef.current, false, playbackRate);
              }}
              onEnded={() => {
                setPlaying(false);
                syncSecondaryVideos(durationS, false, playbackRate);
              }}
              onError={() => setVideoFailed(true)}
            />
          ) : (
            <img
              className="player-media"
              data-testid="primary-hd-frame"
              src={hdFrameSrc}
              alt={`${episodeId} ${selectedCamera} @ ${currentTime.toFixed(2)}s`}
            />
          )}

          <canvas
            ref={canvasRef}
            className="grounding-canvas-readonly"
            data-testid="grounding-canvas"
            aria-label="2D Grounding read-only overlay"
          />

          {isGridMode && primaryCameraSpec && (
            <span className="cam-overlay-label">
              {primaryCameraSpec.name} ({primaryCameraSpec.role} · primary)
            </span>
          )}
        </div>

        {/* Multi-cam 2x2 grid secondary panes */}
        {isGridMode &&
          gridSecondaryCameras.map((cam) => (
            <div
              key={cam.name}
              className="grid-cam-pane"
              data-testid={`grid-cam-${cam.name}`}
              onClick={() => setSelectedCamera(cam.name)}
              title={`Click to set ${cam.name} as primary view`}
            >
              {renderSecondaryMedia(cam, true)}
            </div>
          ))}

        {/* Picture-in-Picture secondary overlay */}
        {isPipMode && pipCamera && (
          <div
            className="pip-overlay"
            data-testid={`pip-cam-${pipCamera.name}`}
            title={`Click to switch ${pipCamera.name} to primary view`}
            onClick={() => setSelectedCamera(pipCamera.name)}
          >
            <span className="pip-cam-badge">{pipCamera.name}</span>
            {renderSecondaryMedia(pipCamera, false)}
          </div>
        )}

        {/* MASK loss subtle banner in Cut-Skip Preview mode */}
        {activeMaskSegment && (
          <div className="mask-loss-banner" data-testid="mask-loss-banner">
            <span>
              MASK · {activeMaskSegment.name || activeMaskSegment.label_id} (
              {activeMaskSegment.mask_side ?? "both"})
            </span>
          </div>
        )}

        {/* Brief cut-skip flash badge */}
        {cutSkipFlash && (
          <div className="cut-skip-flash" data-testid="cut-skip-flash">
            {cutSkipFlash}
          </div>
        )}
      </div>

      <div className="player-controls">
        <div className="transport-buttons">
          <button type="button" className="icon-btn" onClick={() => stepTime(-0.5)} title="-0.5s (J)">
            -0.5s
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(-1 / Math.max(1, fps))}
            title="Previous frame (←)"
          >
            <SkipBack size={14} />
          </button>
          <button
            type="button"
            className="primary-btn compact"
            onClick={togglePlay}
            data-testid="play-pause-btn"
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
            <span>{playing ? "Pause" : "Play"}</span>
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(1 / Math.max(1, fps))}
            title="Next frame (→)"
          >
            <SkipForward size={14} />
          </button>
          <button type="button" className="icon-btn" onClick={() => stepTime(0.5)} title="+0.5s (L)">
            +0.5s
          </button>
          <button
            type="button"
            className={`pill-btn ${loopSegment ? "active" : ""}`}
            disabled={!selectedSegment && !loopSegment}
            data-testid="toggle-segment-loop"
            title={
              selectedSegment
                ? "Loop selected segment (±0.5s padding) (R)"
                : "Select a QC segment first to enable looping (R)"
            }
            onClick={() => {
              const next = !loopSegment;
              setLoopSegment(next);
              if (next && loopBounds) {
                if (currentTime < loopBounds.start || currentTime > loopBounds.end) {
                  seekAllVideos(loopBounds.start);
                }
              }
            }}
          >
            <Repeat size={13} /> Loop (R)
          </button>
          {loopSegment && loopBounds && (
            <span className="loop-range-pill" data-testid="loop-range-pill">
              {loopBounds.start.toFixed(1)}–{loopBounds.end.toFixed(1)}s
            </span>
          )}
        </div>

        <div className="time-readout">
          <strong>{currentTime.toFixed(2)}s</strong> / {durationS.toFixed(2)}s · {fps.toFixed(0)} fps
        </div>

        <div className="speed-selector">
          {PLAYBACK_RATES.map((rate) => (
            <button
              key={rate}
              type="button"
              className={`speed-btn ${playbackRate === rate ? "active" : ""}`}
              onClick={() => setPlaybackRate(rate)}
            >
              {rate}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
