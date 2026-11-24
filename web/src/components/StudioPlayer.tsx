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
  GroundingResult,
  GroundingStatusKind,
  QcSegment,
} from "../api/types";
import { useAuth } from "../stores/authStore";
import { useI18n } from "../stores/i18nStore";
import {
  computeContainRect,
  computeLabelPlacement,
  projectNormBoxToRect,
  resolveGroundedCameraName,
  resolveMediaDimensions,
  sampleTrackBoxAt,
} from "../utils/groundingProjection";

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

type VideoFrameCallbackLike = (now: number, metadata?: { mediaTime?: number }) => void;

type VideoElementWithRVFC = {
  requestVideoFrameCallback?: (callback: VideoFrameCallbackLike) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

function drawCornerBrackets(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  lineWidth: number,
) {
  if (typeof ctx.moveTo !== "function" || typeof ctx.lineTo !== "function") return;
  const arm = Math.max(4, Math.min(14, Math.min(w, h) * 0.24));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash([]);
  ctx.beginPath();
  // Top-left
  ctx.moveTo(x, y + arm);
  ctx.lineTo(x, y);
  ctx.lineTo(x + arm, y);
  // Top-right
  ctx.moveTo(x + w - arm, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + arm);
  // Bottom-right
  ctx.moveTo(x + w, y + h - arm);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - arm, y + h);
  // Bottom-left
  ctx.moveTo(x + arm, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - arm);
  ctx.stroke();
  ctx.restore();
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
  const { t } = useI18n();
  const primaryCamera = useMemo(() => {
    const mainCam = cameras.find((c) => c.role === "main" || c.role === "front");
    return mainCam?.name ?? cameras[0]?.name ?? "main";
  }, [cameras]);

  const [selectedCamera, setSelectedCamera] = useState<string>(primaryCamera);
  const [viewMode, setViewMode] = useState<PlayerViewMode>("single");
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState<number>(1);
  const [showGrounding, setShowGrounding] = useState<boolean>(true);
  const [hiddenObjectIds, setHiddenObjectIds] = useState<Record<string, boolean>>({});
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
  const hdImgRef = useRef<HTMLImageElement | null>(null);
  const secondaryVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const primaryPaneRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastTrackedTimeRef = useRef<number>(currentTime);
  const currentTimeRef = useRef<number>(currentTime);
  const playingRef = useRef<boolean>(playing);
  const hdFrameModeRef = useRef<boolean>(hdFrameMode);
  const videoFailedRef = useRef<boolean>(videoFailed);
  const showGroundingRef = useRef<boolean>(showGrounding);
  const hiddenObjectIdsRef = useRef<Record<string, boolean>>(hiddenObjectIds);
  const groundingRef = useRef<GroundingResult>(grounding);
  const selectedCameraRef = useRef<string>(selectedCamera);
  const primaryCameraRef = useRef<string>(primaryCamera);
  const primaryCameraSpecRef = useRef<CameraSpec | null>(null);
  const prevCameraRef = useRef<string>(selectedCamera);
  const restoringTimeRef = useRef<number | null>(null);
  const shouldResumeAfterLoadRef = useRef<boolean>(false);
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevHdTimeRef = useRef<number>(currentTime);

  currentTimeRef.current = currentTime;
  playingRef.current = playing;
  hdFrameModeRef.current = hdFrameMode;
  videoFailedRef.current = videoFailed;
  showGroundingRef.current = showGrounding;
  hiddenObjectIdsRef.current = hiddenObjectIds;
  groundingRef.current = grounding;
  selectedCameraRef.current = selectedCamera;
  primaryCameraRef.current = primaryCamera;

  useEffect(() => {
    if (!cameras.some((c) => c.name === selectedCamera)) {
      setSelectedCamera(primaryCamera);
    }
  }, [cameras, primaryCamera, selectedCamera]);

  // Reset playback cleanly when switching to a different episode
  useEffect(() => {
    setPlaying(false);
    setVideoFailed(false);
    setFailedSecondary({});
    setHiddenObjectIds((prev) => (Object.keys(prev).length === 0 ? prev : {}));
    restoringTimeRef.current = null;
    shouldResumeAfterLoadRef.current = false;
    if (videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
    }
  }, [episodeId]);

  // Preserve playhead & playback state when switching cameras on the same episode
  useEffect(() => {
    if (prevCameraRef.current !== selectedCamera) {
      prevCameraRef.current = selectedCamera;
      restoringTimeRef.current = currentTimeRef.current;
      shouldResumeAfterLoadRef.current = playingRef.current;
      setVideoFailed(false);
    }
  }, [selectedCamera]);

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
  primaryCameraSpecRef.current = primaryCameraSpec;

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

  const triggerCutSkipFlash = useCallback(
    (seg: QcSegment) => {
      const span = Math.max(0.1, seg.t_end - seg.t_start).toFixed(1);
      setCutSkipFlash(t("player.cutSkipFlash", { span }));
      if (flashTimerRef.current !== null) {
        clearTimeout(flashTimerRef.current);
      }
      flashTimerRef.current = setTimeout(() => {
        setCutSkipFlash(null);
        flashTimerRef.current = null;
      }, 1500);
    },
    [t],
  );

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
      if (vid && Math.abs(vid.currentTime - clamped) > 0.005) {
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

  // Sync external seek into primary & secondary <video> elements (frame-accurate when paused)
  useEffect(() => {
    const clamped = Math.max(0, Math.min(durationS, currentTime));
    const vid = videoRef.current;
    const seekThreshold = playingRef.current ? 0.12 : 0.005;
    if (vid && Math.abs(vid.currentTime - clamped) > seekThreshold) {
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

  const getEffectiveDrawTime = useCallback((): number => {
    const vid = videoRef.current;
    if (
      playingRef.current &&
      !hdFrameModeRef.current &&
      !videoFailedRef.current &&
      vid &&
      !vid.paused &&
      restoringTimeRef.current === null
    ) {
      return vid.currentTime;
    }
    return currentTimeRef.current;
  }, []);

  // Draw read-only 2D grounding canvas strictly aligned to displayed video rect inside object-fit: contain
  const drawGroundingAt = useCallback((tS: number) => {
    const canvas = canvasRef.current;
    const pane = primaryPaneRef.current;
    if (!canvas || !pane) return;

    const paneDomRect = typeof pane.getBoundingClientRect === "function" ? pane.getBoundingClientRect() : null;
    const cssWidth = (paneDomRect && paneDomRect.width > 0 ? paneDomRect.width : pane.clientWidth) || 640;
    const cssHeight = (paneDomRect && paneDomRect.height > 0 ? paneDomRect.height : pane.clientHeight) || 360;
    const dpr = typeof window !== "undefined" ? Math.min(Math.max(window.devicePixelRatio || 1, 1), 2) : 1;
    const backingW = Math.max(1, Math.round(cssWidth * dpr));
    const backingH = Math.max(1, Math.round(cssHeight * dpr));

    if (canvas.width !== backingW) canvas.width = backingW;
    if (canvas.height !== backingH) canvas.height = backingH;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (typeof ctx.setTransform === "function") {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const g = groundingRef.current;
    if (!showGroundingRef.current || !g?.object_tracks_2d) return;

    // Camera matching (F4): only render boxes when selected camera matches grounded camera
    const groundedCam = resolveGroundedCameraName(g, primaryCameraRef.current);
    if (selectedCameraRef.current !== groundedCam) return;

    // Resolve natural media dimensions & exact object-fit: contain content rectangle (F2)
    const [mediaW, mediaH] = resolveMediaDimensions({
      videoEl: videoRef.current,
      imgEl: hdImgRef.current,
      hdFrameMode: hdFrameModeRef.current,
      videoFailed: videoFailedRef.current,
      cameraSpec: primaryCameraSpecRef.current,
      grounding: g,
      fallbackW: cssWidth,
      fallbackH: cssHeight,
    });
    const contentRect = computeContainRect(cssWidth, cssHeight, mediaW, mediaH);
    const rectSig = `${contentRect.x.toFixed(1)},${contentRect.y.toFixed(1)},${contentRect.width.toFixed(1)},${contentRect.height.toFixed(1)}`;
    if (canvas.dataset.contentRect !== rectSig) {
      canvas.dataset.contentRect = rectSig;
    }

    const trackEntries = Object.entries(g.object_tracks_2d);
    const compactPane = contentRect.width < 360;
    const fontSize = compactPane ? 11 : 12;
    const pillH = compactPane ? 17 : 19;

    for (const [trackKey, track] of trackEntries) {
      const objId = track.object_id || trackKey;
      if (hiddenObjectIdsRef.current[objId]) continue;
      const sample = sampleTrackBoxAt(g, track, tS);
      if (!sample) continue;

      const box = projectNormBoxToRect(sample.bbox, contentRect);
      const color = STATUS_COLORS[sample.statusKind] ?? STATUS_COLORS.neutral;
      const isStaticLocked = sample.provenance === "static_locked";
      const isAnchor = sample.provenance === "vlm_anchor";
      const isLowConf = sample.provenance === "tracked" && sample.confPct !== null && sample.confPct < 70;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isAnchor ? 2 : 1.75;
      ctx.globalAlpha = isStaticLocked ? 0.86 : isLowConf ? 0.82 : 1;
      ctx.shadowColor = "rgba(15, 23, 42, 0.72)";
      ctx.shadowBlur = 3;
      ctx.setLineDash(isStaticLocked ? [6, 4] : isLowConf ? [5, 2] : []);
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      ctx.shadowBlur = 0;

      // Crisp corner brackets on active VLM anchors & tracked objects
      if (!isStaticLocked) {
        drawCornerBrackets(ctx, box.x, box.y, box.width, box.height, color, isAnchor ? 3 : 2.5);
      }

      const prefix = isAnchor ? "◆ " : isStaticLocked ? "▪ " : "";
      const baseName = track.name || track.object_id;
      const statePart = sample.state ? ` · ${sample.state}` : "";
      const confPart =
        sample.provenance === "tracked" && sample.confPct !== null && !compactPane
          ? ` ${sample.confPct}%`
          : "";
      const label = `${prefix}${baseName}${statePart}${confPart}`;

      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
      ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;
      const textW = Math.ceil(ctx.measureText(label).width) + 13;
      const pill = computeLabelPlacement(box, contentRect, textW, pillH, track.tag_pos);

      ctx.fillStyle = "rgba(15, 23, 42, 0.86)";
      ctx.fillRect(pill.x, pill.y, textW, pillH);
      ctx.fillStyle = color;
      ctx.fillRect(pill.x, pill.y, 2.5, pillH);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, pill.x + 6, pill.y + pillH - 5);
      ctx.restore();
    }
  }, []);

  const drawGroundingAtRef = useRef(drawGroundingAt);
  drawGroundingAtRef.current = drawGroundingAt;

  // Immediate redraw on scrub, mode switch, camera switch, grounding data update, or toggle
  useEffect(() => {
    drawGroundingAt(getEffectiveDrawTime());
  }, [
    currentTime,
    grounding,
    showGrounding,
    hiddenObjectIds,
    selectedCamera,
    primaryCamera,
    primaryCameraSpec,
    viewMode,
    hdFrameMode,
    videoFailed,
    drawGroundingAt,
    getEffectiveDrawTime,
  ]);

  // True 30fps video frame sync via requestVideoFrameCallback (with requestAnimationFrame fallback)
  useEffect(() => {
    if (!playing || hdFrameMode || videoFailed) return;
    const vid = videoRef.current;
    if (!vid) return;

    let cancelled = false;
    let rvfcHandle: number | null = null;
    let rafHandle: number | null = null;
    const rvfcVid = vid as unknown as VideoElementWithRVFC;

    if (typeof rvfcVid.requestVideoFrameCallback === "function") {
      const onVideoFrame: VideoFrameCallbackLike = (_now, metadata) => {
        if (cancelled) return;
        const mediaT =
          typeof metadata?.mediaTime === "number" && Number.isFinite(metadata.mediaTime)
            ? metadata.mediaTime
            : vid.currentTime;
        drawGroundingAtRef.current(mediaT);
        if (playingRef.current && !vid.ended) {
          rvfcHandle = rvfcVid.requestVideoFrameCallback!(onVideoFrame);
        } else {
          rvfcHandle = null;
        }
      };
      rvfcHandle = rvfcVid.requestVideoFrameCallback(onVideoFrame);
    } else if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
      const onRaf = () => {
        if (cancelled) return;
        drawGroundingAtRef.current(vid.currentTime);
        if (playingRef.current && !vid.ended) {
          rafHandle = window.requestAnimationFrame(onRaf);
        } else {
          rafHandle = null;
        }
      };
      rafHandle = window.requestAnimationFrame(onRaf);
    }

    return () => {
      cancelled = true;
      if (rvfcHandle !== null && typeof rvfcVid.cancelVideoFrameCallback === "function") {
        rvfcVid.cancelVideoFrameCallback(rvfcHandle);
      }
      if (
        rafHandle !== null &&
        typeof window !== "undefined" &&
        typeof window.cancelAnimationFrame === "function"
      ) {
        window.cancelAnimationFrame(rafHandle);
      }
    };
  }, [playing, hdFrameMode, videoFailed, selectedCamera, episodeId, viewMode]);

  // Redraw whenever intrinsic <video> dimensions change
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const onIntrinsicResize = () => {
      drawGroundingAtRef.current(getEffectiveDrawTime());
    };
    vid.addEventListener("resize", onIntrinsicResize);
    return () => vid.removeEventListener("resize", onIntrinsicResize);
  }, [hdFrameMode, videoFailed, selectedCamera, episodeId, viewMode, getEffectiveDrawTime]);

  // Observe container & active media layout resizes (window resize, split drag, Single/PiP/Grid transitions)
  useEffect(() => {
    const pane = primaryPaneRef.current;
    if (!pane || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      drawGroundingAtRef.current(getEffectiveDrawTime());
    });
    ro.observe(pane);
    if (videoRef.current) ro.observe(videoRef.current);
    if (hdImgRef.current) ro.observe(hdImgRef.current);
    return () => ro.disconnect();
  }, [viewMode, hdFrameMode, videoFailed, selectedCamera, episodeId, getEffectiveDrawTime]);

  const handleVideoTimeUpdate = () => {
    const vid = videoRef.current;
    if (!vid) return;
    const rawT = vid.currentTime;
    // Ignore spurious 0s timeupdate emitted while a newly switched camera reloads metadata
    if (restoringTimeRef.current !== null && restoringTimeRef.current > 0.1 && rawT < 0.05) {
      return;
    }
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
            onLoadedMetadata={(e) => {
              const secVid = e.currentTarget;
              secVid.playbackRate = playbackRate;
              const target = Math.max(0, Math.min(durationS, currentTimeRef.current));
              if (Math.abs(secVid.currentTime - target) > 0.08) {
                try {
                  secVid.currentTime = target;
                } catch {
                  // Ignore seek errors before metadata is fully ready
                }
              }
              if (playingRef.current && secVid.paused) {
                void Promise.resolve(secVid.play?.()).catch(() => {});
              }
            }}
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

  const groundingTracks = useMemo(
    () =>
      Object.entries(grounding?.object_tracks_2d ?? {}).map(([key, track]) => ({
        id: track.object_id || key,
        name: track.name || track.object_id || key,
        track,
      })),
    [grounding],
  );
  const trackCount = groundingTracks.length;
  const hasHiddenObjects = useMemo(
    () => groundingTracks.some((item) => Boolean(hiddenObjectIds[item.id])),
    [groundingTracks, hiddenObjectIds],
  );
  const groundedCamera = useMemo(
    () => resolveGroundedCameraName(grounding, primaryCamera),
    [grounding, primaryCamera],
  );
  const isGroundedCamera = selectedCamera === groundedCamera;
  const hasDenseTracking = Boolean(
    grounding?.dense_times_s && grounding.dense_times_s.length > 0,
  );
  const hasMultiCam = cameras.length >= 2;
  const isGridMode = viewMode === "grid" && hasMultiCam;
  const isPipMode = viewMode === "pip" && hasMultiCam && Boolean(pipCamera);

  return (
    <div className="studio-player-card" data-testid="studio-player">
      <div className="player-topbar">
        <div className="camera-tabs" role="tablist" aria-label={t("player.cameraViewsAria")}>
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
          <div
            className="view-mode-switcher"
            role="group"
            aria-label={t("player.viewModeAria")}
            data-testid="view-mode-switcher"
          >
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "single" ? "active" : ""}`}
              onClick={() => setViewMode("single")}
              data-testid="view-mode-single"
              title={t("player.singleTitle")}
            >
              <Maximize2 size={12} /> {t("player.single")}
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "pip" ? "active" : ""}`}
              disabled={!hasMultiCam}
              onClick={() => setViewMode("pip")}
              data-testid="view-mode-pip"
              title={hasMultiCam ? t("player.pipTitle") : t("player.requires2Cams")}
            >
              <PictureInPicture2 size={12} /> {t("player.pip")}
            </button>
            <button
              type="button"
              className={`view-mode-btn ${viewMode === "grid" ? "active" : ""}`}
              disabled={!hasMultiCam}
              onClick={() => setViewMode("grid")}
              data-testid="view-mode-grid"
              title={hasMultiCam ? t("player.gridTitle") : t("player.requires2Cams")}
            >
              <LayoutGrid size={12} /> {t("player.grid")}
            </button>
          </div>

          <button
            type="button"
            className={`pill-btn ${cutSkip ? "active" : ""}`}
            onClick={() => setCutSkip(!cutSkip)}
            title={t("player.cutSkipTitle")}
            data-testid="toggle-cut-skip"
          >
            <Scissors size={13} />
            <span>{t("player.cutSkip")}</span>
          </button>

          <button
            type="button"
            className={`pill-btn ${showGrounding && isGroundedCamera ? "active" : ""}`}
            onClick={() => setShowGrounding((v) => !v)}
            title={
              !isGroundedCamera && trackCount > 0
                ? t("player.boxesTitleMismatch", {
                    selected: selectedCamera,
                    grounded: groundedCamera,
                  })
                : trackCount === 0
                  ? t("player.boxesTitleEmpty")
                  : hasDenseTracking
                    ? t("player.boxesTitleDense")
                    : t("player.boxesTitleSparse")
            }
            data-testid="toggle-grounding"
          >
            {showGrounding ? <Eye size={13} /> : <EyeOff size={13} />}
            <span>{t("player.boxes2d", { count: trackCount })}</span>
            {trackCount > 0 && (
              <span
                className={`grounding-version-chip ${hasDenseTracking ? "dense" : "sparse"}`}
                data-testid="grounding-version-badge"
              >
                {hasDenseTracking ? t("player.v2Dense") : t("player.v1Sparse")}
              </span>
            )}
            {!isGroundedCamera && trackCount > 0 && (
              <span className="grounding-cam-mismatch-hint" data-testid="grounding-cam-mismatch">
                {t("player.camOnly", { cam: groundedCamera })}
              </span>
            )}
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
            {t("player.hdFrame")}
          </button>
        </div>
      </div>

      {groundingTracks.length > 0 && (
        <div
          className={`grounding-object-bar ${!showGrounding ? "dimmed" : ""}`}
          data-testid="grounding-object-bar"
        >
          <span className="grounding-object-bar-label">{t("player.objectsLabel")}</span>
          <div
            className="grounding-object-chips"
            role="group"
            aria-label={t("player.objectsGroupAria")}
          >
            {groundingTracks.map((item) => {
              const visible = !hiddenObjectIds[item.id];
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`object-toggle-chip ${visible ? "active" : "muted-off"}`}
                  aria-pressed={visible}
                  data-testid={`toggle-object-${item.id}`}
                  title={
                    visible
                      ? t("player.hideObjectTitle", { name: item.name })
                      : t("player.showObjectTitle", { name: item.name })
                  }
                  onClick={() =>
                    setHiddenObjectIds((prev) => ({
                      ...prev,
                      [item.id]: !prev[item.id],
                    }))
                  }
                >
                  <span className="object-chip-check">{visible ? "✓" : "✕"}</span>
                  <span>{item.name}</span>
                </button>
              );
            })}
            {hasHiddenObjects && (
              <button
                type="button"
                className="object-show-all-btn"
                data-testid="show-all-objects-btn"
                onClick={() => setHiddenObjectIds({})}
              >
                {t("player.showAll")}
              </button>
            )}
          </div>
        </div>
      )}

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
              onLoadedMetadata={(e) => {
                const vid = e.currentTarget;
                vid.playbackRate = playbackRate;
                const target = Math.max(
                  0,
                  Math.min(durationS, restoringTimeRef.current ?? currentTimeRef.current),
                );
                restoringTimeRef.current = null;
                if (target > 0.01 && Math.abs(vid.currentTime - target) > 0.05) {
                  try {
                    vid.currentTime = target;
                  } catch {
                    // Ignore seek errors
                  }
                }
                lastTrackedTimeRef.current = target;
                drawGroundingAtRef.current(target);
                if (shouldResumeAfterLoadRef.current) {
                  shouldResumeAfterLoadRef.current = false;
                  void Promise.resolve(vid.play?.()).catch(() => {
                    setPlaying(false);
                  });
                }
              }}
              onLoadedData={(e) => {
                drawGroundingAtRef.current(e.currentTarget.currentTime);
              }}
              onSeeked={(e) => {
                drawGroundingAtRef.current(e.currentTarget.currentTime);
              }}
              onTimeUpdate={handleVideoTimeUpdate}
              onPlay={() => {
                setPlaying(true);
                syncSecondaryVideos(videoRef.current?.currentTime ?? currentTimeRef.current, true, playbackRate);
              }}
              onPause={() => {
                setPlaying(false);
                const tNow = videoRef.current?.currentTime ?? currentTimeRef.current;
                syncSecondaryVideos(tNow, false, playbackRate);
                drawGroundingAtRef.current(tNow);
              }}
              onEnded={() => {
                setPlaying(false);
                syncSecondaryVideos(durationS, false, playbackRate);
                drawGroundingAtRef.current(durationS);
              }}
              onError={() => setVideoFailed(true)}
            />
          ) : (
            <img
              ref={hdImgRef}
              className="player-media"
              data-testid="primary-hd-frame"
              src={hdFrameSrc}
              alt={`${episodeId} ${selectedCamera} @ ${currentTime.toFixed(2)}s`}
              onLoad={() => drawGroundingAtRef.current(currentTimeRef.current)}
            />
          )}

          <canvas
            ref={canvasRef}
            className="grounding-canvas-readonly"
            data-testid="grounding-canvas"
            data-grounded-camera={groundedCamera}
            data-camera-matched={String(isGroundedCamera)}
            aria-label={t("player.groundingCanvasAria")}
          />

          {isGridMode && primaryCameraSpec && (
            <span className="cam-overlay-label">
              {t("player.primaryBadge", {
                name: primaryCameraSpec.name,
                role: primaryCameraSpec.role,
              })}
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
              title={t("player.clickPrimaryTitle", { name: cam.name })}
            >
              {renderSecondaryMedia(cam, true)}
            </div>
          ))}

        {/* Picture-in-Picture secondary overlay */}
        {isPipMode && pipCamera && (
          <div
            className="pip-overlay"
            data-testid={`pip-cam-${pipCamera.name}`}
            title={t("player.clickSwitchPrimaryTitle", { name: pipCamera.name })}
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
              {t("player.maskBanner", {
                name: activeMaskSegment.name || activeMaskSegment.label_id,
                side: activeMaskSegment.mask_side ?? "both",
              })}
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
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(-0.5)}
            title={t("player.stepBackHalfTitle")}
          >
            -0.5s
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(-1 / Math.max(1, fps))}
            title={t("player.prevFrameTitle")}
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
            <span>{playing ? t("player.pause") : t("player.play")}</span>
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(1 / Math.max(1, fps))}
            title={t("player.nextFrameTitle")}
          >
            <SkipForward size={14} />
          </button>
          <button
            type="button"
            className="icon-btn"
            onClick={() => stepTime(0.5)}
            title={t("player.stepFwdHalfTitle")}
          >
            +0.5s
          </button>
          <button
            type="button"
            className={`pill-btn ${loopSegment ? "active" : ""}`}
            disabled={!selectedSegment && !loopSegment}
            data-testid="toggle-segment-loop"
            title={selectedSegment ? t("player.loopTitle") : t("player.loopDisabledTitle")}
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
            <Repeat size={13} /> {t("player.loop")}
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
