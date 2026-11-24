import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GroundingResult, GroundingTrack } from "../api/types";
import { AuthProvider } from "../stores/authStore";
import { makeResolvedEpisode } from "../test/fixtures";
import {
  computeContainRect,
  computeLabelPlacement,
  projectNormBoxToRect,
  resolveGroundedCameraName,
  sampleTrackBoxAt,
} from "../utils/groundingProjection";
import { StudioPlayer } from "./StudioPlayer";

interface MockCanvasSpy {
  clearRect: ReturnType<typeof vi.fn>;
  strokeRect: ReturnType<typeof vi.fn>;
  fillRect: ReturnType<typeof vi.fn>;
  fillText: ReturnType<typeof vi.fn>;
  setLineDash: ReturnType<typeof vi.fn>;
  setTransform: ReturnType<typeof vi.fn>;
  beginPath: ReturnType<typeof vi.fn>;
  moveTo: ReturnType<typeof vi.fn>;
  lineTo: ReturnType<typeof vi.fn>;
  stroke: ReturnType<typeof vi.fn>;
  save: ReturnType<typeof vi.fn>;
  restore: ReturnType<typeof vi.fn>;
  measureText: ReturnType<typeof vi.fn>;
}

function installCanvasSpy(): MockCanvasSpy {
  const spy: MockCanvasSpy = {
    clearRect: vi.fn(),
    strokeRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    setLineDash: vi.fn(),
    setTransform: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    measureText: vi.fn((t: string) => ({ width: t.length * 7 })),
  };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
    () =>
      ({
        ...spy,
        closePath: vi.fn(),
        rect: vi.fn(),
        fill: vi.fn(),
        strokeStyle: "",
        fillStyle: "",
        lineWidth: 1,
        globalAlpha: 1,
        shadowColor: "",
        shadowBlur: 0,
        font: "",
      }) as unknown as CanvasRenderingContext2D,
  );
  return spy;
}

function setElementSize(el: HTMLElement, width: number, height: number) {
  Object.defineProperty(el, "clientWidth", { value: width, configurable: true });
  Object.defineProperty(el, "clientHeight", { value: height, configurable: true });
  vi.spyOn(el, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    right: width,
    bottom: height,
    width,
    height,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
}

function makeDenseGrounding(): GroundingResult {
  // Dense 5Hz track over [0.0, 2.0]:
  // - block moves linearly from x=[100..200] at t=0.0 to x=[500..600] at t=0.8 (VLM anchors at 0.0 and 2.0)
  // - occluded during t=[1.0, 1.4] (flag=0, box=[0,0,0,0])
  // - reappears at t=1.6..2.0 at x=[700..800]
  const denseTimes = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
  const blockBoxes: Array<[number, number, number, number]> = [
    [100, 200, 200, 400], // t=0.0 anchor
    [200, 200, 300, 400], // t=0.2 tracked
    [300, 200, 400, 400], // t=0.4 tracked
    [400, 200, 500, 400], // t=0.6 tracked
    [500, 200, 600, 400], // t=0.8 tracked
    [0, 0, 0, 0], // t=1.0 occluded
    [0, 0, 0, 0], // t=1.2 occluded
    [0, 0, 0, 0], // t=1.4 occluded
    [700, 200, 800, 400], // t=1.6 tracked
    [750, 200, 850, 400], // t=1.8 tracked
    [800, 200, 900, 400], // t=2.0 anchor
  ];
  const blockFlags = [2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 2];
  const blockConf = [100, 94, 90, 86, 82, 0, 0, 0, 84, 92, 100];

  const staticBoxes: Array<[number, number, number, number]> = denseTimes.map(() => [
    600, 100, 900, 500,
  ]);
  const staticFlags = denseTimes.map(() => 3);
  const staticConf = denseTimes.map(() => 100);

  return {
    schema_version: 2,
    camera_name: "head_rgbd",
    ref_size: [640, 480],
    keyframes_s: [0.0, 1.2, 2.0],
    track_fps: 5.0,
    dense_times_s: denseTimes,
    camera_motion_detected: false,
    object_tracks_2d: {
      block: {
        object_id: "block",
        name: "Target Block",
        tag_pos: "top",
        is_static_container: false,
        tracking_mode: "bidir_flow_ncc",
        keyframes: [
          {
            row: 0,
            time_s: 0.0,
            visible: true,
            bbox_norm_1000: [100, 200, 200, 400],
            bbox_px: [64, 96, 128, 192],
            state: "grasping",
            status_kind: "good",
          },
          {
            row: 36,
            time_s: 1.2,
            visible: false,
            bbox_norm_1000: [0, 0, 0, 0],
            bbox_px: [0, 0, 0, 0],
            state: "",
            status_kind: "neutral",
          },
          {
            row: 60,
            time_s: 2.0,
            visible: true,
            bbox_norm_1000: [800, 200, 900, 400],
            bbox_px: [512, 96, 576, 192],
            state: "placed",
            status_kind: "warn",
          },
        ],
        dense_boxes_norm_1000: blockBoxes,
        dense_flags: blockFlags,
        dense_conf_pct: blockConf,
      },
      target_bin: {
        object_id: "target_bin",
        name: "Target Bin",
        tag_pos: "bottom",
        is_static_container: true,
        tracking_mode: "static_locked",
        keyframes: [
          {
            row: 0,
            time_s: 0.0,
            visible: true,
            bbox_norm_1000: [600, 100, 900, 500],
            bbox_px: [384, 48, 576, 240],
            state: "stationary",
            status_kind: "neutral",
          },
        ],
        dense_boxes_norm_1000: staticBoxes,
        dense_flags: staticFlags,
        dense_conf_pct: staticConf,
      },
    },
  };
}

describe("groundingProjection pure math & sampling", () => {
  it("computes exact object-fit: contain rects for 16:9, 4:3 pillarbox, stereo, 21:9 letterbox, and 2x2 square grid", () => {
    // 16:9 in 16:9 -> exact full rect
    expect(computeContainRect(800, 450, 1280, 720)).toEqual({
      x: 0,
      y: 0,
      width: 800,
      height: 450,
    });

    // 4:3 (640x480) in 16:9 (800x450) -> pillarboxed left/right by 100px
    const pillar = computeContainRect(800, 450, 640, 480);
    expect(pillar.x).toBeCloseTo(100, 4);
    expect(pillar.y).toBeCloseTo(0, 4);
    expect(pillar.width).toBeCloseTo(600, 4);
    expect(pillar.height).toBeCloseTo(450, 4);

    // Stereo 1600x1296 inside 800x450 -> pillarboxed
    const stereo = computeContainRect(800, 450, 1600, 1296);
    expect(stereo.y).toBe(0);
    expect(stereo.height).toBeCloseTo(450, 4);
    expect(stereo.width).toBeCloseTo(450 * (1600 / 1296), 4);
    expect(stereo.x).toBeCloseTo((800 - stereo.width) / 2, 4);

    // 21:9 (2100x900) in 16:9 (800x450) -> letterboxed top/bottom
    const letter = computeContainRect(800, 450, 2100, 900);
    expect(letter.x).toBe(0);
    expect(letter.width).toBeCloseTo(800, 4);
    expect(letter.height).toBeCloseTo(800 / (21 / 9), 4);
    expect(letter.y).toBeCloseTo((450 - letter.height) / 2, 4);

    // 16:9 inside square 400x400 grid pane -> letterboxed top/bottom by 87.5px
    const gridSq = computeContainRect(400, 400, 1280, 720);
    expect(gridSq.x).toBe(0);
    expect(gridSq.width).toBe(400);
    expect(gridSq.height).toBeCloseTo(225, 4);
    expect(gridSq.y).toBeCloseTo(87.5, 4);

    // Missing media dims fall back cleanly to container rect
    expect(computeContainRect(640, 360, 0, 0)).toEqual({
      x: 0,
      y: 0,
      width: 640,
      height: 360,
    });
  });

  it("projects normalized [0, 1000] boxes and clamps label pills strictly inside contentRect", () => {
    const contentRect = computeContainRect(800, 450, 640, 480); // x=100, y=0, w=600, h=450
    const box = projectNormBoxToRect([100, 0, 200, 200], contentRect);
    expect(box.x).toBeCloseTo(160, 4); // 100 + 0.1 * 600
    expect(box.y).toBeCloseTo(0, 4);
    expect(box.width).toBeCloseTo(60, 4);
    expect(box.height).toBeCloseTo(90, 4);

    // Top label when box.y == 0 flips inside top so pill.y >= contentRect.y
    const topLabel = computeLabelPlacement(box, contentRect, 100, 19, "top");
    expect(topLabel.x).toBeGreaterThanOrEqual(contentRect.x);
    expect(topLabel.y).toBeGreaterThanOrEqual(contentRect.y);
    expect(topLabel.y + 19).toBeLessThanOrEqual(contentRect.y + contentRect.height);

    // Label on box at right edge clamps X so it doesn't bleed into right pillarbox bar
    const rightBox = projectNormBoxToRect([950, 960, 1000, 1000], contentRect);
    const rightBottomLabel = computeLabelPlacement(rightBox, contentRect, 140, 19, "bottom");
    expect(rightBottomLabel.x + 140).toBeLessThanOrEqual(contentRect.x + contentRect.width);
    expect(rightBottomLabel.y + 19).toBeLessThanOrEqual(contentRect.y + contentRect.height);
  });

  it("samples dense tracks via O(log N) binary search, interpolates smoothly, hides occlusion flag=0, and preserves provenance", () => {
    const g = makeDenseGrounding();
    const block = g.object_tracks_2d.block;
    const bin = g.object_tracks_2d.target_bin;

    // Exact VLM anchor at t=0.0 -> provenance = vlm_anchor
    const atAnchor = sampleTrackBoxAt(g, block, 0.0);
    expect(atAnchor).not.toBeNull();
    expect(atAnchor!.provenance).toBe("vlm_anchor");
    expect(atAnchor!.bbox).toEqual([100, 200, 200, 400]);
    expect(atAnchor!.state).toBe("grasping");
    expect(atAnchor!.statusKind).toBe("good");
    expect(atAnchor!.confPct).toBe(100);

    // Intermediate non-anchor midpoint t=0.3 between t=0.2 ([200..300], conf=94) and t=0.4 ([300..400], conf=90)
    const atMid = sampleTrackBoxAt(g, block, 0.3);
    expect(atMid).not.toBeNull();
    expect(atMid!.provenance).toBe("tracked");
    expect(atMid!.bbox[0]).toBeCloseTo(250, 4);
    expect(atMid!.bbox[2]).toBeCloseTo(350, 4);
    expect(atMid!.confPct).toBe(92);
    expect(atMid!.state).toBe("grasping");

    // Occluded frames (flag=0) at t=1.0, t=1.1, t=1.2 -> hidden (null)
    expect(sampleTrackBoxAt(g, block, 1.0)).toBeNull();
    expect(sampleTrackBoxAt(g, block, 1.1)).toBeNull();
    expect(sampleTrackBoxAt(g, block, 1.2)).toBeNull();

    // Boundary crossing flag=1 (t=0.8) -> flag=0 (t=1.0): snaps to nearest
    expect(sampleTrackBoxAt(g, block, 0.85)?.bbox).toEqual([500, 200, 600, 400]);
    expect(sampleTrackBoxAt(g, block, 0.95)).toBeNull();

    // After occlusion at t=1.7 -> interpolates between t=1.6 and t=1.8, pulls nearest visible VLM anchor (t=2.0, skipping invisible t=1.2!)
    const afterOcc = sampleTrackBoxAt(g, block, 1.7);
    expect(afterOcc).not.toBeNull();
    expect(afterOcc!.provenance).toBe("tracked");
    expect(afterOcc!.bbox[0]).toBeCloseTo(725, 4);
    expect(afterOcc!.state).toBe("placed");
    expect(afterOcc!.statusKind).toBe("warn");

    // Static locked container returns static_locked provenance
    const staticSample = sampleTrackBoxAt(g, bin, 1.1);
    expect(staticSample?.provenance).toBe("static_locked");
    expect(staticSample?.bbox).toEqual([600, 100, 900, 500]);
  });

  it("falls back smoothly to smoothstep interpolation (jump <= 150) & occlusion hiding on legacy v1 sparse grounding artifacts", () => {
    const legacyTrack: GroundingTrack = {
      object_id: "block",
      name: "Target Block",
      tag_pos: "top",
      is_static_container: false,
      keyframes: [
        {
          row: 0,
          time_s: 0.0,
          visible: true,
          bbox_norm_1000: [100, 100, 200, 200], // center (150, 150)
          bbox_px: [128, 72, 256, 144],
          state: "start",
          status_kind: "neutral",
        },
        {
          row: 60,
          time_s: 2.0,
          visible: true,
          bbox_norm_1000: [200, 100, 300, 200], // center (250, 150) -> jump = 100 <= 150
          bbox_px: [256, 72, 384, 144],
          state: "moving",
          status_kind: "good",
        },
        {
          row: 120,
          time_s: 4.0,
          visible: false,
          bbox_norm_1000: [0, 0, 0, 0],
          bbox_px: [0, 0, 0, 0],
          state: "",
          status_kind: "neutral",
        },
      ],
    };
    const legacyGrounding: GroundingResult = {
      schema_version: 1,
      ref_size: [1280, 720],
      keyframes_s: [0.0, 2.0, 4.0],
      object_tracks_2d: { block: legacyTrack },
    };

    // Midpoint t=1.0 -> smoothstep(0.5) = 0.5 -> x1 = 150
    const mid = sampleTrackBoxAt(legacyGrounding, legacyTrack, 1.0);
    expect(mid).not.toBeNull();
    expect(mid!.bbox[0]).toBeCloseTo(150, 4);
    expect(mid!.provenance).toBe("tracked");

    // Quarter t=0.5 -> u=0.25 -> smoothstep(0.25) = 0.15625 -> x1 = 100 + 100*0.15625 = 115.625
    const quarter = sampleTrackBoxAt(legacyGrounding, legacyTrack, 0.5);
    expect(quarter!.bbox[0]).toBeCloseTo(115.625, 4);

    // Near invisible anchor t=3.5 (closer to t=4.0 invisible than t=2.0 visible) -> hidden (null)
    expect(sampleTrackBoxAt(legacyGrounding, legacyTrack, 3.5)).toBeNull();
    // Near visible anchor t=2.4 (closer to t=2.0 visible) -> visible
    expect(sampleTrackBoxAt(legacyGrounding, legacyTrack, 2.4)?.bbox).toEqual([200, 100, 300, 200]);

    // Legacy grounding camera_name absent -> resolves to primaryCamera
    expect(resolveGroundedCameraName(legacyGrounding, "head_rgbd")).toBe("head_rgbd");
  });

  it("enforces legacy v1 jump guard: smoothstep for <=150 (inclusive) and clean snap-to-nearest for >150 horizontal/diagonal jumps", () => {
    const makeSparsePair = (
      b0: [number, number, number, number],
      b1: [number, number, number, number],
    ): { grounding: GroundingResult; track: GroundingTrack } => {
      const track: GroundingTrack = {
        object_id: "block",
        name: "Target Block",
        tag_pos: "top",
        is_static_container: false,
        keyframes: [
          {
            row: 0,
            time_s: 0.0,
            visible: true,
            bbox_norm_1000: b0,
            bbox_px: [0, 0, 10, 10],
            state: "A",
            status_kind: "neutral",
          },
          {
            row: 60,
            time_s: 2.0,
            visible: true,
            bbox_norm_1000: b1,
            bbox_px: [0, 0, 10, 10],
            state: "B",
            status_kind: "good",
          },
        ],
      };
      return {
        grounding: {
          schema_version: 1,
          ref_size: [1280, 720],
          keyframes_s: [0.0, 2.0],
          object_tracks_2d: { block: track },
        },
        track,
      };
    };

    // Exact 150.0 horizontal boundary -> preserves smoothstep
    const exact150 = makeSparsePair([100, 100, 200, 200], [250, 100, 350, 200]);
    const exactMid = sampleTrackBoxAt(exact150.grounding, exact150.track, 1.0);
    expect(exactMid?.provenance).toBe("tracked");
    expect(exactMid?.bbox[0]).toBeCloseTo(175, 4);

    // Exact 150.0 diagonal (3-4-5 triangle: dx=90, dy=120 -> hypot=150) -> preserves smoothstep
    const exactDiag = makeSparsePair([100, 100, 200, 200], [190, 220, 290, 320]);
    const exactDiagQuarter = sampleTrackBoxAt(exactDiag.grounding, exactDiag.track, 0.5);
    expect(exactDiagQuarter?.provenance).toBe("tracked");
    expect(exactDiagQuarter?.bbox[0]).toBeCloseTo(100 + 90 * 0.15625, 4);
    expect(exactDiagQuarter?.bbox[1]).toBeCloseTo(100 + 120 * 0.15625, 4);

    // Horizontal jump 400 > 150 (dt=2.0 <= 4.5s) -> disables smoothstep, snaps cleanly to nearest
    const bigHoriz = makeSparsePair([100, 100, 200, 200], [500, 100, 600, 200]);
    const snapLeft = sampleTrackBoxAt(bigHoriz.grounding, bigHoriz.track, 0.6);
    expect(snapLeft?.provenance).toBe("vlm_anchor");
    expect(snapLeft?.bbox).toEqual([100, 100, 200, 200]);
    const snapRight = sampleTrackBoxAt(bigHoriz.grounding, bigHoriz.track, 1.4);
    expect(snapRight?.provenance).toBe("vlm_anchor");
    expect(snapRight?.bbox).toEqual([500, 100, 600, 200]);

    // Diagonal jump dx=110, dy=110 (hypot ~= 155.56 > 150) -> snaps cleanly to nearest
    const bigDiag = makeSparsePair([100, 100, 200, 200], [210, 210, 310, 310]);
    const diagLeft = sampleTrackBoxAt(bigDiag.grounding, bigDiag.track, 0.6);
    expect(diagLeft?.provenance).toBe("vlm_anchor");
    expect(diagLeft?.bbox).toEqual([100, 100, 200, 200]);
    const diagRight = sampleTrackBoxAt(bigDiag.grounding, bigDiag.track, 1.4);
    expect(diagRight?.provenance).toBe("vlm_anchor");
    expect(diagRight?.bbox).toEqual([210, 210, 310, 310]);
  });

  it("applies >150 sparse jump guard when schema_version=2 lacks dense_times_s", () => {
    const track: GroundingTrack = {
      object_id: "block",
      name: "Target Block",
      tag_pos: "top",
      is_static_container: false,
      keyframes: [
        {
          row: 0,
          time_s: 0.0,
          visible: true,
          bbox_norm_1000: [100, 100, 200, 200],
          bbox_px: [0, 0, 10, 10],
          state: "A",
          status_kind: "neutral",
        },
        {
          row: 60,
          time_s: 2.0,
          visible: true,
          bbox_norm_1000: [500, 100, 600, 200], // jump = 400 > 150
          bbox_px: [0, 0, 10, 10],
          state: "B",
          status_kind: "good",
        },
      ],
    };
    const v2WithoutDense: GroundingResult = {
      schema_version: 2,
      ref_size: [1280, 720],
      keyframes_s: [0.0, 2.0],
      object_tracks_2d: { block: track },
    };

    const snapLeft = sampleTrackBoxAt(v2WithoutDense, track, 0.6);
    expect(snapLeft?.provenance).toBe("vlm_anchor");
    expect(snapLeft?.bbox).toEqual([100, 100, 200, 200]);

    const snapRight = sampleTrackBoxAt(v2WithoutDense, track, 1.4);
    expect(snapRight?.provenance).toBe("vlm_anchor");
    expect(snapRight?.bbox).toEqual([500, 100, 600, 200]);
  });

  it("preserves static_locked provenance when staticMode sparse track snaps across >150 jump", () => {
    const makeStaticTrack = (
      overrides: Pick<GroundingTrack, "is_static_container" | "tracking_mode">,
    ): GroundingTrack => ({
      object_id: "target_bin",
      name: "Target Bin",
      tag_pos: "top",
      is_static_container: overrides.is_static_container,
      tracking_mode: overrides.tracking_mode,
      keyframes: [
        {
          row: 0,
          time_s: 0.0,
          visible: true,
          bbox_norm_1000: [100, 100, 250, 300],
          bbox_px: [0, 0, 10, 10],
          state: "idle",
          status_kind: "neutral",
        },
        {
          row: 60,
          time_s: 2.0,
          visible: true,
          bbox_norm_1000: [500, 100, 650, 300], // jump = 400 > 150
          bbox_px: [0, 0, 10, 10],
          state: "relocated",
          status_kind: "neutral",
        },
      ],
    });

    const containerTrack = makeStaticTrack({ is_static_container: true, tracking_mode: undefined });
    const modeTrack = makeStaticTrack({ is_static_container: false, tracking_mode: "static_locked" });
    const grounding: GroundingResult = {
      schema_version: 1,
      ref_size: [1280, 720],
      keyframes_s: [0.0, 2.0],
      object_tracks_2d: { target_bin: containerTrack, mode_bin: modeTrack },
    };

    const leftContainer = sampleTrackBoxAt(grounding, containerTrack, 0.6);
    expect(leftContainer?.provenance).toBe("static_locked");
    expect(leftContainer?.bbox).toEqual([100, 100, 250, 300]);

    const rightMode = sampleTrackBoxAt(grounding, modeTrack, 1.4);
    expect(rightMode?.provenance).toBe("static_locked");
    expect(rightMode?.bbox).toEqual([500, 100, 650, 300]);
  });
});

describe("StudioPlayer frame-level 2D grounding integration", () => {
  const originalDpr = window.devicePixelRatio;
  const originalRO = globalThis.ResizeObserver;

  beforeEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(window, "devicePixelRatio", { value: 2, configurable: true });
  });

  afterEach(() => {
    Object.defineProperty(window, "devicePixelRatio", { value: originalDpr, configurable: true });
    globalThis.ResizeObserver = originalRO;
  });

  it("projects bboxes strictly into object-fit: contain video rect across Single, PiP, and 2x2 Grid + HiDPI scaling", () => {
    const spy = installCanvasSpy();
    const ep = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });
    // Set primary camera natural spec to 4:3 (640x480)
    ep.meta.cameras[0].width = 640;
    ep.meta.cameras[0].height = 480;

    render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.3}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const pane = screen.getByTestId("primary-cam-pane");
    const canvas = screen.getByTestId("grounding-canvas") as HTMLCanvasElement;
    const video = screen.getByTestId("primary-video") as HTMLVideoElement;

    // 16:9 pane (800x450) displaying 4:3 video (640x480) -> contentRect = { x:100, y:0, w:600, h:450 }
    setElementSize(pane, 800, 450);
    Object.defineProperty(video, "videoWidth", { value: 640, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 480, configurable: true });
    Object.defineProperty(video, "currentTime", { value: 0.3, writable: true, configurable: true });

    spy.strokeRect.mockClear();
    spy.setTransform.mockClear();
    fireEvent.loadedMetadata(video);

    // HiDPI backing buffer = 1600x900 (dpr=2), transform scaled by 2
    expect(canvas.width).toBe(1600);
    expect(canvas.height).toBe(900);
    expect(spy.setTransform).toHaveBeenCalledWith(2, 0, 0, 2, 0, 0);
    expect(canvas.dataset.contentRect).toBe("100.0,0.0,600.0,450.0");

    // Block at t=0.3 has interpolated bbox [250, 200, 350, 400]
    // Projected into { x:100, y:0, w:600, h:450 } -> x = 100 + 0.25*600 = 250, y = 0.2*450 = 90, w = 60, h = 90
    expect(spy.strokeRect).toHaveBeenCalledWith(250, 90, 60, 90);

    // Switch to PiP & 2x2 Grid with narrower square pane (400x400) -> 4:3 video letterboxed { x:0, y:50, w:400, h:300 }
    fireEvent.click(screen.getByTestId("view-mode-pip"));
    expect(screen.getByTestId("primary-cam-pane")).toContainElement(canvas);

    setElementSize(pane, 400, 400);
    spy.strokeRect.mockClear();
    fireEvent.click(screen.getByTestId("view-mode-grid"));

    // Block [250, 200, 350, 400] projected into { x:0, y:50, w:400, h:300 } -> x = 100, y = 50 + 60 = 110, w = 40, h = 60
    expect(canvas.dataset.contentRect).toBe("0.0,50.0,400.0,300.0");
    expect(spy.strokeRect).toHaveBeenCalledWith(100, 110, 40, 60);
  });

  it("clears canvas & hides boxes on camera mismatch (F4) and hides occluded tracks (flag=0)", () => {
    const spy = installCanvasSpy();
    const ep = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });

    const { rerender } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const canvas = screen.getByTestId("grounding-canvas");
    expect(canvas).toHaveAttribute("data-camera-matched", "true");
    expect(spy.strokeRect).toHaveBeenCalledTimes(2); // block + target_bin

    // Switch to secondary camera `torso_rgbd` -> camera mismatch clears canvas and draws 0 boxes
    spy.strokeRect.mockClear();
    spy.clearRect.mockClear();
    fireEvent.click(screen.getByRole("tab", { name: /torso_rgbd/ }));
    expect(canvas).toHaveAttribute("data-camera-matched", "false");
    expect(screen.getByTestId("grounding-cam-mismatch")).toHaveTextContent("head_rgbd only");
    expect(spy.clearRect).toHaveBeenCalled();
    expect(spy.strokeRect).not.toHaveBeenCalled();

    // Switch back to grounded camera `head_rgbd` -> boxes return immediately
    spy.strokeRect.mockClear();
    fireEvent.click(screen.getByRole("tab", { name: /head_rgbd/ }));
    expect(canvas).toHaveAttribute("data-camera-matched", "true");
    expect(spy.strokeRect).toHaveBeenCalledTimes(2);

    // Move to occluded timestamp t=1.2 (block flag=0, target_bin flag=3) -> ONLY static target_bin drawn (1 box!)
    spy.strokeRect.mockClear();
    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={1.2}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );
    expect(spy.strokeRect).toHaveBeenCalledTimes(1);
  });

  it("drives 30fps redraws via requestVideoFrameCallback during playback and falls back to requestAnimationFrame", () => {
    const spy = installCanvasSpy();
    const ep = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });
    ep.meta.cameras[0].width = 1000;
    ep.meta.cameras[0].height = 1000;

    const { unmount } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const pane = screen.getByTestId("primary-cam-pane");
    setElementSize(pane, 1000, 1000);
    const video = screen.getByTestId("primary-video") as HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
      cancelVideoFrameCallback?: (id: number) => void;
    };
    Object.defineProperty(video, "videoWidth", { value: 1000, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 1000, configurable: true });

    let nextId = 1;
    const queuedCallbacks = new Map<number, (now: number, meta: { mediaTime: number }) => void>();
    const cancelRvfc = vi.fn((id: number) => {
      queuedCallbacks.delete(id);
    });
    video.requestVideoFrameCallback = vi.fn((cb) => {
      const id = nextId++;
      queuedCallbacks.set(id, cb);
      return id;
    });
    video.cancelVideoFrameCallback = cancelRvfc;

    // Trigger playback -> registers requestVideoFrameCallback
    fireEvent.play(video);
    expect(video.requestVideoFrameCallback).toHaveBeenCalledTimes(1);

    // Fire frame callback at intermediate t=0.5 (without any React currentTime update!)
    // Block at t=0.5 interpolates between t=0.4 ([300,200,400,400]) and t=0.6 ([400,200,500,400]) -> [350,200,450,400]
    spy.strokeRect.mockClear();
    const firstEntry = [...queuedCallbacks.entries()][0];
    queuedCallbacks.delete(firstEntry[0]);
    act(() => {
      firstEntry[1](16.6, { mediaTime: 0.5 });
    });
    expect(spy.strokeRect).toHaveBeenCalledWith(350, 200, 100, 200);
    // Re-armed for next video frame
    expect(video.requestVideoFrameCallback).toHaveBeenCalledTimes(2);

    // Pause cancels active rVFC handle cleanly
    fireEvent.pause(video);
    expect(cancelRvfc).toHaveBeenCalled();
    unmount();

    // Verify requestAnimationFrame fallback when requestVideoFrameCallback is absent
    const rafQueue = new Map<number, FrameRequestCallback>();
    let rafSeq = 10;
    const rafSpy = vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      const id = rafSeq++;
      rafQueue.set(id, cb);
      return id;
    });
    const cancelRafSpy = vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      rafQueue.delete(id);
    });

    const { unmount: unmountFallback } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const pane2 = screen.getByTestId("primary-cam-pane");
    setElementSize(pane2, 1000, 1000);
    const vid2 = screen.getByTestId("primary-video") as HTMLVideoElement;
    Object.defineProperty(vid2, "videoWidth", { value: 1000, configurable: true });
    Object.defineProperty(vid2, "videoHeight", { value: 1000, configurable: true });
    Object.defineProperty(vid2, "currentTime", { value: 0.7, writable: true, configurable: true });

    fireEvent.play(vid2);
    expect(rafSpy).toHaveBeenCalled();

    spy.strokeRect.mockClear();
    const rafEntry = [...rafQueue.entries()][0];
    rafQueue.delete(rafEntry[0]);
    act(() => {
      rafEntry[1](33.3);
    });
    // Block at t=0.7 -> [450, 200, 550, 400]
    expect(spy.strokeRect).toHaveBeenCalledWith(450, 200, 100, 200);

    fireEvent.pause(vid2);
    expect(cancelRafSpy).toHaveBeenCalled();
    unmountFallback();
  });

  it("redraws immediately on ResizeObserver layout changes, video seeked, and HD image load", () => {
    const spy = installCanvasSpy();
    const roCallbacks: ResizeObserverCallback[] = [];
    const observedEls: Element[] = [];
    class CapturingResizeObserver {
      cb: ResizeObserverCallback;
      constructor(cb: ResizeObserverCallback) {
        this.cb = cb;
        roCallbacks.push(cb);
      }
      observe(el: Element) {
        observedEls.push(el);
      }
      unobserve() {}
      disconnect() {}
    }
    globalThis.ResizeObserver = CapturingResizeObserver as unknown as typeof ResizeObserver;

    const ep = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });
    ep.meta.cameras[0].width = 640;
    ep.meta.cameras[0].height = 480;

    render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const pane = screen.getByTestId("primary-cam-pane");
    const video = screen.getByTestId("primary-video") as HTMLVideoElement;
    const canvas = screen.getByTestId("grounding-canvas") as HTMLCanvasElement;

    expect(observedEls).toContain(pane);
    expect(observedEls).toContain(video);

    // Trigger ResizeObserver after changing pane to 800x450 -> 4:3 pillarbox (x=100, y=0, w=600, h=450)
    setElementSize(pane, 800, 450);
    Object.defineProperty(video, "videoWidth", { value: 640, configurable: true });
    Object.defineProperty(video, "videoHeight", { value: 480, configurable: true });
    spy.strokeRect.mockClear();
    act(() => {
      for (const cb of roCallbacks) cb([], {} as ResizeObserver);
    });
    expect(canvas.dataset.contentRect).toBe("100.0,0.0,600.0,450.0");
    // Block at t=0.0 [100, 200, 200, 400] -> x=100+60=160, y=90, w=60, h=90
    expect(spy.strokeRect).toHaveBeenCalledWith(160, 90, 60, 90);

    // Resize pane to 400x400 -> 4:3 letterbox (x=0, y=50, w=400, h=300)
    setElementSize(pane, 400, 400);
    spy.strokeRect.mockClear();
    act(() => {
      for (const cb of roCallbacks) cb([], {} as ResizeObserver);
    });
    expect(canvas.dataset.contentRect).toBe("0.0,50.0,400.0,300.0");
    expect(spy.strokeRect).toHaveBeenCalledWith(40, 110, 40, 60);

    // Seek video to t=0.6 -> seeked event redraws block at [400, 200, 500, 400] -> x=160, y=110, w=40, h=60
    Object.defineProperty(video, "currentTime", { value: 0.6, writable: true, configurable: true });
    spy.strokeRect.mockClear();
    fireEvent.seeked(video);
    expect(spy.strokeRect).toHaveBeenCalledWith(160, 110, 40, 60);

    // Toggle HD Frame mode -> img load redraws using img.naturalWidth/naturalHeight (2100x900 letterbox inside 800x450)
    fireEvent.click(screen.getByTestId("toggle-hd-frame"));
    const hdImg = screen.getByTestId("primary-hd-frame") as HTMLImageElement;
    setElementSize(pane, 800, 450);
    Object.defineProperty(hdImg, "naturalWidth", { value: 2100, configurable: true });
    Object.defineProperty(hdImg, "naturalHeight", { value: 900, configurable: true });
    spy.strokeRect.mockClear();
    fireEvent.load(hdImg);
    expect(canvas.dataset.contentRect).toBe("0.0,53.6,800.0,342.9");
    expect(spy.strokeRect).toHaveBeenCalled();
  });

  it("displays v2 Dense vs v1 Sparse indicator on toggle-grounding when trackCount > 0 and omits it when trackCount === 0", () => {
    installCanvasSpy();
    const epDense = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });
    const epSparse = makeResolvedEpisode(); // schema_version=1, 1 track, no dense_times_s
    const epEmpty = makeResolvedEpisode({
      grounding: {
        schema_version: 1,
        ref_size: [1280, 720],
        keyframes_s: [],
        object_tracks_2d: {},
      },
    });

    const { rerender } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={epDense.episode_id}
          cameras={epDense.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={epDense.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    const toggleBtn = screen.getByTestId("toggle-grounding");
    expect(toggleBtn).toHaveTextContent("2D Boxes (2)");
    const badge = screen.getByTestId("grounding-version-badge");
    expect(badge).toHaveTextContent("v2 Dense");
    expect(badge.className).toContain("dense");

    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={epSparse.episode_id}
          cameras={epSparse.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={epSparse.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    expect(screen.getByTestId("toggle-grounding")).toHaveTextContent("2D Boxes (1)");
    const sparseBadge = screen.getByTestId("grounding-version-badge");
    expect(sparseBadge).toHaveTextContent("v1 Sparse");
    expect(sparseBadge.className).toContain("sparse");

    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={epEmpty.episode_id}
          cameras={epEmpty.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={epEmpty.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    expect(screen.getByTestId("toggle-grounding")).toHaveTextContent("2D Boxes (0)");
    expect(screen.queryByTestId("grounding-version-badge")).toBeNull();
    expect(screen.queryByTestId("grounding-object-bar")).toBeNull();
  });

  it("renders per-object grounding toggle chips and hides/restores individual tracks on canvas without mutating grounding data", () => {
    const spy = installCanvasSpy();
    const ep = makeResolvedEpisode({
      grounding: makeDenseGrounding(),
    });
    const originalKeys = Object.keys(ep.grounding.object_tracks_2d);

    const { rerender } = render(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId={ep.episode_id}
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );

    // Object filter chips appear for both tracks and start active
    expect(screen.getByTestId("grounding-object-bar")).toBeInTheDocument();
    const blockChip = screen.getByTestId("toggle-object-block");
    const binChip = screen.getByTestId("toggle-object-target_bin");
    expect(blockChip).toHaveTextContent("Target Block");
    expect(blockChip).toHaveAttribute("aria-pressed", "true");
    expect(binChip).toHaveTextContent("Target Bin");
    expect(binChip).toHaveAttribute("aria-pressed", "true");
    expect(spy.strokeRect).toHaveBeenCalledTimes(2);

    // Toggle Target Block off -> only Target Bin is drawn (1 strokeRect)
    spy.strokeRect.mockClear();
    fireEvent.click(blockChip);
    expect(blockChip).toHaveAttribute("aria-pressed", "false");
    expect(spy.strokeRect).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("show-all-objects-btn")).toBeInTheDocument();

    // Backend grounding object map was not mutated
    expect(Object.keys(ep.grounding.object_tracks_2d)).toEqual(originalKeys);

    // Click Show All -> both tracks drawn again
    spy.strokeRect.mockClear();
    fireEvent.click(screen.getByTestId("show-all-objects-btn"));
    expect(blockChip).toHaveAttribute("aria-pressed", "true");
    expect(spy.strokeRect).toHaveBeenCalledTimes(2);
    expect(screen.queryByTestId("show-all-objects-btn")).toBeNull();

    // Hide block again, then switch episodeId -> filter resets cleanly
    fireEvent.click(blockChip);
    expect(blockChip).toHaveAttribute("aria-pressed", "false");
    rerender(
      <AuthProvider initialAuth={{ token: "tok", username: "rev1", role: "reviewer" }}>
        <StudioPlayer
          episodeId="demo_ds/ep_002"
          cameras={ep.meta.cameras}
          durationS={10.0}
          fps={30}
          grounding={ep.grounding}
          currentTime={0.0}
          onTimeUpdate={vi.fn()}
        />
      </AuthProvider>,
    );
    expect(screen.getByTestId("toggle-object-block")).toHaveAttribute("aria-pressed", "true");
  });
});
