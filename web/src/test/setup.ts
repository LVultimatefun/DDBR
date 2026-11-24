import "@testing-library/jest-dom/vitest";

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
}

if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() => {
    return {
      clearRect: () => {},
      strokeRect: () => {},
      fillRect: () => {},
      fillText: () => {},
      measureText: (t: string) => ({ width: t.length * 7 }),
      setLineDash: () => {},
      setTransform: () => {},
      beginPath: () => {},
      moveTo: () => {},
      lineTo: () => {},
      closePath: () => {},
      rect: () => {},
      stroke: () => {},
      fill: () => {},
      save: () => {},
      restore: () => {},
      strokeStyle: "",
      fillStyle: "",
      lineWidth: 1,
      globalAlpha: 1,
      font: "",
    };
  }) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

if (typeof HTMLMediaElement !== "undefined") {
  HTMLMediaElement.prototype.play = async () => {};
  HTMLMediaElement.prototype.pause = () => {};
}
