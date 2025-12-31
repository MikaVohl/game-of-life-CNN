import { useState, useEffect, useRef, useMemo } from "react";
import type { ChangeEvent, PointerEvent } from "react";
import "./App.css";
import { predictWithCnn, simulateSteps, type Grid } from "./lifeModel";

const SIZE = 32;
const STEPS = 5;
const SIM_DELAY = 650;

function LifeGrid({
  grid,
  cell = 12,
  interactive = false,
  dimmed = false,
  onDown,
  onEnter,
  onUp,
}: {
  grid: Grid;
  cell?: number;
  interactive?: boolean;
  dimmed?: boolean;
  onDown?: (r: number, c: number) => void;
  onEnter?: (r: number, c: number) => void;
  onUp?: () => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const activePointer = useRef<number | null>(null);
  const lastCell = useRef<string | null>(null);
  const strokeStarted = useRef(false);
  const activeTouch = useRef(false);
  const touchStarted = useRef(false);
  const lastTouchCell = useRef<string | null>(null);

  const resolveCell = (clientX: number, clientY: number) => {
    const container = containerRef.current;
    if (!container) return null;
    const hit = document.elementFromPoint(clientX, clientY) as HTMLElement | null;
    if (!hit) return null;
    const cellEl = hit.closest?.("[data-life-cell='1']") as HTMLElement | null;
    if (!cellEl || !container.contains(cellEl)) return null;
    const r = Number(cellEl.dataset.r);
    const c = Number(cellEl.dataset.c);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return null;
    return { r, c, key: `${r}-${c}` };
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!interactive) return;
    if (event.pointerType === "touch") return;
    event.preventDefault();
    activePointer.current = event.pointerId;
    lastCell.current = null;
    strokeStarted.current = false;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const cellHit = resolveCell(event.clientX, event.clientY);
    if (!cellHit) return;
    lastCell.current = cellHit.key;
    strokeStarted.current = true;
    onDown?.(cellHit.r, cellHit.c);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!interactive || activePointer.current !== event.pointerId) return;
    if (event.pointerType === "touch") return;
    event.preventDefault();
    const cellHit = resolveCell(event.clientX, event.clientY);
    if (!cellHit || cellHit.key === lastCell.current) return;
    lastCell.current = cellHit.key;
    if (!strokeStarted.current) {
      strokeStarted.current = true;
      onDown?.(cellHit.r, cellHit.c);
      return;
    }
    onEnter?.(cellHit.r, cellHit.c);
  };

  const handlePointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "touch") return;
    if (activePointer.current !== event.pointerId) return;
    activePointer.current = null;
    lastCell.current = null;
    strokeStarted.current = false;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    onUp?.();
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!interactive || !container) return;
    const onTouchStart = (event: TouchEvent) => {
      if (!interactive) return;
      event.preventDefault();
      activeTouch.current = true;
      touchStarted.current = false;
      lastTouchCell.current = null;
      const touch = event.touches[0];
      if (!touch) return;
      const cellHit = resolveCell(touch.clientX, touch.clientY);
      if (!cellHit) return;
      lastTouchCell.current = cellHit.key;
      touchStarted.current = true;
      onDown?.(cellHit.r, cellHit.c);
    };

    const onTouchMove = (event: TouchEvent) => {
      if (!interactive || !activeTouch.current) return;
      event.preventDefault();
      const touch = event.touches[0];
      if (!touch) return;
      const cellHit = resolveCell(touch.clientX, touch.clientY);
      if (!cellHit || cellHit.key === lastTouchCell.current) return;
      lastTouchCell.current = cellHit.key;
      if (!touchStarted.current) {
        touchStarted.current = true;
        onDown?.(cellHit.r, cellHit.c);
        return;
      }
      onEnter?.(cellHit.r, cellHit.c);
    };

    const onTouchEnd = (event: TouchEvent) => {
      if (!interactive) return;
      event.preventDefault();
      activeTouch.current = false;
      touchStarted.current = false;
      lastTouchCell.current = null;
      onUp?.();
    };

    container.addEventListener("touchstart", onTouchStart, { passive: false });
    container.addEventListener("touchmove", onTouchMove, { passive: false });
    container.addEventListener("touchend", onTouchEnd, { passive: false });
    container.addEventListener("touchcancel", onTouchEnd, { passive: false });
    return () => {
      container.removeEventListener("touchstart", onTouchStart);
      container.removeEventListener("touchmove", onTouchMove);
      container.removeEventListener("touchend", onTouchEnd);
      container.removeEventListener("touchcancel", onTouchEnd);
    };
  }, [interactive, onDown, onEnter, onUp, resolveCell]);

  return (
    <div
      ref={containerRef}
      className={`inline-grid gap-[1px] border border-2 border-gray-300 ${dimmed && "opacity-60"} ${interactive ? "select-none" : ""}`}
      style={{
        gridTemplateColumns: `repeat(${grid.length}, ${cell}px)`,
        touchAction: interactive ? "none" : "auto",
        WebkitUserSelect: interactive ? "none" : "auto",
      }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
    >
      {grid.map((row, r) =>
        row.map((v, c) => (
          <div
            key={`${r}-${c}`}
            data-life-cell="1"
            data-r={r}
            data-c={c}
            style={{ width: cell, height: cell, touchAction: interactive ? "none" : "auto" }}
            className={v ? "bg-blue-600" : "bg-white"}
          />
        ))
      )}
    </div>
  );
}

function Card({ title, children, placeholder }: { title: string; children?: React.ReactNode; placeholder?: string }) {
  return (
    <div className="flex flex-col items-center rounded-lg bg-white p-4 min-w-[340px]">
      <h3 className="mb-3 font-semibold">{title}</h3>
      {children ?? <span className="text-gray-400 italic">{placeholder}</span>}
    </div>
  );
}

function StepBar({ step }: { step: number }) {
  const steps = ["Draw", "Predict", "Simulate"];
  return (
    <ol className="flex justify-center gap-6 text-sm font-medium mb-4">
      {steps.map((s, i) => (
        <li
          key={s}
          className={`px-2 pb-[2px] border-b-2 ${i === step ? "border-blue-600 text-blue-700" : "border-transparent text-gray-400"}`}
        >
          {s}
        </li>
      ))}
    </ol>
  );
}

export default function App() {
  type Stage = 0 | 1 | 2;
  const empty = (): Grid => Array.from({ length: SIZE }, () => Array(SIZE).fill(0));

  const [stage, setStage] = useState<Stage>(0);
  const [grid, setGrid] = useState<Grid>(empty());
  const [prediction, setPrediction] = useState<Grid | null>(null);
  const [frames, setFrames] = useState<Grid[]>([]);
  const [idx, setIdx] = useState(-1);
  const [autoPlay, setAutoPlay] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // drawing helpers
  const drag = useRef(false);
  const paintVal = useRef(1);
  useEffect(() => {
    const up = () => (drag.current = false);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    return () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  }, []);

  const setCell = (r: number, c: number, v: number) =>
    setGrid((g) => {
      const n = g.map((row) => [...row]);
      n[r][c] = v;
      return n;
    });

  // local inference + simulation
  const predict = async () => {
    setBusy(true);
    setError("");
    try {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const p = predictWithCnn(grid);
      setPrediction(p);
      setStage(1);
    } catch {
      setError("Prediction failed. Try again.");
    } finally {
      setBusy(false);
    }
  };

  const simulate = async () => {
    setBusy(true);
    setError("");
    try {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const simulations = simulateSteps(grid, STEPS);
      setFrames(simulations);
      setIdx(0);
      setAutoPlay(true);
      setStage(2);
    } catch {
      setError("Simulation failed. Try again.");
    } finally {
      setBusy(false);
    }
  };

  // autoplay
  useEffect(() => {
    if (stage !== 2 || !autoPlay) return;
    if (idx >= frames.length - 1) {
      setAutoPlay(false);
      return;
    }
    const t = setTimeout(() => setIdx((i) => i + 1), SIM_DELAY);
    return () => clearTimeout(t);
  }, [stage, autoPlay, idx, frames.length]);

  // manual controls
  const hop = (d: number) => {
    setIdx((i) => Math.min(Math.max(i + d, 0), frames.length - 1));
    setAutoPlay(false);
  };
  const onSlider = (e: ChangeEvent<HTMLInputElement>) => {
    setIdx(Number(e.target.value));
    setAutoPlay(false);
  };

  const resetToGrid = (nextGrid: Grid) => {
    setGrid(nextGrid);
    setPrediction(null);
    setFrames([]);
    setIdx(-1);
    setStage(0);
    setError("");
    setAutoPlay(false);
  };

  const reset = () => resetToGrid(empty());

  const leftGrid = stage === 2 ? frames[Math.max(0, idx)] : grid;

  const matchPct = useMemo(() => {
    if (stage !== 2 || !prediction || frames.length === 0) return null;
    const finalSim = frames[frames.length - 1];
    let same = 0;
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (finalSim[r][c] === prediction[r][c]) same++;
      }
    }
    return Math.round((same / (SIZE * SIZE)) * 100);
  }, [stage, frames, prediction]);

  const simulationFinished = stage === 2 && idx >= frames.length - 1 && !autoPlay;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      <header className="text-center space-y-2">
        <h1 className="text-3xl font-bold">
          Conway's Game of Life
        </h1>
        <p className="mt-2 text-base text-gray-600">
          <a
            href="https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"
            className="underline text-blue-600"
          >
            Conway's Game of Life
          </a>{" "}
          is governed by a set of simple rules, but it can produce complex patterns. It is an inherently chaotic system, which has no closed-form mathematical solution.
        </p>
        <p className="mt-2 text-base font-medium italic text-gray-600">
          While simulations must run step-by-step, can a neural network predict the 5th step right away?
        </p>
        <a
          href="https://github.com/MikaVohl/game-of-life-CNN"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center text-gray-600 font-bold hover:text-gray-800"
        >
          <svg
            className="mr-2"
            height="20"
            viewBox="0 0 16 16"
            version="1.1"
            width="20"
            aria-hidden="true"
          >
            <path
              fill="currentColor"
              d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 
                 6.53 5.47 7.59.4.07.55-.17.55-.38 
                 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52
                 -.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 
                 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 
                 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 
                 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 
                 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 
                 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 
                 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42
                 -3.58-8-8-8z"
            />
          </svg>
          View source code
        </a>
      </header>
      {error.trim() && <div className="bg-red-100 text-red-700 p-3 rounded">{error}</div>}

      <StepBar step={stage} />

      <div className="flex flex-wrap justify-center gap-6">
        {/* Left card */}
        <Card title={stage === 0 ? "Draw initial state" : "Simulation"}>
          <LifeGrid
            grid={leftGrid}
            interactive={stage === 0 && !busy}
            dimmed={stage > 0}
            onDown={(r, c) => {
              if (busy || stage !== 0) return;
              drag.current = true;
              paintVal.current = grid[r][c] ? 0 : 1;
              setCell(r, c, paintVal.current);
            }}
            onEnter={(r, c) => drag.current && stage === 0 && setCell(r, c, paintVal.current)}
            onUp={() => {
              drag.current = false;
            }}
          />

          {/* playback controls */}
          {stage === 2 && (
            <div className="mt-4 flex flex-col items-center gap-3">
              <input
                className="w-64 accent-blue-600"
                type="range"
                min={0}
                max={frames.length - 1}
                value={idx}
                onChange={onSlider}
              />
              <div className="flex items-center gap-3">
                <button className="btn-gray px-3" disabled={idx === 0} onClick={() => hop(-1)}>
                  ◀
                </button>
                {autoPlay ? (
                  <button className="btn-gray px-4" onClick={() => setAutoPlay(false)}>
                    ❚❚ Pause
                  </button>
                ) : (
                  <button className="btn-gray px-4" onClick={() => setAutoPlay(true)}>
                    ▶ Play
                  </button>
                )}
                <button className="btn-gray px-3" disabled={idx === frames.length - 1} onClick={() => hop(1)}>
                  ▶
                </button>
                <span className="text-sm text-gray-600">
                  Frame {idx + 1}/{frames.length}
                </span>
              </div>
            </div>
          )}
        </Card>

        {/* Right card */}
        {stage !== 0 && (
          <Card title="Neural Network Prediction" placeholder="(awaiting prediction…)">
            {prediction && <LifeGrid grid={prediction} />}
          </Card>
        )}
      </div>

      {/* accuracy banner */}
      {simulationFinished && matchPct !== null && (
        <div className="text-xl text-center font-bold text-green-500">
          {matchPct}% pixel match
        </div>
      )}

      {/* Action buttons */}
      <div className="flex flex-col items-center gap-3">
        <div className="flex justify-center gap-4">
          {stage === 0 && (
            <button disabled={busy} onClick={predict} className="btn-primary">
              {busy ? "Predicting…" : "Predict"}
            </button>
          )}
          {stage === 1 && (
            <button disabled={busy} onClick={simulate} className="btn-blue">
              {busy ? "Loading…" : "Simulate"}
            </button>
          )}
          {simulationFinished && (
            <button onClick={reset} className="btn-secondary">
              Start over
            </button>
          )}
        </div>

        {stage === 0 && (
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button className="btn-gray" onClick={reset} disabled={busy}>
              Clear
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
