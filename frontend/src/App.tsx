import { useState, useEffect, useRef, useMemo } from "react";
import type { ChangeEvent, PointerEvent } from "react";
import "./App.css";
import { predictWithCnn, simulateSteps, type Grid } from "./lifeModel";

const SIZE = 32;
const STEPS = 5;
const SIM_DELAY = 650;
const CELL = 14;

const buttonBase =
  "inline-flex items-center justify-center rounded-[6px] px-4 py-2 text-sm font-semibold tracking-[0.02em] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--accent-0)] focus-visible:ring-offset-2 focus-visible:ring-offset-[color:var(--bg)] disabled:cursor-not-allowed disabled:opacity-50";
const buttonPrimary = `${buttonBase} bg-[color:var(--accent-0)] text-[color:var(--text)] shadow-sm hover:bg-[color:var(--accent-1)] active:translate-y-[1px] active:shadow-none`;
const buttonSecondary = `${buttonBase} bg-[color:var(--card)] text-[color:var(--text)] shadow-sm hover:bg-[color:var(--bg)] active:translate-y-[1px] active:shadow-none`;
const buttonOutline = `${buttonBase} border border-[color:var(--border)] bg-transparent text-[color:var(--text)] shadow-none hover:bg-[color:var(--bg)] active:translate-y-[1px]`;
const buttonGhost = `${buttonOutline} h-8 px-3 py-1`;
const buttonIcon = `${buttonOutline} h-8 px-3 py-1`;

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
      className={`inline-block origin-top scale-[0.82] rounded-xl bg-[color:var(--card)] p-3 sm:scale-100 sm:p-4 ${
        dimmed && "opacity-70"
      }`}
    >
      <div
        ref={containerRef}
        className={`inline-grid gap-[1px] rounded-xs bg-[color:var(--gridline)] p-[1px] ${interactive ? "select-none" : ""}`}
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
              className={
                v
                  ? "rounded-[2px] bg-[color:var(--cell-on)] opacity-100 transition-[background-color,opacity] duration-150 ease-out"
                  : "rounded-[2px] bg-[color:var(--cell)] opacity-90 transition-[background-color,opacity] duration-150 ease-out"
              }
            />
          ))
        )}
      </div>
    </div>
  );
}

function Card({ title, children, placeholder }: { title: string; children?: React.ReactNode; placeholder?: string }) {
  return (
    <div className="flex w-full max-w-[520px] flex-col items-center rounded-xl bg-[color:var(--card)] p-4 sm:p-5 border border-[color:var(--border)]">
      <h3 className="font-semibold text-[color:var(--text)]">{title}</h3>
      {children ?? <span className="text-[color:rgba(18,18,18,0.55)] italic">{placeholder}</span>}
    </div>
  );
}

function StepBar({ step }: { step: number }) {
  const steps = ["Draw", "Predict", "Simulate"];
  return (
    <ol className="flex justify-center gap-3 text-xs font-medium mb-3 text-[color:var(--text)] sm:gap-6 sm:text-sm">
      {steps.map((s, i) => (
        <li
          key={s}
          className={`px-2 pb-[2px] border-b-2 ${
            i === step ? "border-[color:var(--border)]" : "border-transparent opacity-40"
          }`}
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
  const [toast, setToast] = useState("");
  const toastTimer = useRef<number | null>(null);

  // drawing helpers
  const drag = useRef(false);
  const paintVal = useRef(1);
  const lastPaintCell = useRef<{ r: number; c: number } | null>(null);
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

  const showToast = (message: string) => {
    setToast(message);
    if (toastTimer.current !== null) {
      window.clearTimeout(toastTimer.current);
    }
    toastTimer.current = window.setTimeout(() => {
      setToast("");
      toastTimer.current = null;
    }, 2200);
  };

  useEffect(() => {
    return () => {
      if (toastTimer.current !== null) {
        window.clearTimeout(toastTimer.current);
      }
    };
  }, []);

  const isGridEmpty = (current: Grid) => {
    for (let r = 0; r < current.length; r++) {
      const row = current[r];
      for (let c = 0; c < row.length; c++) {
        if (row[c]) return false;
      }
    }
    return true;
  };

  const linePoints = (from: { r: number; c: number }, to: { r: number; c: number }) => {
    let x0 = from.c;
    let y0 = from.r;
    const x1 = to.c;
    const y1 = to.r;
    const points: Array<[number, number]> = [];

    const dx = Math.abs(x1 - x0);
    const sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0);
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;

    while (true) {
      if (y0 >= 0 && y0 < SIZE && x0 >= 0 && x0 < SIZE) {
        points.push([y0, x0]);
      }
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x0 += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y0 += sy;
      }
    }

    return points;
  };

  const paintLine = (from: { r: number; c: number }, to: { r: number; c: number }, v: number) => {
    const points = linePoints(from, to);
    setGrid((g) => {
      const n = g.map((row) => [...row]);
      for (const [r, c] of points) {
        n[r][c] = v;
      }
      return n;
    });
  };

  // local inference + simulation
  const predict = async () => {
    if (isGridEmpty(grid)) {
      showToast("Add a few live cells before predicting.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      const p = predictWithCnn(grid);
      setPrediction(p);
      setStage(2);
      setFrames([grid]);
      setIdx(0);
      setAutoPlay(false);
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
      const simulations = simulateSteps(grid, STEPS);
      setFrames([grid, ...simulations]);
      setIdx(0);
      setAutoPlay(true);
    } catch {
      setError("Prediction failed. Try again.");
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
  const startPlayback = () => {
    if (frames.length === 0) return;
    setIdx((i) => (i >= frames.length - 1 ? 0 : i));
    setAutoPlay(true);
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
    <div className="max-w-6xl mx-auto px-3 py-6 space-y-4 sm:px-4 sm:py-8 sm:space-y-6">
      <header className="space-y-2 px-2 text-left sm:px-8 sm:text-center md:px-20">
        <h1 className="text-2xl font-bold text-[color:var(--text)] sm:text-3xl">
          Predicting Conway's Game of Life
        </h1>
        <p className="mt-2 text-sm text-[color:var(--text)] sm:text-base">
          <a
            href="https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"
            className="underline decoration-[color:var(--accent-0)] decoration-2 underline-offset-4 text-[color:var(--text)]"
          >
            Conway's Game of Life
          </a>{" "}
          is governed by a set of simple rules
          <span className="sm:hidden">, creating complex patterns.</span>
          <span className="hidden sm:inline">
            , but it can produce complex patterns. It is an inherently chaotic system, which has no closed-form
            mathematical solution.
          </span>
        </p>
        <p className="mt-2 text-sm font-medium text-[color:var(--text)] sm:hidden">
          Can a neural net jump straight to the 5th next state?
        </p>
        <p className="mt-2 text-base font-medium text-[color:var(--text)] hidden sm:block">
          {/* While simulations must run step-by-step, can a neural network predict the 5th step right away? */}
          Instead of simulating the game step-by-step, can a neural network predict the 5th next state in one shot?
        </p>
        <a
          href="https://github.com/MikaVohl/game-of-life-CNN"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex w-full items-center justify-center font-bold text-[color:var(--text)] text-sm sm:w-auto sm:text-base"
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
          View Source Code
        </a>
      </header>
      {error.trim() && (
        <div className="bg-[color:var(--card)] text-[color:var(--text)] p-3 rounded-lg border border-[color:var(--border)]">
          {error}
        </div>
      )}
      {toast && (
        <div className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-lg border border-[color:var(--border)] bg-[color:var(--card)] px-4 py-2 text-sm font-medium text-[color:var(--text)] shadow-sm">
          {toast}
        </div>
      )}

      <StepBar step={stage} />

      <div
        className={`grid gap-6 justify-items-center mb-3 ${stage === 0 ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2"}`}
      >
        {/* Left card */}
        <Card title={stage === 0 ? "Draw initial state" : "Simulation"}>
          <LifeGrid
            grid={leftGrid}
            cell={CELL}
            interactive={stage === 0 && !busy}
            dimmed={stage > 0}
            onDown={(r, c) => {
              if (busy || stage !== 0) return;
              drag.current = true;
              paintVal.current = grid[r][c] ? 0 : 1;
              lastPaintCell.current = { r, c };
              setCell(r, c, paintVal.current);
            }}
            onEnter={(r, c) => {
              if (!drag.current || stage !== 0) return;
              const last = lastPaintCell.current;
              if (!last) {
                lastPaintCell.current = { r, c };
                setCell(r, c, paintVal.current);
                return;
              }
              if (last.r === r && last.c === c) return;
              paintLine(last, { r, c }, paintVal.current);
              lastPaintCell.current = { r, c };
            }}
            onUp={() => {
              drag.current = false;
              lastPaintCell.current = null;
            }}
          />
          {stage === 0 && (
            <div className="flex justify-center">
              <button className={buttonOutline} onClick={reset} disabled={busy}>
                Clear grid
              </button>
            </div>
          )}

          {/* playback controls */}
          {stage === 2 && (
            <div className="flex flex-col items-center gap-2">
              <input
                className="w-56 accent-[#ff6a00] sm:w-64"
                type="range"
                min={0}
                max={frames.length - 1}
                value={idx}
                onChange={onSlider}
              />
              <div className="flex items-center gap-3">
                <button className={buttonIcon} disabled={idx === 0} onClick={() => hop(-1)}>
                  ◀
                </button>
                {autoPlay ? (
                  <button className={buttonGhost} onClick={() => setAutoPlay(false)}>
                    Pause
                  </button>
                ) : (
                  <button className={buttonGhost} onClick={startPlayback}>
                    Start
                  </button>
                )}
                <button className={buttonIcon} disabled={idx === frames.length - 1} onClick={() => hop(1)}>
                  ▶
                </button>
              </div>
            </div>
          )}
        </Card>

        {/* Right card */}
        {stage !== 0 && (
          <Card title="Neural Network Prediction" placeholder="(awaiting prediction…)">
            {prediction && <LifeGrid grid={prediction} cell={CELL} />}
            {simulationFinished && matchPct !== null && (
              <div className="mt-4 text-xl font-bold text-[color:var(--text)]">
                {matchPct}% pixel match
              </div>
            )}
          </Card>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex flex-col items-center gap-3">
        <div className="flex flex-wrap justify-center gap-4">
          {stage === 0 && (
            <button disabled={busy} onClick={predict} className={buttonPrimary}>
              {busy ? "Predicting…" : "Predict"}
            </button>
          )}
          {simulationFinished && (
            <button onClick={reset} className={buttonSecondary}>
              Start over
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
