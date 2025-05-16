import { useState, useEffect, useRef } from "react";
import "./App.css";

// Types & constants

type Grid = number[][];
const GRID_SIZE = 32;
const STEPS = 5;

// UI helpers

function GridBoard({
  grid,
  cell = 8,
  clickable = false,
  onCellDown,
  onCellEnter,
}: {
  grid: Grid;
  cell?: number;
  clickable?: boolean;
  onCellDown?: (r: number, c: number) => void;
  onCellEnter?: (r: number, c: number) => void;
}) {
  return (
    <div
      className="inline-grid gap-[1px] bg-gray-400 border border-black"
      style={{ gridTemplateColumns: `repeat(${grid.length}, ${cell}px)` }}
    >
      {grid.map((row, r) =>
        row.map((v, c) => (
          <div
            key={`${r}-${c}`}
            style={{ width: cell, height: cell }}
            className={`${v ? "bg-blue-600" : "bg-white"} ${
              clickable ? "cursor-pointer" : ""
            }`}
            onMouseDown={() => clickable && onCellDown?.(r, c)}
            onMouseEnter={() => clickable && onCellEnter?.(r, c)}
          />
        ))
      )}
    </div>
  );
}

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = (value / max) * 100;
  return (
    <div className="mx-auto h-2 w-64 overflow-hidden rounded bg-gray-200">
      <div
        className="h-full bg-blue-500 transition-all duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function ComparisonCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border bg-white p-4 shadow-sm">
      <h2 className="mb-2 text-center font-semibold">{title}</h2>
      <div className="flex justify-center">{children}</div>
    </div>
  );
}

// Main component

export default function App() {
  const [grid, setGrid] = useState<Grid>(
    Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0))
  );
  const [simulations, setSimulations] = useState<Grid[]>([]);
  const [prediction, setPrediction] = useState<Grid | null>(null);
  const [stepIndex, setStepIndex] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [isRunning, setIsRunning] = useState(false);

  // track dragging state
  const dragging = useRef(false);
  const dragValue = useRef(1);

  // cleanup on mouse up
  useEffect(() => {
    const handleMouseUp = () => {
      dragging.current = false;
    };
    document.addEventListener("mouseup", handleMouseUp);
    return () => document.removeEventListener("mouseup", handleMouseUp);
  }, []);

  // cell update helper
  const setCell = (r: number, c: number, val: number) => {
    setGrid(prev => {
      const next = prev.map(row => [...row]);
      next[r][c] = val;
      return next;
    });
  };

  const handleCellDown = (r: number, c: number) => {
    if (isRunning || stepIndex >= 0) return;
    dragging.current = true;
    const current = grid[r][c];
    dragValue.current = current ? 0 : 1;
    setCell(r, c, dragValue.current);
  };

  const handleCellEnter = (r: number, c: number) => {
    if (dragging.current) {
      setCell(r, c, dragValue.current);
    }
  };

  const reset = () => {
    setGrid(Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0)));
    setSimulations([]);
    setPrediction(null);
    setStepIndex(-1);
    setIsRunning(false);
  };

  const runSimulation = async () => {
    setLoading(true);
    try {
      const simRes = await fetch("http://localhost:5001/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grid, steps: STEPS }),
      });
      const { simulations: sims } = await simRes.json();
      const predRes = await fetch("http://localhost:5001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grid }),
      });
      const { prediction: pred } = await predRes.json();
      setSimulations(sims);
      setPrediction(pred);
      setStepIndex(0);
      setIsRunning(true);
    } catch (err) {
      console.error("Run error", err);
    } finally {
      setLoading(false);
    }
  };

  // animate
  useEffect(() => {
    if (!isRunning) return;
    if (stepIndex < simulations.length) {
      const t = setTimeout(() => setStepIndex(i => i + 1), 550);
      return () => clearTimeout(t);
    }
    setIsRunning(false);
  }, [isRunning, stepIndex, simulations.length]);

  const currentGrid: Grid =
    stepIndex < 0
      ? grid
      : stepIndex < simulations.length
      ? simulations[stepIndex]
      : simulations[simulations.length - 1];

  return (
    <main className="w-full px-4 sm:px-6 lg:px-8 space-y-8 py-6">
      <header className="text-center">
        <h1 className="text-3xl font-bold">
          Conway’s Game of Life —
          <span className="text-blue-600"> Prediction vs Simulation</span>
        </h1>
        <p className="mt-2 text-base text-gray-600">
          Drag to paint live cells, then run five-step comparison.
        </p>
      </header>

      <section className="overflow-x-auto flex justify-center mb-6">
        <GridBoard
          grid={currentGrid}
          cell={12}
          clickable={!isRunning && stepIndex < 0}
          onCellDown={handleCellDown}
          onCellEnter={handleCellEnter}
        />
      </section>

      <section className="space-y-4">
        {isRunning && <ProgressBar value={stepIndex + 1} max={simulations.length} />}
        <div className="flex justify-center gap-6">
          <button
            className="btn-primary px-6 py-3 text-lg"
            disabled={loading || isRunning}
            onClick={runSimulation}
          >
            {loading
              ? "Running…"
              : isRunning
              ? `Step ${Math.min(stepIndex + 1, simulations.length)}/${simulations.length}`
              : "Run"}
          </button>
          <button
            className="btn-secondary px-6 py-3 text-lg"
            disabled={isRunning || loading}
            onClick={reset}
          >
            Reset
          </button>
        </div>
      </section>

      {stepIndex >= simulations.length && prediction && (
        <section className="grid gap-6 lg:grid-cols-2 overflow-x-auto">
          <ComparisonCard title="Simulation (step 5)">
            <GridBoard grid={simulations[simulations.length - 1]} cell={8} />
          </ComparisonCard>
          <ComparisonCard title="CNN Prediction">
            <GridBoard grid={prediction} cell={8} />
          </ComparisonCard>
        </section>
      )}
    </main>
  );
}
