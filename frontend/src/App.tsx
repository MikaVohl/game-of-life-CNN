import { useState, useEffect, useRef } from "react";
import "./App.css";

// Types & constants

type Grid = number[][];
const GRID_SIZE = 32;
const STEPS = 5;
const SIM_RATE = 650

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
    <div className="bg-white p-4">
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
      const simRes = await fetch("https://4xpmr3lpioemmys4ftyvzt37oe0kdjlx.lambda-url.us-east-1.on.aws/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grid, steps: STEPS }),
      });
      const { simulations: sims } = await simRes.json();
      const predRes = await fetch("https://4xpmr3lpioemmys4ftyvzt37oe0kdjlx.lambda-url.us-east-1.on.aws/predict", {
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
      // no pause for the jump from state 1→state 2
      const delay = stepIndex === 0 ? 0 : SIM_RATE;
      const t = setTimeout(
        () => setStepIndex(i => i + 1),
        delay
      );
      return () => clearTimeout(t);
    }
    setIsRunning(false);
  }, [isRunning, stepIndex, simulations.length]);

  const currentGrid: Grid =
    stepIndex < 0
    ? grid
    : stepIndex < simulations.length
    ? simulations[stepIndex]
  : grid;   

  return (
    <main className="w-full px-4 sm:px-6 lg:px-8 space-y-8 py-6">
      <header className="text-center space-y-2">
        <h1 className="text-3xl font-bold">
          Conway's Game of Life —
          <span className="text-blue-600"> Prediction vs Simulation</span>
        </h1>
        <p className="mt-2 text-base text-gray-600">
          Drag to paint live cells, then run five-step comparison.
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
        {(loading || isRunning) && (
          <ProgressBar
          value={loading ? 1 : stepIndex + 1}
          max={loading ? STEPS : simulations.length}
          />
        )}
        <div className="flex justify-center gap-6">
          <button
            className="btn-primary px-6 py-3 text-lg"
            disabled={loading || isRunning}
            onClick={runSimulation}
          >
          {loading
            ? `Step 1/${STEPS}`
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
