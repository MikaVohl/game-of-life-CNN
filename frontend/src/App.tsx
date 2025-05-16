import { useState, useEffect } from 'react';
import './App.css';

function App() {
  // Core state
  const [grid, setGrid] = useState(() =>
    Array.from({ length: 64 }, () => Array(64).fill(0))
  );
  const [simulations, setSimulations] = useState<number[][][]>([]);
  const [prediction, setPrediction] = useState<number[][] | null>(null);
  const [loading, setLoading] = useState(false);
  const [stepIndex, setStepIndex] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);

  // Toggle a cell
  const toggleCell = (row: number, col: number) => {
    if (isRunning || stepIndex >= 0) return;
    setGrid(prev => {
      const newGrid = prev.map(r => [...r]);
      newGrid[row][col] = prev[row][col] === 1 ? 0 : 1;
      return newGrid;
    });
  };

  // Run simulate + predict
  const runSimulation = async () => {
    setLoading(true);
    try {
      // 1) fetch simulations
      const simRes = await fetch('http://localhost:5001/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid, steps: 5 })
      });
      const simData = await simRes.json();
      const sims: number[][][] = simData.simulations;

      // 2) fetch prediction of 5th step
      const predRes = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid })
      });
      const predData = await predRes.json();
      const pred: number[][] = predData.prediction;

      // set states and start animation
      setSimulations(sims);
      setPrediction(pred);
      setStepIndex(0);
      setIsRunning(true);
    } catch (error) {
      console.error('Error during run:', error);
    } finally {
      setLoading(false);
    }
  };

  // Animation effect
  useEffect(() => {
    if (isRunning) {
      if (stepIndex < simulations.length) {
        const timer = setTimeout(() => setStepIndex(stepIndex + 1), 500);
        return () => clearTimeout(timer);
      } else {
        setIsRunning(false);
      }
    }
  }, [isRunning, stepIndex, simulations.length]);

  // Render a grid; cells clickable when in initial state
  const renderGrid = (g: number[][]) => (
    <div
      className="grid gap-px border border-gray-600"
      style={{ gridTemplateColumns: 'repeat(64, 1rem)' }}
    >
      {g.map((row, i) =>
        row.map((cell, j) => (
          <div
            key={`${i}-${j}`}
            className={`${cell ? 'bg-black' : 'bg-white'} w-4 h-4`}
            onClick={() => toggleCell(i, j)}
          />
        ))
      )}
    </div>
  );

  // Determine which single grid to show during animation
  const currentGrid = () => {
    if (stepIndex < 0) return grid;
    if (stepIndex < simulations.length) return simulations[stepIndex];
    return simulations[simulations.length - 1];
  };

  return (
    <div className="container mx-auto p-4">
      <h2 className="bg-blue-500 text-white p-4 text-center text-2xl font-bold">
        Game of Life Simulation & Prediction
      </h2>

      {/* Grid Stage */}
      <div className="mt-4">
        {(stepIndex < simulations.length && stepIndex >= 0)
          ? renderGrid(currentGrid())
          : renderGrid(grid)
        }
      </div>

      {/* Side-by-Side Final */}
      {stepIndex >= simulations.length && prediction && (
        <div className="mt-4 flex justify-center gap-8">
          <div>
            <h3 className="text-center font-medium">Simulation (5th)</h3>
            {renderGrid(simulations[simulations.length - 1])}
          </div>
          <div>
            <h3 className="text-center font-medium">Prediction</h3>
            {renderGrid(prediction)}
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="mt-4 text-center">
        <button
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
          onClick={runSimulation}
          disabled={loading || isRunning}
        >
          {loading
            ? 'Loading...'
            : isRunning
            ? `Step ${Math.min(stepIndex + 1, simulations.length)}/${simulations.length}`
            : 'Run Simulation'}
        </button>
      </div>
    </div>
  );
}

export default App;
