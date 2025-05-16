import React, { useState } from 'react';
import './App.css';

function App() {
  // Initialize a 64×64 grid of zeros
  const [grid, setGrid] = useState(() =>
    Array.from({ length: 64 }, () => Array(64).fill(0))
  );
  const [loading, setLoading] = useState(false);

  // Toggle a cell between 0 and 1
  const toggleCell = (row: number, col: number) => {
    setGrid(prev => {
      const newGrid = prev.map(r => [...r]);
      newGrid[row][col] = prev[row][col] === 1 ? 0 : 1;
      return newGrid;
    });
  };

  // Send grid to API and update UI with response
  const postData = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ grid })
      });
      const result = await response.json();
      // Update grid state with prediction
      setGrid(result.prediction);
      console.log('Updated grid:');
      printGrid(result.prediction);
    } catch (error) {
      console.error('Error during POST request:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h2 className="bg-blue-500 text-white p-4 text-center text-2xl font-bold">
        Game of Life
      </h2>

      {/* Grid with border */}
      <div
        className="mt-4 grid gap-px cursor-pointer border-2 border-gray-600"
        style={{
          gridTemplateColumns: 'repeat(64, 1rem)',
        }}
      >
        {grid.map((row, i) =>
          row.map((cell, j) => (
            <div
              key={`${i}-${j}`}
              className={`${cell ? 'bg-black' : 'bg-white'} w-4 h-4`}
              onClick={() => toggleCell(i, j)}
            />
          ))
        )}
      </div>

      {/* Button to trigger API */}
      <div className="mt-4 text-center">
        <button
          className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
          onClick={postData}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Run Prediction'}
        </button>
      </div>
    </div>
  );
}

function printGrid(grid: number[][]) {
  let output = '';
  grid.forEach(row => {
    row.forEach(cell => {
      output += cell === 1 ? '⬛' : '⬜';
    });
    output += '\n';
  });
  console.log(output);
}

export default App;