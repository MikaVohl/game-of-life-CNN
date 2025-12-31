import rawModel from "./model_weights.json";

export type Grid = number[][];

type RawConv = {
  weight: number[];
  bias: number[];
  shape: number[];
  padding: number;
};

type RawBatchNorm = {
  weight: number[];
  bias: number[];
  mean: number[];
  var: number[];
  eps: number;
};

type RawModel = {
  blocks: { conv: RawConv; bn: RawBatchNorm }[];
  final: { conv: RawConv };
  size: number;
};

type ConvWeights = {
  weight: Float32Array;
  bias: Float32Array;
  shape: [number, number, number, number];
  padding: number;
};

type BatchNorm = {
  weight: Float32Array;
  bias: Float32Array;
  mean: Float32Array;
  variance: Float32Array;
  eps: number;
};

type Model = {
  blocks: { conv: ConvWeights; bn: BatchNorm }[];
  final: { conv: ConvWeights };
  size: number;
};

const parsedModel = rawModel as RawModel;

const model: Model = {
  size: parsedModel.size,
  blocks: parsedModel.blocks.map((block) => ({
    conv: {
      weight: new Float32Array(block.conv.weight),
      bias: new Float32Array(block.conv.bias),
      shape: block.conv.shape as [number, number, number, number],
      padding: block.conv.padding,
    },
    bn: {
      weight: new Float32Array(block.bn.weight),
      bias: new Float32Array(block.bn.bias),
      mean: new Float32Array(block.bn.mean),
      variance: new Float32Array(block.bn.var),
      eps: block.bn.eps,
    },
  })),
  final: {
    conv: {
      weight: new Float32Array(parsedModel.final.conv.weight),
      bias: new Float32Array(parsedModel.final.conv.bias),
      shape: parsedModel.final.conv.shape as [number, number, number, number],
      padding: parsedModel.final.conv.padding,
    },
  },
};

function gridToTensor(grid: Grid): Float32Array {
  const size = grid.length;
  const tensor = new Float32Array(size * size);
  for (let r = 0; r < size; r++) {
    const row = grid[r];
    for (let c = 0; c < size; c++) {
      tensor[r * size + c] = row[c] ? 1 : 0;
    }
  }
  return tensor;
}

function tensorToGrid(tensor: Float32Array, size: number): Grid {
  const grid: Grid = Array.from({ length: size }, () => Array(size).fill(0));
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      grid[r][c] = tensor[r * size + c] > 0.5 ? 1 : 0;
    }
  }
  return grid;
}

function conv2d(input: Float32Array, inChannels: number, size: number, conv: ConvWeights): Float32Array {
  const [outChannels, weightInChannels, kH, kW] = conv.shape;
  if (weightInChannels !== inChannels) {
    throw new Error(`Conv expected ${weightInChannels} channels, got ${inChannels}`);
  }

  const output = new Float32Array(outChannels * size * size);
  const pad = conv.padding;
  const area = size * size;
  const kArea = kH * kW;

  for (let oc = 0; oc < outChannels; oc++) {
    const outBase = oc * area;
    const weightOc = oc * inChannels * kArea;
    const bias = conv.bias[oc];

    for (let y = 0; y < size; y++) {
      const rowBase = y * size;
      for (let x = 0; x < size; x++) {
        let sum = bias;

        for (let ic = 0; ic < inChannels; ic++) {
          const weightIc = weightOc + ic * kArea;
          const inBase = ic * area;

          for (let ky = 0; ky < kH; ky++) {
            const inY = y + ky - pad;
            if (inY < 0 || inY >= size) continue;
            const inRow = inY * size;

            for (let kx = 0; kx < kW; kx++) {
              const inX = x + kx - pad;
              if (inX < 0 || inX >= size) continue;
              const weightIdx = weightIc + ky * kW + kx;
              sum += conv.weight[weightIdx] * input[inBase + inRow + inX];
            }
          }
        }

        output[outBase + rowBase + x] = sum;
      }
    }
  }

  return output;
}

function batchNormReluInPlace(input: Float32Array, channels: number, size: number, bn: BatchNorm): Float32Array {
  const area = size * size;
  for (let c = 0; c < channels; c++) {
    const scale = bn.weight[c] / Math.sqrt(bn.variance[c] + bn.eps);
    const shift = bn.bias[c] - bn.mean[c] * scale;
    const base = c * area;

    for (let i = 0; i < area; i++) {
      const v = input[base + i] * scale + shift;
      input[base + i] = v > 0 ? v : 0;
    }
  }
  return input;
}

export function predictWithCnn(grid: Grid): Grid {
  const size = grid.length;
  if (size !== model.size) {
    throw new Error(`Expected ${model.size}x${model.size} grid`);
  }

  let tensor = gridToTensor(grid);
  let channels = 1;

  for (const block of model.blocks) {
    tensor = conv2d(tensor, channels, size, block.conv);
    channels = block.conv.shape[0];
    batchNormReluInPlace(tensor, channels, size, block.bn);
  }

  tensor = conv2d(tensor, channels, size, model.final.conv);

  return tensorToGrid(tensor, size);
}

export function simulateSteps(grid: Grid, steps: number): Grid[] {
  const frames: Grid[] = [];
  let current = grid;
  for (let i = 0; i < steps; i++) {
    current = nextGeneration(current);
    frames.push(current);
  }
  return frames;
}

export function nextGeneration(grid: Grid): Grid {
  const size = grid.length;
  const next: Grid = Array.from({ length: size }, () => Array(size).fill(0));

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      let count = 0;
      for (let dr = -1; dr <= 1; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= size) continue;
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const cc = c + dc;
          if (cc < 0 || cc >= size) continue;
          if (grid[rr][cc]) count++;
        }
      }
      next[r][c] = count === 3 || (grid[r][c] && count === 2) ? 1 : 0;
    }
  }

  return next;
}
