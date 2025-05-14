import h5py
import numpy as np
from simulator import Grid

path = "life_128.h5"
index = 0

with h5py.File(path, "r") as f:
    x = np.array(f["X"][index]) # current board
    y = np.array(f["Y"][index]) # expected next board

print("Loaded shapes:", x.shape, y.shape) # (128, 128)
print("Live-cell counts:", x.sum(), "->", y.sum())

g = Grid(grid_size=x.shape[0], grid=x.tolist())
computed = np.array(g.step(), dtype=np.uint8)

print("Match?", np.array_equal(computed, y))