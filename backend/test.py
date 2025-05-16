import h5py
import numpy as np
from simulator import Grid

path = "life_64.h5"
index = 0

with h5py.File(path, "r") as f:
    x = f["X"][index]      # uint8 array of shape (64,64)
    y = f["Y"][index]      # uint8 array of shape (64,64)

print("Loaded shapes:", x.shape, y.shape)
print("Live-cell counts:", int(x.sum()), "->", int(y.sum()))

# build from boolean grid directly
g = Grid(grid_size=x.shape[0], grid=(x > 0))

# gen_pair returns (cur, next) as uint8 arrays
_, computed = g.gen_pair()

print("Match?", bool(np.array_equal(computed, y)))