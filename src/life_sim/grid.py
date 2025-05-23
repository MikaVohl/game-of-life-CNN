import numpy as np

class Grid:
    def __init__(self, grid_size: int, grid: np.ndarray | None = None, random_init: bool = False, starting_cells: int = 0, rng: np.random.Generator | None = None,):
        self.grid_size = grid_size
        self.rng = rng or np.random.default_rng()
        if random_init:
            self.grid = self.init_random(grid_size, starting_cells)
        else:
            if grid is None:
                self.grid = np.zeros((grid_size, grid_size), dtype=bool)
            else:
                self.grid = grid.astype(bool, copy=False)

    def init_random(self, grid_size: int, n_alive: int) -> np.ndarray:
        flat = np.zeros(grid_size * grid_size, dtype=bool)
        if n_alive:
            idx = self.rng.choice(flat.size, n_alive, replace=False)
            flat[idx] = True
        return flat.reshape(grid_size, grid_size)

    def _next_generation(self) -> np.ndarray:
        g = self.grid
        padded = np.pad(g, 1, constant_values=False).astype(np.uint8) # (N+2, N+2). Pad the entire grid
        neigh = ( # creates a integer 2D array with each element representing how many neighbours it has
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] +                    padded[1:-1, 2:] +
            padded[2:,  :-2] + padded[2:,  1:-1] + padded[2:,  2:]
        ) # uint8 array of neighbour counts (shape N×N)
        # apply Conway’s rules
        return (neigh == 3) | (g & (neigh == 2))

    def step(self) -> list[list[bool]]:
        self.grid = self._next_generation()

    def gen_pair(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        start = self.grid.astype(np.uint8, copy=False)
        for _ in range(N):
            self.step()
        return start, self.grid.astype(np.uint8, copy=False) # cast boolean array to integer


def generate_pairs(N: int, size: int, start_range: tuple[int, int], n: int):
    rng = np.random.default_rng()
    for _ in range(n):
        n_start = rng.integers(*start_range, endpoint=False)
        g = Grid(size, random_init=True, starting_cells=n_start, rng=rng)
        yield g.gen_pair(N)

def print_grid(grid: list[list[bool]]):
    output = ''
    for row in grid:
        for entry in row:
            output += '⬜' if entry else '⬛'
        output += '\n'
    print(output)

def save_to_hdf5(path: str, N: int, size: int, start_range: tuple[int, int], n: int):
    import h5py
    chunk_shape = (1, size, size)
    with h5py.File(path, "w-") as f:
        ds_x = f.create_dataset(
            "X", (n, size, size), "uint8", chunks=chunk_shape,
            compression="gzip", compression_opts=4
        )
        ds_y = f.create_dataset(
            "Y", (n, size, size), "uint8", chunks=chunk_shape,
            compression="gzip", compression_opts=4
        )
        for idx, (x, y) in enumerate(generate_pairs(N, size, start_range, n)):
            ds_x[idx:idx+1] = x[None] # if x is of shape (size, size), then x[None] is of shape (1, size, size)
            ds_y[idx:idx+1] = y[None]
            if idx % 100 == 0:
                print(f"  wrote {idx}/{n}")
    print("Done writing", path)


if __name__ == "__main__":
    save_to_hdf5("life_32_10_50k.h5", N=10, size=32, start_range=(0, (32**2) * 1), n=50000)