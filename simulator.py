from __future__ import annotations
import random, h5py, numpy as np

class Grid:
    def __init__(
        self,
        grid_size: int,
        grid: list[list[bool]] | None = None,
        random_init: bool = False,
        starting_cells: int = 0,
    ):
        self.grid_size = grid_size
        if random_init:
            self.grid = Grid.init_random(grid_size, starting_cells)
        else:
            self.grid = grid or [[False]*grid_size for _ in range(grid_size)]

    @staticmethod
    def init_random(grid_size: int, starting_cells: int) -> list[list[bool]]:
        grid = [[False]*grid_size for _ in range(grid_size)]
        for pos in random.sample(range(grid_size**2), starting_cells):
            grid[pos // grid_size][pos % grid_size] = True
        return grid

    def _evaluate(self, row: int, col: int) -> bool:
        cnt = 0
        for i in range(max(row-1, 0), min(row+2, self.grid_size)):
            for j in range(max(col-1, 0), min(col+2, self.grid_size)):
                if (i != row or j != col) and self.grid[i][j]:
                    cnt += 1
        return cnt == 3 or (self.grid[row][col] and cnt == 2)

    def step(self) -> list[list[bool]]:
        self.grid = [
            [self._evaluate(i, j) for j in range(self.grid_size)]
            for i in range(self.grid_size)
        ]
        return self.grid

    def gen_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (current, next) as uint8 arrays."""
        cur = np.array(self.grid, dtype=np.uint8)
        nxt = np.array(self.step(), dtype=np.uint8)
        return cur, nxt


def generate_pairs(size: int, start_range: tuple[int, int], n: int):
    for _ in range(n):
        n_start = random.randint(*start_range)
        g = Grid(size, random_init=True, starting_cells=n_start)
        yield g.gen_pair()


def save_to_hdf5(path: str, size: int, start_range: tuple[int, int], n: int):
    chunk_shape = (1, size, size)
    
    with h5py.File(path, "w-") as f:
        ds_x = f.create_dataset(
            "X",
            shape=(n, size, size),
            dtype="uint8",
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=4,   # you can tune this (1–9)
        )
        ds_y = f.create_dataset(
            "Y",
            shape=(n, size, size),
            dtype="uint8",
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=4,
        )

        for idx, (x, y) in enumerate(generate_pairs(size, start_range, n)):
            # make sure x and y are np.ndarray of shape (size,size), dtype uint8
            ds_x[idx : idx+1, :, :] = x[None, ...]
            ds_y[idx : idx+1, :, :] = y[None, ...]

            if idx % 100 == 0:
                print(f"  wrote {idx}/{n}")

    print("Done writing", path)

if __name__ == "__main__":
    save_to_hdf5("life_128.h5", size=128, start_range=(200, 400), n=1000)