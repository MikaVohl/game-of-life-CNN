import random


class FixedBoard():
    def __init__(self, grid_size: int, grid: list[list[bool]]):
        self.grid_size = grid_size
        self.grid = grid

    def _evaluate(self, row, col):
        count = 0
        for i in range(max(row - 1, 0), min(row + 2, self.grid_size)):
            for j in range(max(col - 1, 0), min(col + 2, self.grid_size)):
                if self.grid[i][j] and not (i == row and j == col): count += 1
        if self.grid[row][col]:
            return count == 2 or count == 3
        else:
            return count == 3

    def forward_step(self):
        new_grid = [[False] * self.grid_size for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                new_grid[i][j] = self._evaluate(i,j)
        self.grid = new_grid
        return self.grid
    
def print_grid(grid: list[list[bool]]):
    output = ''
    for row in grid:
        for entry in row:
            output += '⬜' if entry else '⬛'
        output += '\n'
    print(output)

def random_grid(grid_size: int, starting_cells: int):
    grid = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    positions = random.sample(range(grid_size * grid_size), starting_cells)
    for pos in positions:
        row = pos // grid_size
        col = pos % grid_size
        grid[row][col] = True
    return grid
