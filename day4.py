from pathlib import Path

import numpy as np
from itrx import Itr
from neworder.domain import StateGrid  # not cheating cos I wrote this package!

test_data = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@."""


with Path("./day4.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    grid = StateGrid(np.array(Itr(data.split("\n")).map(lambda row: [{".": 0.0, "@": 1.0}[c] for c in row]).collect()))
    n = grid.count_neighbours(lambda x: x > 0)
    return np.logical_and(n < 4, grid.state == 1.0).sum()


def part2(data: str) -> int:
    grid = StateGrid(np.array(Itr(data.split("\n")).map(lambda row: [{".": 0.0, "@": 1.0}[c] for c in row]).collect()))
    initial_count = grid.state.sum()
    while True:
        n = grid.count_neighbours(lambda x: x > 0)
        to_remove = np.logical_and(n < 4, grid.state == 1.0)
        if to_remove.sum() == 0:
            break
        grid.state -= to_remove
    return int(initial_count - grid.state.sum())


if __name__ == "__main__":
    print(part1(test_data))  # 13
    print(part1(data))  # 1540
    print(part2(test_data))  # 43
    print(part2(data))  # 8972
