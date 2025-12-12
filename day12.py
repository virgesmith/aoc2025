from pathlib import Path

import numpy as np
from itrx import Itr

test_data = """0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2"""

with Path("./day12.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    chunks = Itr(data.split("\n\n"))
    shapes = chunks.take(6).map(lambda raw: np.array([list(row) for row in raw.split("\n")[1:]]) == "#").collect()
    regions = Itr(line.split(":") for line in next(chunks).split("\n")).map(
        lambda sq: (tuple(map(int, sq[0].split("x"))), tuple(map(int, sq[1].split())))
    )

    print(shapes)
    print(regions.collect())

    return 0


def part2(data: str) -> int:
    pass


if __name__ == "__main__":
    print(part1(test_data))  #
    print(part1(data))  #
    # print(part2(test_data2))  # 2
    # print(part2(data))  #
