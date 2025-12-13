from math import prod
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
    # parse input
    chunks = Itr(data.split("\n\n"))
    shapes = chunks.take(6).map(lambda raw: np.array([list(row) for row in raw.split("\n")[1:]]) == "#").collect()
    regions = Itr(line.split(":") for line in next(chunks).split("\n")).map(
        lambda sq: (tuple(map(int, sq[0].split("x"))), tuple(map(int, sq[1].split())))
    )

    # compute areas of shapes and regions, check if area of shapes isn't greater than area of region
    shape_areas = Itr(shapes).map(np.ndarray.sum).collect()
    region_area, coverage = regions.unzip()
    coverage = coverage.map(lambda idx: sum(map(prod, zip(idx, shape_areas, strict=True))))

    # not sure why this alone works, but its the right answer
    return coverage.zip(region_area.map(prod)).filter(lambda a: a[0] <= a[1]).count()


def part2(_: str) -> str:
    return "There is no part 2!"


if __name__ == "__main__":
    print(part1(test_data))  #
    print(part1(data))  #
    print(part2(test_data))  # 2
    print(part2(data))  #
