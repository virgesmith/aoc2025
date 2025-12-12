from pathlib import Path

import numpy as np
import numpy.typing as npt
from itrx import Itr

test_data = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"""


with Path("./day9.txt").open() as fd:
    data = fd.read()


def rect_areas(corners: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    areas = (abs(corners[:, np.newaxis, :] - corners[np.newaxis, :, :]) + 1).prod(axis=2)
    # mask duplicates
    return np.triu(areas)


def sorted_unique_indices(a: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    order = np.argsort(a.ravel(), kind="stable")
    indices = np.vstack(np.unravel_index(order, a.shape)).T
    # only return the non-zero indices
    n_unique_pairings = a.shape[0] * (a.shape[0] - 1) // 2
    return indices[-n_unique_pairings:][::-1]


def part1(data: str) -> int:
    corners = Itr(data.split("\n")).map(lambda s: tuple(map(int, s.split(","))))
    areas = rect_areas(np.array(corners.collect()))
    return areas.max()


def part2(data: str) -> int:
    base = Itr(data.split("\n")).map(lambda s: tuple(map(int, s.split(","))))
    # join up the ends so we arent missing a line
    base = base.chain(base.copy().take(1))

    def make_line(a: int, b: int):
        return range(a, b + 1) if a < b else range(b, a + 1)

    horz = (
        base.copy()
        .pairwise()
        .filter(lambda pair: pair[0][1] == pair[1][1])
        .map(lambda pair: (make_line(pair[0][0], pair[1][0]), pair[0][1]))
    ).collect()
    vert = (
        base.copy()
        .pairwise()
        .filter(lambda pair: pair[0][0] == pair[1][0])
        .map(lambda pair: (make_line(pair[0][1], pair[1][1]), pair[0][0]))
    ).collect()

    def inside(p: int, lines) -> bool:
        return bool(sum(p in line for line, _ in lines) % 2)

    all_corners = base.copy().collect()

    areas = rect_areas(np.array(all_corners))
    pairs = sorted_unique_indices(areas)

    for pair in pairs:

        def interior_span(pair):
            x0, y0 = all_corners[pair[0]]
            x1, y1 = all_corners[pair[1]]
            return range(min(x0, x1) + 1, max(x0, x1)), range(min(y0, y1) + 1, max(y0, y1))

        x_range, y_range = interior_span(pair)

        def edges_intersect(span, xy, edges) -> bool:
            edges = [e for e in edges if e[1] in span and xy in e[0]]
            return len(edges) > 0

        if (
            edges_intersect(x_range, y_range.start, vert)
            or edges_intersect(x_range, y_range.stop - 1, vert)
            or edges_intersect(y_range, x_range.start, horz)
            or edges_intersect(y_range, x_range.stop - 1, horz)
        ):
            continue

        return areas[*pair]

    return 0


if __name__ == "__main__":
    print(part1(test_data))  # 50
    print(part1(data))  # 4748769124
    print(part2(test_data))  # 24
    print(part2(data))  # 1525991432
