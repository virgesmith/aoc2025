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

    def non_edge_corners(pair):
        all = [all_corners[pair[0]], all_corners[pair[1]]]  # these ARE edge corners by definition
        all.append((all[0][0], all[1][1]))
        all.append((all[1][0], all[0][1]))
        return Itr(all).filter(lambda c: c not in all_corners).collect()

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
"""
 01234567890123
0..............
1.......0---1..
2.......|...|..
3..6----7...|..
4..|........|..
5..5------4.|..
6.........|.|..
7.........3-2..
8..............
"""


# def old():
#     def intersects(horiz, point):
#         xmin = min(horiz[0][0], horiz[1][0])
#         xmax = max(horiz[0][0], horiz[1][0])
#         return (xmin <= point[0] <= xmax) and (0 <= horiz[0][1] <= point[1])

#     for pair in pairs:

#         if areas[*pair] != 24:
#             continue

#         print(pair, areas[*pair])
#         x0, y0 = corners[pair[0]]
#         x1, y1 = corners[pair[1]]

#         interior_x = range(min(x0, x1), max(x0, x1) + 1)
#         interior_y = range(min(y0, y1), max(y0, y1) + 1)

#         sides = []

#         # TODO since holes are not possible, just test the boundary?
#         def inside() -> bool:
#             for x in interior_x:
#                 crossings= 0
#                 for h in horiz:
#                     crossings += intersects(h, (x, y0))
#                 print(f"({x}, {y0}):{crossings0} ({x}, {y1}):{crossings1}")
#                 if crossings0 % 2 == 0 or crossings1 % 2 == 0:
#                     return False

#             # for y in interior_y:
#             #     for x in interior_x:
#             #         crossings = 0
#             #         for h in horiz:
#             #             crossings += intersects(h, (x, y))
#             #         if crossings % 2 == 0:
#             #             return False
#             return True

#         if inside():
#             return areas[*pair]
#     return 0
