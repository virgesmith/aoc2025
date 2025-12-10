from math import prod
from pathlib import Path

import numpy as np
import numpy.typing as npt
from itrx import Itr

test_data = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689"""


with Path("./day8.txt").open() as fd:
    data = fd.read()


def sorted_unique_indices(a: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    order = np.argsort(a.ravel(), kind="stable")
    indices = np.vstack(np.unravel_index(order, a.shape)).T
    # only return the non-zero distance indices
    n_unique_pairings = a.shape[0] * (a.shape[0] - 1) // 2
    return indices[-n_unique_pairings:]


def dist_squared(p: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    d2 = ((p[:, np.newaxis, :] - p[np.newaxis, :, :]) ** 2).sum(axis=2)
    # fill the lower diagonal with zeros (will ignore)
    return np.triu(d2)


def part1(data: str, n: int) -> int:
    positions = Itr(data.split("\n")).map(lambda s: tuple(map(int, s.split(",")))).collect()

    # compute (squared) distances
    dist2s = dist_squared(np.array(positions))

    # start off with connection only to self
    connections = [{i} for i in range(len(positions))]

    def update_connections(pair: tuple[int, int]) -> None:
        nonlocal connections
        a, b = pair
        itr_a, itr_b = Itr(connections).enumerate().tee()
        ia = itr_a.skip_while(lambda c: a not in c[1]).next()[0]
        ib = itr_b.skip_while(lambda c: b not in c[1]).next()[0]
        if ia != ib:
            connections[min(ia, ib)] |= connections.pop(max(ia, ib))

    # select n shortest pairings and track connections
    Itr(sorted_unique_indices(dist2s)).take(n).for_each(update_connections)

    top3 = sorted({frozenset(ci) for ci in connections}, key=len)[-3:]
    return prod(len(s) for s in top3)


def part2(data: str) -> int:
    positions = Itr(data.split("\n")).map(lambda s: tuple(map(int, s.split(",")))).collect()

    # compute (squared) distances
    dist2s = dist_squared(np.array(positions))

    connections = [{i} for i in range(len(positions))]
    last_pair = -1, -1

    def update_connections(pair: tuple[int, int]) -> bool:
        nonlocal connections, last_pair
        a, b = pair
        itr_a, itr_b = Itr(connections).enumerate().tee()
        ia = itr_a.skip_while(lambda c: a not in c[1]).next()[0]
        ib = itr_b.skip_while(lambda c: b not in c[1]).next()[0]
        if ia != ib:
            connections[min(ia, ib)] |= connections.pop(max(ia, ib))
        if len(connections) == 1:
            last_pair = pair
            return False
        return True

    # select ordered pairings and track connections until all connected
    Itr(sorted_unique_indices(dist2s)).take_while(update_connections).consume()
    return positions[last_pair[0]][0] * positions[last_pair[1]][0]


if __name__ == "__main__":
    print(part1(test_data, 10))  # 40
    print(part1(data, 1000))  # 117000
    print(part2(test_data))  # 25272
    print(part2(data))  # 8368033065
