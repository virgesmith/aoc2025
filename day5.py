from operator import add
from pathlib import Path

from itrx import Itr

test_data = """3-5
10-14
16-20
12-18

1
5
8
11
17
32"""

with Path("./day5.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    available = Itr(data.split("\n"))
    fresh = (
        available.take_while(lambda line: line != "")
        .map(lambda line: line.split("-"))
        .map(lambda ij: range(int(ij[0]), int(ij[1]) + 1))
        .collect()
    )
    # remaining is available
    return available.filter(lambda n: any(int(n) in f for f in fresh)).count()


def part2(data: str) -> int:
    fresh = (Itr(data.split("\n"))
        .take_while(lambda line: line != "")
        .map(lambda line: line.split("-"))
        .map(lambda ij: (int(ij[0]), int(ij[1])))
        .collect(list)
    )

    def consolidate(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        consolidated = []
        for r in ranges:
            used = False
            for i, e in enumerate(consolidated):
                if r[0] <= e[1] <= r[1] or e[0] <= r[1] <= e[1]:
                    consolidated[i] = (min(e[0], r[0]), max(e[1], r[1]))
                    used = True
                    break
            if not used:
                consolidated.append(r)
        return consolidated

    n = len(fresh)
    while True:
        fresh = consolidate(fresh)
        if len(fresh) == n:  # no further consolidation
            break
        n = len(fresh)

    return sum(r[1] - r[0] + 1 for r in fresh)

if __name__ == "__main__":
    print(part1(test_data))  # 3
    print(part1(data))  # 821
    print(part2(test_data))  # 14
    print(part2(data))  # 344771884978261


