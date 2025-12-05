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
    fresh = Itr(
        sorted(
            Itr(data.split("\n"))
            .take_while(lambda line: line != "")
            .map(lambda line: [int(n) for n in line.split("-")])
            .collect()
        )
    )

    total = 0
    current = fresh.next()

    def merge(r: list[int]) -> None:
        nonlocal current, total
        if r[0] <= current[1] + 1:  # merge overlapping/adjacent intervals
            current[1] = max(current[1], r[1])
        else:
            total += current[1] - current[0] + 1
            current = r

    fresh.for_each(merge)
    total += current[1] - current[0] + 1
    return total


if __name__ == "__main__":
    print(part1(test_data))  # 3
    print(part1(data))  # 821
    print(part2(test_data))  # 14
    print(part2(data))  # 344771884978261
