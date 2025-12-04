from operator import add
from pathlib import Path

from itrx import Itr

test_data = """L68
L30
R48
L5
R60
L55
L1
L99
R14
L82"""

with Path("./day1.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    return (
        Itr(data.split("\n"))
        .map(lambda line: int(line.replace("R", "").replace("L", "-")))
        .accumulate(add, initial=50)
        .filter(lambda n: n % 100 == 0)
        .count()
    )


def part2(data: str) -> int:
    return (
        Itr(data.split("\n"))
        .map(lambda line: int(line.replace("R", "").replace("L", "-")))
        .flat_map(lambda n: [n / abs(n)] * abs(n))
        .accumulate(add, initial=50)
        .filter(lambda n: n % 100 == 0)
        .count()
    )


if __name__ == "__main__":
    print(part1(test_data))  # 3
    print(part1(data))  # 1029

    print(part2(test_data))  # 6
    print(part2(data))  # 5892
