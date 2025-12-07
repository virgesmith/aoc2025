from pathlib import Path

from itrx import Itr

test_data = """.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
..............."""

with Path("./day7.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    initial = Itr(data.split("\n")).map(lambda line: list(line))

    def propagate_beams(window: tuple[list[str], list[str]]) -> list[str]:
        above, below = window
        for i in range(len(above)):
            if above[i] in ["|", "S"]:
                if below[i] != "^":
                    below[i] = "|"
                else:
                    below[i - 1] = "|"
                    below[i + 1] = "|"
        return below

    splits = 0

    def count_splits(window: tuple[list[str], list[str]]) -> None:
        above, below = window
        nonlocal splits

        splits += Itr(above).zip(below).filter(lambda ab: ab == ("|", "^")).count()

    initial.pairwise().map(propagate_beams).pairwise().for_each(count_splits)
    return splits


def part2(data: str) -> int:
    # skip the alternate rows (no branches)
    initial = Itr(data.split("\n")).map(lambda line: list(line)).step_by(2)

    # use the first row to initialise the counter
    paths = [1 if c == "S" else 0 for c in initial.next()]

    def propagate(row: list[str]) -> None:
        nonlocal paths
        new_paths = paths.copy()
        for i, c in enumerate(row):
            if c == "^":
                new_paths[i] = 0
                new_paths[i - 1] += paths[i]
                new_paths[i + 1] += paths[i]
        paths = new_paths

    initial.for_each(propagate)

    return sum(paths)


if __name__ == "__main__":
    print(part1(test_data))  # 21
    print(part1(data))  # 1550
    print(part2(test_data))  # 40
    print(part2(data))  # 9897897326778
