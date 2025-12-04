from operator import add
from pathlib import Path

from itrx import Itr

test_data = """987654321111111
811111111111119
234234234234278
818181911112111"""

with Path("./day3.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    def max2(n: list[int]) -> int:
        d1 = max(n[:-1])
        # print(d1, n.index(d1))
        d2 = max(n[n.index(d1) + 1 :])
        return d1 * 10 + d2

    return Itr(data.split("\n")).map(lambda s: [int(n) for n in s]).map(max2).reduce(add)


def part2(data: str) -> int:
    def max12(n: list[int]) -> int:
        digits = []
        idx = -1
        for i in range(12):
            start = idx + 1
            end = -11 + i if i < 11 else None
            d = max(n[start:end])
            pos = n[start:end].index(d)
            digits.append(d)
            idx += pos + 1
        return int("".join(str(d) for d in digits))

    # # copilot's slightly more elegant version
    # def max12(n: list[int]) -> int:
    #     # choose the lexicographically largest subsequence of length 12
    #     k = 12
    #     to_remove = len(n) - k
    #     stack: list[int] = []
    #     for d in n:
    #         while stack and to_remove and stack[-1] < d:
    #             stack.pop()
    #             to_remove -= 1
    #         stack.append(d)
    #     return int("".join(str(x) for x in stack[:k]))

    return Itr(data.split("\n")).map(lambda s: [int(n) for n in s]).map(max12).reduce(add)


if __name__ == "__main__":
    print(part1(test_data))  # 357
    print(part1(data))  # 17493
    print(part2(data))  # 173685428989126
