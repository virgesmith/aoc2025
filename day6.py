from operator import add, mul
from pathlib import Path

from itrx import Itr

with Path("./day6test.txt").open() as fd:
    test_data = fd.read()

with Path("./day6.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    homework = Itr(data.split("\n")).map(lambda line: line.split()).rev()
    opmap = {"*": mul, "+": add}

    homework.next()  # skip blank line
    ops = Itr(homework.next()).map(lambda op: opmap[op]).collect()

    def do_ops(line0: list[int], line1: list[int]) -> list[int]:
        return [op(line0[i], line1[i]) for i, op in enumerate(ops)]

    return sum(homework.map(lambda line: list(map(int, line))).reduce(do_ops))


def part2(data: str) -> int:
    homework = Itr(data.split("\n")).map(lambda line: list(line)).rev()
    homework.next()  # skip blank line
    raw_ops = homework.next()

    total = 0
    subtotal = 0
    current_op = add

    def compute(next: tuple[str, str]) -> None:
        nonlocal total, subtotal, current_op
        n, op = next
        if op == "*":
            current_op = mul
            subtotal = 1
        elif op == "+":
            current_op = add
            subtotal = 0
        if n != "":
            subtotal = current_op(subtotal, int(n))
        else:
            total += subtotal

    Itr(map(list, zip(*homework.rev().collect(), strict=True))).map(lambda chars: "".join(chars).strip()).zip(
        raw_ops
    ).for_each(compute)
    return total + subtotal


if __name__ == "__main__":
    print(part1(test_data))  # 4277556
    print(part1(data))  # 6299564383938
    print(part2(test_data))  # 3263827
    print(part2(data))  # 11950004808442
