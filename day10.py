from itertools import permutations, product
from math import prod
from pathlib import Path

import numpy as np
import numpy.typing as npt
from itrx import Itr

test_data = """[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}"""


with Path("./day10.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    machines = Itr(data.split("\n")).map(lambda line: Itr(line.split(" ")))

    total_presses = 0

    for line in machines:
        lights = [c == "#" for c in list(next(line))[1:-1]]
        *buttons, _ = line.collect()
        buttons = Itr(buttons).map(eval).collect()

        def apply(presses) -> list[bool]:
            state = [False] * len(lights)
            for p in presses:
                if isinstance(p, int):
                    state[p] = not state[p]
                else:
                    for i in p:
                        state[i] = not state[i]
            return state

        n_presses = 1
        while n_presses <= len(buttons):
            matched = False
            for presses in permutations(buttons, n_presses):
                if apply(presses) == lights:
                    matched = True
                    # print("matched", n_presses, lights, presses)
                    break
            if matched:
                total_presses += n_presses
                break
            n_presses += 1
    return total_presses


def part2(data: str) -> int:
    machines = Itr(data.split("\n")).map(lambda line: Itr(line.split(" ")))

    total_presses = 0

    for line in machines:
        n_lights = len(line.next()) - 2
        *buttons, joltage = line.collect()
        buttons = Itr(buttons).map(eval).map(lambda b: {b} if isinstance(b, int) else set(b))

        joltage = np.array(eval(joltage.replace("}", "]").replace("{", "[")), dtype=int)

        def press(button: set[int]) -> npt.NDArray:
            v = np.zeros(n_lights, dtype=int)
            v[list(button)] += 1
            return v

        presses = np.vstack(Itr(buttons).map(press).collect()).T

        # solve Ax = b where x is no of times each button press, b is joltage
        def brute_force_integer_solve(A: npt.NDArray, b: npt.NDArray) -> npt.NDArray | None:
            """
            Brute-force integer solution search for very small problems.
            """

            result = np.full(b.shape, max(b) + 1)

            ubounds = presses.T * joltage
            ubounds = np.where(ubounds == 0, 1000000, ubounds).min(axis=1)

            if prod(ubounds) > 1_000_000_000:
                raise ValueError(f"search space is: {prod(ubounds)}")

            for x_tuple in product(*(list(range(n + 1)) for n in ubounds)):
                x = np.array(x_tuple)
                if np.array_equal(A @ x_tuple, b) and x.sum() < result.sum():
                    result = x
            return result.sum() if result.sum() <= b.sum() else None

        n_presses = brute_force_integer_solve(presses, joltage)
        # print(n_presses)
        if n_presses:
            total_presses += n_presses

    return total_presses


if __name__ == "__main__":
    print(part1(test_data))  # 7
    print(part1(data))  # 438
    print(part2(test_data))  # 33
    # bruce-force fails miserably on part 2
    print(part2(data))  #
