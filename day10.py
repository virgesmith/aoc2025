from itertools import permutations
from pathlib import Path

import numpy as np
import numpy.typing as npt
from itrx import Itr
from scipy.optimize import LinearConstraint, milp

test_data = """[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}"""


with Path("./day10.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    machines = Itr(data.split("\n")).map(lambda line: Itr(line.split(" ")))

    total_presses = 0

    for line in machines:
        lights = Itr(next(line)).map(lambda c: c == "#").collect(list)[1:-1]
        n_lights = len(lights)

        *buttons, _ = line.collect()
        buttons = Itr(buttons).map(eval).collect()

        def apply(presses: tuple[int, ...], n_lights: int) -> list[bool]:
            state = [False] * n_lights
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
                if apply(presses, n_lights) == lights:
                    matched = True
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

        def press(button: set[int], n_lights=n_lights) -> npt.NDArray:
            v = np.zeros(n_lights, dtype=int)
            v[list(button)] += 1
            return v

        presses = np.vstack(Itr(buttons).map(press).collect()).T

        x0 = np.ones(presses.shape[1])  # setting initially to zero sometime gives results that arent the minimum
        constraints = LinearConstraint(presses, joltage, joltage)
        integrality = np.ones(presses.shape[1], dtype=int)
        res = milp(x0, constraints=constraints, integrality=integrality)

        if not res.success:
            raise RuntimeError("milp failed")

        total_presses += int(sum(res.x))

    return total_presses


if __name__ == "__main__":
    print(part1(test_data))  # 7
    print(part1(data))  # 438
    print(part2(test_data))  # 33
    print(part2(data))  # 16463
