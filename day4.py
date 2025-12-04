from pathlib import Path

import numpy as np
import numpy.typing as npt
from itrx import Itr
from xenoform import compile

test_data = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@."""


with Path("./day4.txt").open() as fd:
    data = fd.read()


@compile(extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"])
def neighbours_with(grid: npt.NDArray[np.int32], *, value: int) -> npt.NDArray[np.int32]:  # type: ignore[empty-body]
    """
    py::buffer_info buf = grid.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Input array must be 2D");

    size_t m = buf.shape[0];
    size_t n = buf.shape[1];

    py::array_t<int> result({m, n});
    auto r = result.mutable_unchecked<2>();
    auto p = grid.unchecked<2>();

    auto getval = [=](size_t i, size_t j) -> int {
        return (i < 0 || i >= m || j < 0 || j >= n) ? 0 : (p(i, j) == value);
    };

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            r(i, j) = getval(i - 1, j - 1) + getval(i - 1, j) + getval(i - 1, j + 1)
                    + getval(i    , j - 1)                    + getval(i    , j + 1)
                    + getval(i + 1, j - 1) + getval(i + 1, j) + getval(i + 1, j + 1);
        }
    }

    return result;
    """


def part1(data: str) -> int:
    grid = np.array(Itr(data.split("\n")).map(lambda row: [{".": 0, "@": 1}[c] for c in row]).collect())
    n = neighbours_with(grid, value=1)
    return np.logical_and(n < 4, grid == 1).sum()


def part2(data: str) -> int:
    grid = np.array(Itr(data.split("\n")).map(lambda row: [{".": 0.0, "@": 1.0}[c] for c in row]).collect())
    initial_count = grid.sum()
    while True:
        n = neighbours_with(grid, value=1)
        to_remove = np.logical_and(n < 4, grid == 1)
        if to_remove.sum() == 0:
            break
        grid -= to_remove
    return int(initial_count - grid.sum())


if __name__ == "__main__":
    print(part1(test_data))  # 13
    print(part1(data))  # 1540
    print(part2(test_data))  # 43
    print(part2(data))  # 8972
