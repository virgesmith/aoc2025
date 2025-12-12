from graphlib import TopologicalSorter
from operator import mul
from pathlib import Path

from itrx import Itr

test_data = """aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out"""

test_data2 = """svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out"""


with Path("./day11.txt").open() as fd:
    data = fd.read()


def part1(data: str) -> int:
    dag = (
        Itr(data.split("\n"))
        .map(lambda line: line.split(":"))
        .map(lambda kv: (kv[0], kv[1].strip().split(" ")))
        .collect(dict)
    )

    stack = ["you"]
    count = 0
    while stack:
        pos = stack.pop()
        if pos == "out":
            count += 1
        elif pos in dag:
            stack.extend(dag[pos])
    return count


def part2(data: str) -> int:
    dag = (
        Itr(data.split("\n"))
        .map(lambda line: line.split(":"))
        .map(lambda kv: (kv[0], kv[1].strip().split(" ")))
        .collect(dict)
    )

    def count_paths_dfs(start: str, target: str) -> int:
        memo: dict[str, int] = {}

        def dfs_impl(node: str) -> int:
            if node == target:
                return 1
            if node in memo:
                return memo[node]

            total = 0
            for child in dag.get(node, []):
                total += dfs_impl(child)

            memo[node] = total
            return total

        return dfs_impl(start)

    # there are actually no svr->dac->fft->out paths
    return Itr(["svr", "dac", "fft", "out"]).pairwise().starmap(count_paths_dfs).reduce(mul) + Itr(
        ["svr", "fft", "dac", "out"]
    ).pairwise().starmap(count_paths_dfs).reduce(mul)


if __name__ == "__main__":
    print(part1(test_data))  # 5
    print(part1(data))  # 764
    print(part2(test_data2))  # 33
    print(part2(data))  # 462444153119850
