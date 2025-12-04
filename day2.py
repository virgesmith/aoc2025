from operator import add

from itrx import Itr

test_data = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124"

data = "119-210,907313-1048019,7272640820-7272795557,6315717352-6315818282,42-65,2234869-2439411,1474-2883,33023-53147,1-15,6151-14081,3068-5955,65808-128089,518490556-518593948,3535333552-3535383752,7340190-7548414,547-889,34283147-34389716,44361695-44519217,607492-669180,7071078-7183353,67-115,969-1469,3636264326-3636424525,762682710-762831570,827113-906870,205757-331544,290-523,86343460-86510016,5536957-5589517,132876-197570,676083-793651,23-41,17920-31734,440069-593347"


def part1(data: str) -> int:
    def is_invalid(n: int) -> bool:
        s = str(n)
        split = len(s) // 2
        return split > 0 and int(s[:split] + s[:split]) == n

    return (
        Itr(data.split(","))
        .map(lambda r: [int(n) for n in r.split("-")])
        .flat_map(lambda ab: range(ab[0], ab[1] + 1))
        .filter(is_invalid)
    ).fold(0, add)


def part2(data: str) -> int:
    def is_invalid(n: int) -> bool:
        s = str(n)
        return any(int(s[i:] + s[:i]) == n for i in range(1, len(s) // 2 + 1))

    return (
        Itr(data.split(","))
        .map(lambda r: [int(n) for n in r.split("-")])
        .flat_map(lambda ab: range(ab[0], ab[1] + 1))
        .filter(is_invalid)
    ).fold(0, add)


if __name__ == "__main__":
    print(part1(test_data))  # 1227775554
    print(part1(data))  # 31000881061

    print(part2(test_data))  # 4174379265
    print(part2(data))  # 46769308485
