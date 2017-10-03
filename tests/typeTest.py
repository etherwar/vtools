from typing import Union, List, Tuple, NewType

check_odd = Union[Tuple[int], int]


def test(var: Tuple[int, int]):
    print(var)


test((3, 2))
test((3, ))
test(3)


def test2(var: check_odd):
    print(var)


test2((3, 2))
test2((3, ))
test2(3)
