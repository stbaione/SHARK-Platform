# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, List
from collections.abc import Iterable
from itertools import zip_longest
from operator import eq
import os
from contextlib import AbstractContextManager


def assert_equal(a: Any, b: Any, /, equal_fn: Callable[[Any, Any], bool] = eq):
    assert equal_fn(a, b), f"{a} != {b}"


def assert_sets_equal(set1: set, set2: set, /):
    assert len(set1) == len(
        set2
    ), f"Sets have different number of elements, {len(set1)} != {len(set2)}"
    for s1 in set1:
        assert s1 in set2, f"Element {s1} not found in set {set2}"


def verify_exactly_one_is_not_none(**kwargs):
    count = 0
    for v in kwargs.values():
        if v is not None:
            count += 1
    if count != 1:
        raise ValueError(f"Exactly one of {kwargs.keys()} must be set.")


def longest_equal_range(l1: List[Any], l2: List[Any]) -> int:
    """Find the longest range that is the same from the start of both lists.
    Returns the greatest `i` such that `l1[0:i] == l2[0:i]`."""
    for i, (a, b) in enumerate(zip(l1, l2)):
        if a != b:
            return i
    return min(len(list(l1)), len(list(l2)))


def iterables_equal(
    iterable1: Iterable,
    iterable2: Iterable,
    *,
    elements_equal: Callable[[Any, Any], bool] | None = None,
) -> bool:
    non_existent_value = object()
    elements_equal = elements_equal or eq

    def elements_equal_fn(x: Any, y: Any) -> bool:
        if x is non_existent_value or y is non_existent_value:
            return False
        return elements_equal(x, y)

    return all(
        elements_equal_fn(v1, v2)
        for v1, v2 in zip_longest(iterable1, iterable2, fillvalue=non_existent_value)
    )


class chdir(AbstractContextManager):
    """Context that changes the current working directory.

    with chdir("/path/to/new/cwd"):
        ...

    TODO: swap with contextlib.chdir once we drop support for Python 3.10
    """

    def __init__(self, path):
        self.path = path
        self._old_cwd = []

    def __enter__(self):
        self._old_cwd.append(os.getcwd())
        os.chdir(self.path)

    def __exit__(self, *excinfo):
        os.chdir(self._old_cwd.pop())


def parse_version(v: str, /) -> tuple[int, ...]:
    """Parse a version string into a tuple of ints.
    E.g.
    "1.2.3" -> (1, 2, 3)
    "3.4" -> (3, 4, 0)
    """
    res = [int(num_str) for num_str in v.split(".")]
    if len(res) < 3:
        res += [0] * (3 - len(res))
    return tuple(res)
