from m5.multilevel_tools import Scanner
from sortedcontainers import SortedSet
from typing import NamedTuple

import jax
import pytest


class X(NamedTuple):
    name: str
    time: int
    group: int

xs = [
    X("a", 2, 3),
    X("b", 1, 2),
    X("a", 3, 3),
    X("a", 1, 3),
    X("c", 2, 2),
    X("b", 2, 2),
]


@pytest.fixture
def scanner():
    scanner = Scanner(
        xs,
        group_by=lambda x: x.name,
        sort_by=lambda x: x.time,
        prepare_constants=(
            lambda x: x.time,
            lambda x: (x.time, x.group)
        ),
        prepare_domains=(
            lambda x: x.name,
            lambda x: x.time,
            lambda x: (x.name, x.time)
        )
    )

    return scanner


@pytest.fixture
def zero_parameters(scanner : Scanner):
    parameters = jax.tree_util.tree_map(lambda x: jax.numpy.zeros(len(x)), scanner.domains)
    return parameters


def test_constant(scanner : Scanner):
    assert scanner.groups == SortedSet(["a", "b", "c"])


def test_constant(scanner : Scanner, zero_parameters):
    def accumulate_time_constant(previous, constants, parameters):
        time, time_group = constants
        return previous + time
    
    initial_values = jax.numpy.zeros(len(scanner.groups), dtype=int)    

    output = scanner(
        accumulate_time_constant,
        initial_values=initial_values,
        parameters=zero_parameters
    )

    assert [3, 1, 6, 1, 2, 3] == output.tolist()


def test_parameter_domains(scanner : Scanner):
    name_domain, time_domain, name_time_domain = scanner.domains

    assert name_domain == SortedSet(["a", "b", "c"])
    assert time_domain == SortedSet([1, 2, 3])
    assert name_time_domain == SortedSet([
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", 1),
        ("b", 2),
        ("c", 2)
    ])


def test_initial_values(scanner: Scanner, zero_parameters):
    def scanner_function(previous, constants, parameters):
        name_value, time_value, name_time_value = parameters
        return previous + name_value + time_value + name_time_value

    initial_values = jax.numpy.array([10.0, 20.0, 30.0])

    output = scanner(
        scanner_function,
        initial_values=initial_values,
        parameters=zero_parameters
    )

    assert output == pytest.approx([
        10.0,
        20.0,
        10.0,
        10.0,
        30.0,
        20.0
    ])


def test_parameter_use(scanner: Scanner):
    name_values = jax.numpy.array([1.0, 2.0, 3.0])
    time_values = jax.numpy.array([0.1, 0.2, 0.3])
    name_time_values = jax.numpy.array([
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06
    ])

    def scanner_function(previous, constants, parameters):
        name_value, time_value, name_time_value = parameters
        return previous + name_value + time_value + name_time_value

    initial_values = jax.numpy.array([10.0, 20.0, 30.0])

    output = scanner(
        scanner_function,
        initial_values=initial_values,
        parameters=(
            name_values,
            time_values,
            name_time_values
        )
    )

    assert output == pytest.approx([
        10.0 + 1.11 + 1.22,
        20.0 + 2.14,
        10.0 + 1.11 + 1.22 + 1.33,
        10.0 + 1.11,
        30.0 + 3.26,
        20.0 + 2.14 + 2.25
    ])
