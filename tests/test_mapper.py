from m5.multilevel_tools import Mapper
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
def mapper():
    mapper = Mapper(
        xs,
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

    return mapper


@pytest.fixture
def zero_parameters(mapper : Mapper):
    parameters = jax.tree_util.tree_map(lambda x: jax.numpy.zeros(len(x)), mapper.domains)
    return parameters


def test_constant(mapper : Mapper, zero_parameters):
    def check_time_constant(constants, parameters):
        time, time_group = constants
        return time

    output = mapper(
        check_time_constant,
        zero_parameters
    )

    assert [2, 1, 3, 1, 2, 2] == output.tolist()


def test_parameter_domains(mapper : Mapper):
    name_domain, time_domain, name_time_domain = mapper.domains

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


def test_parameter_use(mapper: Mapper):
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

    def mapper_function(constants, parameters):
        name_value, time_value, name_time_value = parameters
        return name_value + time_value + name_time_value

    output = mapper(
        mapper_function,
        (
            name_values,
            time_values,
            name_time_values
        )
    )

    assert output == pytest.approx([1.22, 2.14, 1.33, 1.11, 3.26, 2.25])
