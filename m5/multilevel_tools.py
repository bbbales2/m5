from sortedcontainers import SortedSet
from typing import List, Callable, TypeVar, Any, Generic, Optional
from collections import Counter

import jax

X = TypeVar("X")
Y = TypeVar("Y")
G = TypeVar("G")
K = TypeVar("K")


def pytree_to_vector(pytree):
    flat_pytree, treedef = jax.tree_util.tree_flatten(pytree)

    shapes = [x.shape for x in flat_pytree]
    sizes = [x.size for x in flat_pytree]
    starts = []
    total = 0
    for size in sizes:
        starts.append(total)
        total += size

    vector = jax.numpy.concatenate([x.flatten() for x in flat_pytree])

    def vector_to_pytree(x):
        flat_pytree = [
            x[start:(start + size)].reshape(shape)
            for shape, size, start in zip(shapes, sizes, starts)
        ]
        return jax.tree_util.tree_unflatten(treedef, flat_pytree)

    return vector, vector_to_pytree


class Mapper(Generic[X, Y]):
    domains: SortedSet

    _constants: jax.numpy.ndarray
    _domain_indices: jax.numpy.ndarray

    def __init__(
            self,
            xs: List[X],
            prepare_constants: Optional[Any] = None,
            prepare_domains: Optional[Any] = None
    ):
        # Compute constants and save as pytree of ndarray
        self._constants = jax.tree_map(lambda f: jax.numpy.array([f(x) for x in xs]), prepare_constants)

        flat_prepare_domains, domain_treedef = jax.tree_util.tree_flatten(prepare_domains)

        flat_domain_values = [[f(x) for x in xs] for f in flat_prepare_domains]
        flat_domains = [SortedSet(values) for values in flat_domain_values]
        flat_domain_indices = [
            jax.numpy.array([domain.index(value) for value in values])
            for domain, values in zip(flat_domains, flat_domain_values)
        ]

        self.domains = jax.tree_util.tree_unflatten(domain_treedef, flat_domains)
        self._domain_indices = jax.tree_util.tree_unflatten(domain_treedef, flat_domain_indices)

    def __call__(
            self,
            map_function: Callable[[Any, Any], Y],
            parameters: Optional[Any] = None
    ) -> jax.numpy.ndarray:
        flat_parameters, parameters_treedef = jax.tree_util.tree_flatten(parameters)
        flat_domain_indices, domain_indices_treedef = jax.tree_util.tree_flatten(self._domain_indices)

        assert parameters_treedef == domain_indices_treedef

        indexed_parameters = jax.tree_util.tree_unflatten(
            parameters_treedef,
            (parameter[index] for parameter, index in zip(flat_parameters, flat_domain_indices))
        )

        def mapper(args):
            constants, local_parameters = args
            return map_function(constants, local_parameters)

        return jax.vmap(mapper)((self._constants, indexed_parameters))


class Scanner(Generic[X, Y, G, K]):
    groups: SortedSet[G]

    _mapper: Mapper
    _initial_value_indices: jax.numpy.ndarray
    _original_indices: jax.numpy.ndarray

    def __init__(
            self,
            xs: List[X],
            sort_by: Callable[[X], K],
            group_by: Callable[[X], G],
            prepare_constants: Optional[Any] = None,
            prepare_domains: Optional[Any] = None
    ):
        #self._constants = self._grouper.partition(self._mapper._constants)
        #self._domain_indices = self._grouper.partition(self._mapper._domain_indices)

        one_to_N = list(range(len(xs)))
        reordering = sorted(one_to_N, key = lambda i : (group_by(xs[i]), sort_by(xs[i])))
        ordered_xs = [xs[i] for i in reordering]
        
        group_sizes = Counter(group_by(x) for x in xs)
        self.groups = SortedSet(group_sizes)
        self._original_indices = jax.numpy.array(sorted(one_to_N, key = lambda i : reordering[i]))

        self._mapper = Mapper(xs=ordered_xs, prepare_constants=prepare_constants, prepare_domains=prepare_domains)

        # Build an array to figure out if we need to reset our calculation
        #  The value will be positive when this is true and be an index
        #  into the initial values array
        # The indices are stored as if arrays were indexed from 1 because there's
        #  no difference in a negative zero and a positive zero
        initial_value_index_builder = []
        for group, group_size in group_sizes.items():
            initial_value_index = self.groups.index(group) + 1
            initial_value_index_builder.append(initial_value_index)
            initial_value_index_builder.extend((group_size - 1) * [-1 * initial_value_index])
        self._initial_value_indices = jax.numpy.array(initial_value_index_builder)

    @property
    def domains(self):
        return self._mapper.domains

    def __call__(
            self,
            update: Callable[[Y, Any, Any], Y],
            initial_values: jax.numpy.ndarray,
            parameters: Optional[Any] = None
    ):
        def scanner(previous, args):
            initial_value_index, constants, domain_indices = args

            flat_parameters, parameters_treedef = jax.tree_util.tree_flatten(parameters)
            flat_domain_indices, domain_indices_treedef = jax.tree_util.tree_flatten(domain_indices)

            assert parameters_treedef == domain_indices_treedef

            local_parameters = jax.tree_util.tree_unflatten(
                parameters_treedef,
                (parameter[index] for parameter, index in zip(flat_parameters, flat_domain_indices))
            )

            previous_or_initial_condition = jax.numpy.where(
                initial_value_index > 0,
                initial_values[initial_value_index - 1],
                previous
            )

            current = update(previous_or_initial_condition, constants, local_parameters)

            return current, current

        _, output = jax.lax.scan(
                f=scanner,
                init=jax.numpy.zeros(initial_values[0].shape, dtype=initial_values.dtype),
                xs=(
                    self._initial_value_indices,
                    self._mapper._constants,
                    self._mapper._domain_indices
                )
            )

        return output[self._original_indices]
