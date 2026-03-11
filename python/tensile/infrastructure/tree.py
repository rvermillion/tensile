# Copyright © 2023 Richard Vermillion.

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, Optional, Self, TypeVar, Union

from ..shims import ten
from .function import identity
from .predicate import Predicate, Predicates as predicates
from .transform import Transform, Transforms as transforms
from .types import PredicateLike

Array = ten.Array

T = TypeVar('T')

Tree = Union[T, list['Tree[T]'], dict[str, 'Tree[T]'], 'TreeNode[T]']


class TreeNode(Generic[T]):

    __slots__ = ()

    def __getitem__(self, key: str) -> Tree[T]:
        raise NotImplementedError()

    def __setitem__(self, key: str, value: Tree[T]):
        raise NotImplementedError()

    def items(self) -> Iterable[tuple[str, Tree[T]]]:
        raise NotImplementedError()


TreeStep = Union[str, int]


TreeFilter = Predicate['TreeEntry']
TreeAction = Callable[['TreeEntry'], Any]


# noinspection PyShadowingNames
class Traverser:

    __slots__ = ('parent_first', 'include', 'descend')

    parent_first: bool
    include: TreeFilter
    descend: TreeFilter

    def __init__(self, *, is_leaf: TreeFilter = None, include: TreeFilter = None,
                 descend: TreeFilter = None, include_intermediate: bool = False, parent_first: bool = True,
                 ) -> None:
        if include is None:
            if is_leaf is None:
                if include_intermediate:
                    self.include = predicates.always
                else:
                    self.include = is_not_container
            else:
                self.include = is_leaf
        else:
            self.include = include
        if descend is None:
            if is_leaf is not None:
                self.descend = predicates.invert(is_leaf)
            else:
                self.descend = predicates.always
        else:
            self.descend = descend
        self.parent_first = parent_first


class TreeEntry(tuple[str, T]):

    __slot__ = ('parent', 'step')

    parent: Optional['TreeEntry[T]']
    step: TreeStep

    def __new__(cls, path: str, value: Tree[T], parent: Optional['TreeEntry[T]'], step: TreeStep):
        e = tuple.__new__(cls, (path, value))
        e.parent = parent
        e.step = step
        return e

    @property
    def path(self) -> str:
        return self[0]

    @property
    def value(self) -> Tree[T]:
        return self[1]

    # node = value

    @property
    def prefix(self) -> str:
        if path := self[0]:
            return path + '.'
        return ''

    @property
    def parent_node(self) -> Optional[Tree]:
        if parent := self.parent:
            return parent.value
        return None

    # noinspection PyShadowingNames
    def get(self, path: str, *, relative: bool = False, carp: bool = True) -> Self:
        entry = self
        if path:
            try:
                s = 0
                prefix = '' if relative else self.prefix
                while True:
                    node = entry.value
                    dot = path.find('.', s)
                    if dot < 0:
                        step = path[s:]
                        partial = prefix + path
                    else:
                        step = path[s:dot]
                        partial = prefix + path[:dot]
                    s = dot+1

                    if isinstance(node, (list, tuple)):
                        i = int(step)
                        entry = TreeEntry(partial, node[i], parent=entry, step=i)
                    elif isinstance(node, (dict, TreeNode)):
                        entry = TreeEntry(partial, node[step], parent=entry, step=step)
                    else:
                        raise ValueError(f"Cannot get path {step} in non-container: {node}")
                    if dot < 0:
                        break
            except (KeyError, IndexError, ValueError) as e:
                if carp:
                    raise e
                return None
        return entry

    def replace(self, replacement: Tree[T]) -> Self:
        if replacement is self.value:
            return self
        if parent := self.parent:
            parent_node = parent.value
            if isinstance(parent_node, TreeNode):
                setattr(parent_node, self.step, replacement)
            else:
                parent_node[self.step] = replacement

            return parent.get_child(self.step)
        else:
            raise ValueError(f'Cannot replace without a parent: {self}')

    def join(self, other: Self, traverser: 'Traverser') -> Iterable[tuple[Self, Self]]:
        if traverser.parent_first and traverser.include(self):
            yield self, other
        if traverser.descend(self):
            for ochild in other.children():
                child = self.get_child(ochild.step)
                yield from child.join(ochild, traverser)
        if not traverser.parent_first and traverser.include(self):
            yield self, other

    def left_join(self, other: Optional[Self], traverser: Traverser) -> Iterable[tuple[Self, Optional[Self]]]:
        if traverser.parent_first and traverser.include(self):
            yield self, other
        if traverser.descend(self):
            for child in self.children():
                ochild = None if other is None else other.maybe_child(child.step)
                yield from child.left_join(ochild, traverser)
        if not traverser.parent_first and traverser.include(self):
            yield self, other

    def is_leaf(self) -> bool:
        return is_leaf_entry(self)
        # return not isinstance(self.node, (list, tuple, dict, TreeNode))

    def not_leaf(self) -> bool:
        return not_leaf_entry(self)
        # return isinstance(self.node, (list, tuple, dict, TreeNode))

    def is_sequence(self) -> bool:
        return is_sequence_entry(self)
        # return isinstance(self.node, (list, tuple))

    def is_mapping(self) -> bool:
        return is_mapping_entry(self)
        # return isinstance(self.node, (dict, TreeNode))

    def has_children(self) -> bool:
        return has_children(self)
        # return self.not_leaf() and bool(self.node)

    def get_child(self, step: TreeStep) -> Self:
        if self.has_children():
            return TreeEntry(f"{self.prefix}{step}", self.value[step], self, step)
        raise KeyError(f'No such child: {step}')

    def maybe_child(self, step: TreeStep) -> Optional[Self]:
        try:
            child_value = self.value[step]
            return TreeEntry(f"{self.prefix}{step}", child_value, self, step)
        except (KeyError, IndexError):
            return None

    def make_child(self, step: TreeStep, value: Tree[T]) -> Self:
        return TreeEntry(f"{self.prefix}{step}", value, self, step)

    @property
    def is_enumerable(self) -> bool:
        return is_sequence_entry(self)
        # return isinstance(self.node, (list, tuple))

    @property
    def has_items(self) -> bool:
        return is_mapping_entry(self)
        # return isinstance(self.node, (dict, TreeNode))

    def enumerate_child_nodes(self) -> Iterable[tuple[str, Tree[T]]]:
        value = self.value
        if self.is_enumerable:
            return enumerate(value)
        elif self.has_items:
            return value.items()
        else:
            return ()

    def children(self) -> Iterable[Self]:
        prefix = self.prefix
        for k, t in self.enumerate_child_nodes():
            yield TreeEntry(f"{prefix}{k}", t, self, k)

    def traverse(self, traverser: Traverser, force_descend: bool = False) -> Iterable['TreeEntry']:

        if traverser.parent_first and traverser.include(self):
            yield self

        if traverser.descend(self) or force_descend:
            for child in self.children():
                yield from child.traverse(traverser)

        if not traverser.parent_first and traverser.include(self):
            yield self

    def any(self, traverser: Traverser) -> bool:
        return traverser.include(self) or (
            traverser.descend(self) and any(c.any(traverser=traverser) for c in self.children())
        )

    def all(self, traverser: Traverser) -> bool:
        if traverser.include(self):
            return not traverser.descend(self) or all(c.all(traverser=traverser) for c in self.children())
        else:
            return traverser.descend(self) and all(c.all(traverser=traverser) for c in self.children())

    def unflat(self,
               map_fn: Optional[Callable] = identity,
               traverser: Traverser = None,
               force_descend: bool = False,
               ) -> Tree:
        if traverser.descend(self) or force_descend:
            if self.is_enumerable:
                return [c.unflat(map_fn=map_fn, traverser=traverser) for c in self.children()]
            if self.has_items:
                unflat = {}
                for c in self.children():
                    val = c.unflat(map_fn=map_fn, traverser=traverser)
                    if val is not None:
                        unflat[c.step] = val
                if unflat:
                    return unflat
        return map_fn(self) if traverser.include(self) else None

    def entries(self,
                is_leaf: Optional[Callable] = None,
                include_intermediate: bool = False,
                include: Optional[TreeFilter] = None,
                descend: Optional[TreeFilter] = None,
                parent_first: bool = True,
                traverser: Traverser = None,
                ) -> Iterable[Self]:
        if traverser is None:
            traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)

        return self.traverse(traverser)

    def with_value(self, value: Tree[T]) -> Self:
        if value is not self.value:
            return TreeEntry(self.path, value, self.parent, self.step)
        return self

    def with_prefix(self, prefix: str) -> Self:
        if prefix:
            return TreeEntry(f'{prefix}.{self.path}', self.value, self.parent, self.step)
        return self

    def __repr__(self):
        return f'({self.path!r}, {self.value!r}, {self.step})'

    # noinspection PyShadowingNames
    @classmethod
    def root(cls, tree: Tree) -> Self:
        return TreeEntry('', tree, None, '')



path_transform: Transform[TreeEntry, str] = transforms.get_attr('path')
value_transform: Transform[TreeEntry, Any] = transforms.get_attr('value')
parent_transform: Transform[TreeEntry, Optional[TreeEntry]] = transforms.get_attr('parent')
step_transform: Transform[TreeEntry, str] = transforms.get_attr('step')


def path_predicate(path_pred: PredicateLike[str]) -> TreeFilter:
    return predicates.transform(path_transform, path_pred)


def value_predicate(value_pred: PredicateLike) -> TreeFilter:
    return predicates.transform(value_transform, value_pred)


def step_predicate(step_pred: PredicateLike[str]) -> TreeFilter:
    return predicates.transform(step_transform, step_pred)


def parent_predicate(parent_pred: PredicateLike['TreeEntry']) -> TreeFilter:
    return predicates.transform(parent_transform, parent_pred)


predicates.register('tree.step', step_predicate)
predicates.register('tree.path', path_predicate)
predicates.register('tree.value', value_predicate)
predicates.register('tree.parent', parent_predicate)


def value_is_instance(*cls: type) -> TreeFilter:
    return value_predicate(predicates.is_instance(*cls))


is_array: Predicate = predicates.function(ten.is_array)
is_dtype: Predicate = predicates.function(ten.is_dtype)

is_array_entry: TreeFilter = value_predicate(is_array)


is_container: TreeFilter = value_is_instance(list, dict)

is_not_container: TreeFilter = ~is_container

is_node: TreeFilter = value_is_instance(list, dict, TreeNode)


def path_equals(path: str) -> TreeFilter:
    return path_predicate(predicates.eq(path))


def path_startswith(prefix: str) -> TreeFilter:
    return path_predicate(predicates.starts_with(prefix))


def path_contains(part: str) -> TreeFilter:
    return path_predicate(predicates.contains(part))


def path_endswith(suffix: str) -> TreeFilter:
    return path_predicate(predicates.ends_with(suffix))


def path_matches(pattern: str) -> TreeFilter:
    return path_predicate(predicates.matches(pattern))


def step_equals(path: str) -> TreeFilter:
    return step_predicate(predicates.eq(path))


def step_startswith(prefix: str) -> TreeFilter:
    return step_predicate(predicates.starts_with(prefix))


def step_contains(part: str) -> TreeFilter:
    return step_predicate(predicates.contains(part))


def step_endswith(suffix: str) -> TreeFilter:
    return step_predicate(predicates.ends_with(suffix))


def step_matches(pattern: str) -> TreeFilter:
    return step_predicate(predicates.matches(pattern))




not_leaf_entry: TreeFilter = is_node

is_leaf_entry: TreeFilter = ~not_leaf_entry

is_sequence_entry: TreeFilter = value_is_instance(list, tuple)

is_mapping_entry: TreeFilter = value_is_instance(dict, TreeNode)

has_children: TreeFilter = value_predicate(
    predicates.is_instance(list, tuple, dict, TreeNode) & predicates.is_true
)



def entry_outer_join(left: Optional[TreeEntry], right: Optional[TreeEntry], traverser: Traverser, traverse_right: bool = False) -> Iterable[tuple[Optional[TreeEntry], Optional[TreeEntry]]]:
    check = right if traverse_right else left
    if traverser.parent_first and traverser.include(check):
        yield left, right
    if traverser.descend(check):
        seen = set()
        if left is not None:
            for lchild in left.children():
                seen.add(lchild.step)
                rchild = None if right is None else right.maybe_child(lchild.step)
                yield from entry_outer_join(lchild, rchild, traverser, traverse_right)
        lchild = None
        if right is not None:
            for rchild in right.children():
                if rchild.step not in seen:
                    yield from entry_outer_join(lchild, rchild, traverser, not traverse_right)
    if not traverser.parent_first and traverser.include(check):
        yield left, right

#
# TreeEntry = TreeEntry[Array]


# noinspection PyShadowingNames
def tree_get(
    tree: Tree,
    path: str,
) -> Tree[T]:
    dot = path.find('.')
    if dot < 0:
        step = path
        remaining = None
    else:
        step = path[:dot]
        remaining = path[dot + 1:]

    if isinstance(tree, (list, tuple)):
        i = int(step)
        node = tree[i]
    elif isinstance(tree, dict):
        node = tree[step]
    else:
        raise ValueError(f"Cannot get from a non-container: {tree}")

    if remaining is None:
        return node
    else:
        return tree_get(node, remaining)


# noinspection PyShadowingNames,PyShadowingBuiltins
def map(
    fn: Callable, tree: Tree, *rest: Any, is_leaf: Optional[Callable] = None
) -> Any:
    if is_leaf is not None and is_leaf(tree):
        return fn(tree, *rest)
    elif isinstance(tree, (list, tuple)):
        seq_type = type(tree)
        return seq_type(
            map(fn, child, *rest, is_leaf=is_leaf)
            for i, child in enumerate(tree)
        )
    elif isinstance(tree, dict):
        return {
            k: map(fn, child, *rest, is_leaf=is_leaf)
            for k, child in tree.items()
        }
    else:
        return fn(tree, *rest)


def _spread_entry(fn: Callable[[str, Tree[T]], Any]) -> Callable[[TreeEntry], Any]:
    """Helper function to spread the entry path and value into the function call."""

    # noinspection PyShadowingNames
    def spread_fn(entry: TreeEntry) -> Any:
        return fn(entry.path, entry.value)
    return spread_fn


# noinspection PyShadowingNames
def map_with_path(
    fn: Callable,
    tree: Tree,
    is_leaf: Optional[Callable] = None,
    prefix: str = "",
    traverser: Optional[Traverser] = None,
    include_intermediate: bool = False,
    include: Optional[Callable] = None,
    descend: Optional[Callable] = None,
    parent_first: bool = False,
    spread: bool = False,
) -> Any:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    if spread:
        fn = _spread_entry(fn)
    return entry.unflat(fn, traverser=traverser)


# noinspection PyShadowingNames
def join(
    tree: Tree,
    other: Tree,
    is_leaf: Optional[Callable] = None,
    prefix: str = '',
    traverser: Optional[Traverser] = None,
    include_intermediate: bool = False,
    include: Optional[Callable] = None,
    descend: Optional[Callable] = None,
    parent_first: bool = False,
) -> Iterable[tuple[TreeEntry, TreeEntry]]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    oentry = TreeEntry('', other, None, '')
    return entry.join(oentry, traverser=traverser)


# noinspection PyShadowingNames
def left_join(
    tree: Tree,
    other: Tree,
    is_leaf: Optional[Callable] = None,
    prefix: str = '',
    traverser: Optional[Traverser] = None,
    include_intermediate: bool = False,
    include: Optional[Callable] = None,
    descend: Optional[Callable] = None,
    parent_first: bool = False,
) -> Iterable[tuple[TreeEntry, Optional[TreeEntry]]]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    oentry = TreeEntry('', other, None, '')
    return entry.left_join(oentry, traverser=traverser)


# noinspection PyShadowingNames
def outer_join(
    tree: Tree,
    other: Tree,
    is_leaf: Optional[Callable] = None,
    prefix: str = '',
    traverser: Optional[Traverser] = None,
    include_intermediate: bool = False,
    include: Optional[Callable] = None,
    descend: Optional[Callable] = None,
    parent_first: bool = False,
) -> Iterable[tuple[TreeEntry, Optional[TreeEntry]]]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    oentry = TreeEntry('', other, None, '')
    return outer_join(entry, oentry, traverser=traverser)


# noinspection PyShadowingNames
def entry(
    tree: Tree,
    path: str,
) -> TreeEntry:

    entry = TreeEntry('', tree, None, '')
    return entry.get(path)


# noinspection PyShadowingNames
def traverse(
    tree: Tree[T],
    prefix: str = "",
    is_leaf: Optional[Callable] = None,
    include_intermediate: bool = False,
    include: Optional[TreeFilter] = None,
    descend: Optional[TreeFilter] = None,
    parent_first: bool = True,
    force_descend: bool = False,
    traverser: Traverser = None,
) -> Iterable[TreeEntry]:
    """Flattens a Python tree to a list of key, value tuples.

    The keys are using the dot notation to define trees of arbitrary depth and
    complexity.

    .. code-block:: python

        from mlx.utils import tree_flatten

        print(tree_flatten([[[0]]]))
        # [("0.0.0", 0)]

        print(tree_flatten([[[0]]], ".hello"))
        # [("hello.0.0.0", 0)]

    .. note::
       Dictionaries should have keys that are valid Python identifiers.

    Args:
        tree (Any): The Python tree to be flattened.
        prefix (str): A prefix to use for the keys. The first character is
            always discarded.
        is_leaf (callable): An optional callable that returns True if the
            passed object is considered a leaf or False otherwise.

    Returns:
        List[Tuple[str, Any]]: The flat representation of the Python tree.
        :param force_descend:
        :param parent_first:
        :param traverser:
        :param tree:
        :param prefix:
        :param is_leaf:
        :param descend:
        :param include:
        :param include_intermediate:
    """
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    return entry.traverse(traverser, force_descend=force_descend)


def map_value(fn: Callable) -> Callable[[TreeEntry], Any]:
    """Helper function to map the node into the function call."""
    def map_fn(e: TreeEntry) -> Any:
        return fn(e.value)
    return map_fn


def replace_value(fn: Callable) -> Callable[[TreeEntry], Any]:
    """Helper function to map the node into the function call."""
    def map_fn(e: TreeEntry) -> Any:
        return e.replace(fn(e.value))
    return map_fn


# noinspection PyShadowingBuiltins,PyShadowingNames
def filter(
    tree: Tree[T],
    prefix: str = "",
    map_fn: Optional[Callable[[Any], Any]] = None,
    is_leaf: Optional[TreeFilter] = None,
    include_intermediate: bool = False,
    include: Optional[TreeFilter] = None,
    descend: Optional[TreeFilter] = None,
    parent_first: bool = True,
    force_descend: bool = False,
    traverser: Traverser = None,
) -> Tree[T]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    entry = TreeEntry(prefix, tree, None, '')
    map_fn = value_transform if map_fn is None else map_value(map_fn)
    return entry.unflat(map_fn=map_fn, traverser=traverser, force_descend=force_descend)


# noinspection PyShadowingNames
def apply(
    tree: Tree,
    apply_fn: TreeAction,
    prefix: str = "",
    is_leaf: TreeFilter = None,
    include: TreeFilter = None,
    descend: TreeFilter = None,
    include_intermediate: bool = False,
    parent_first: bool = True,
    traverser: Traverser = None,
) -> None:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)

    entry = TreeEntry(prefix, tree, None, '')
    if parent_first:
        stack = [entry]
        while stack:
            entry = stack.pop()
            if traverser.include(entry):
                apply_fn(entry)
            if traverser.descend(entry):
                stack.extend(entry.children())
    else:
        def recurse(e: TreeEntry):
            if traverser.descend(e):
                for c in e.children():
                    recurse(c)
            if traverser.include(e):
                apply_fn(e)

        recurse(entry)


# noinspection PyShadowingNames
def flatten(
    tree: Tree,
    prefix: str = "",
    is_leaf: TreeFilter = None,
    include: TreeFilter = None,
    descend: TreeFilter = None,
    include_intermediate: bool = False,
    parent_first: bool = True,
    force_descend: bool = False,
    traverser: Traverser = None,
) -> list[TreeEntry]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    return list(traverse(tree, prefix=prefix, traverser=traverser, force_descend=force_descend))


# noinspection PyShadowingNames
def flatdict(
    tree: Tree,
    prefix: str = "",
    is_leaf: TreeFilter = None,
    include: TreeFilter = None,
    descend: TreeFilter = None,
    include_intermediate: bool = False,
    parent_first: bool = True,
    traverser: Traverser = None,
) -> dict[str, Tree[T]]:
    if traverser is None:
        traverser = Traverser(is_leaf=is_leaf, include_intermediate=include_intermediate, include=include, descend=descend, parent_first=parent_first)
    # noinspection PyTypeChecker
    return dict(traverse(tree, prefix=prefix, traverser=traverser))


# noinspection PyShadowingNames
def unflatten(tree: list[tuple[str, T]]) -> Tree[T]:
    """Recreate a Python tree from its flat representation.

    .. code-block:: python

        from mlx.utils import tree_unflatten

        d = tree_unflatten([("hello.world", 42)])
        print(d)
        # {"hello": {"world": 42}}

    Args:
        tree (list[tuple[str, Any]]): The flat representation of a Python tree.
           For instance as returned by :meth:`tree_flatten`.

    Returns:
        A Python tree.
    """
    if len(tree) == 1 and tree[0][0] == "":
        return tree[0][1]

    try:
        int(tree[0][0].split(".", maxsplit=1)[0])
        is_list = True
    except ValueError:
        is_list = False

    # collect children
    children = defaultdict(list)
    for key, value in tree:
        current_idx, *next_idx = key.split(".", maxsplit=1)
        next_idx = "" if not next_idx else next_idx[0]
        children[current_idx].append((next_idx, value))

    # recursively map them to the original container
    if is_list:
        keys = sorted((int(idx), idx) for idx in children.keys())
        l = []
        for i, k in keys:
            # if i <= len(l), no {} will be appended.
            l.extend([{} for _ in range(i - len(l))])
            l.append(unflatten(children[k]))
        return l
    else:
        return {k: unflatten(v) for k, v in children.items()}


def reduce(fn, tree, *,
           initializer=None,
           is_leaf: TreeFilter = None,
           include: TreeFilter = None,
           descend: TreeFilter = None,
           include_intermediate: bool = False,
           parent_first: bool = True,
           traverser: Traverser = None):

    entries = traverse(tree, is_leaf=is_leaf, include=include, descend=descend,
                       include_intermediate=include_intermediate, parent_first=parent_first,
                       traverser=traverser)

    if initializer is None:
        accumulator = None

        for e in entries:
            accumulator = e.value if accumulator is None else fn(accumulator, e.value)

    else:
        accumulator = initializer

        for e in entries:
            accumulator = fn(accumulator, e.value)

    return accumulator


# noinspection PyUnusedLocal
def update(tree: Tree[T], updates: Tree[T], should_update: Callable[[Tree[T], Tree[T]], bool] = None) -> Tree[T]:
    if tree is None:
        return updates
    elif updates is None:
        return tree
    elif ten.is_array(tree):
        if ten.is_array(updates):
            return updates
        return tree
    elif isinstance(tree, dict):
        if isinstance(updates, dict):
            return {
                key: update(tree.get(key), updates.get(key))
                for key in tree.keys() | updates.keys()
            }
        else:
            raise ValueError('expected extract to be a dict, got:', updates)
    elif isinstance(tree, (list, tuple)):
        if isinstance(updates, (list, tuple)):
            min_len = min(len(tree), len(updates))
            seq = [update(tree[i], updates[i]) for i in range(min_len)]
            if min_len != len(updates):
                seq.extend(updates[min_len:])
            elif min_len != len(tree):
                seq.extend(tree[min_len:])
            return seq
        else:
            raise ValueError('expected extract to be a list or tuple, got:', updates)
    else:
        return updates


# noinspection PyShadowingNames
def value_extract(tree: Tree[T], extract: Any, arrays: list[Array]) -> None:
    if tree is None or extract is None:
        return
    elif ten.is_array(tree):
        arrays.append(tree)
    elif isinstance(tree, dict):
        if isinstance(extract, dict):
            dict_extract(tree, extract, arrays)
        else:
            raise ValueError('expected extract to be a dict, got:', extract)
    elif isinstance(tree, (list, tuple)):
        if isinstance(extract, (list, tuple)):
            sequence_extract(tree, extract, arrays)
        else:
            raise ValueError('expected extract to be a list or tuple, got:', extract)


# noinspection PyShadowingNames
def sequence_extract(tree: Sequence[Tree[T]], extract: Sequence[Tree[T]], arrays: list[Array]) -> None:
    for i in range(min(len(tree), len(extract))):
        value_extract(tree[i], extract[i], arrays)


# noinspection PyShadowingNames
def dict_extract(tree: dict[str, Tree[T]], extract: dict[str, Tree[T]], arrays: list[Array]) -> None:
    for key, ext in extract.items():
        value_extract(tree.get(key), ext, arrays)


# noinspection PyShadowingNames
def extract(tree: Tree, extract: Tree = None, arrays: list[Array] = None) -> list[Array]:
    extracted: list[Array] = [] if arrays is None else arrays
    if extract is None: extract = tree

    value_extract(tree, extract, extracted)
    return extracted


# noinspection PyShadowingNames
def get_entry(tree: Tree, path: str, *, relative: bool = False, carp: bool = True) -> Optional[TreeEntry]:
    return entry(tree, '').get(path, relative=relative, carp=carp)


# noinspection PyShadowingNames
def get(tree: Tree, path: str, *, relative: bool = False, carp: bool = True) -> Optional[T]:
    e = get_entry(tree, path, relative=relative, carp=carp)
    return None if e is None else e.value


ResultList = list[tuple[Array, T]]


# noinspection PyShadowingNames
def _join(tree: Tree[T], join: Optional[T], arrays: ResultList[T]) -> None:
    if tree is None or join is None:
        return
    elif ten.is_array(tree):
        arrays.append((tree, join))
    elif isinstance(tree, dict):
        if isinstance(join, dict):
            for key, ext in join.items():
                _join(tree.get(key), ext, arrays)
        else:
            raise ValueError('expected merge to be a dict, got:', join)
    elif isinstance(tree, (list, tuple)):
        if isinstance(join, (list, tuple)):
            for i in range(min(len(tree), len(join))):
                _join(tree[i], join[i], arrays)
        else:
            raise ValueError('expected merge to be a list or tuple, got:', join)


# noinspection PyShadowingNames
def join_mangle(tree: Tree, join: Tree[T] = None) -> ResultList[T]:
    if join is None: join = tree
    extracted: ResultList[T] = []
    _join(tree, join, extracted)
    return extracted


tree_entries = traverse
tree_extract = extract
tree_flatten = flatten
tree_map = map
tree_reduce = reduce
tree_unflatten = unflatten


__all__ = [
    'Tree',
    'TreeEntry',
    'tree_entries',
    'tree_extract',
    'tree_flatten',
    'tree_map',
    'tree_reduce',
    'tree_unflatten',
]