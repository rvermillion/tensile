#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.
from .module.module import apply_to_modules
from ..infra import RootObject
from .module import Module, Instrument

from .common import *


T = TypeVar('T')


class PatchFunction(Protocol):

    def __call__(self, root: Tree) -> Any: ...


class PatchFactory(Protocol):

    def __call__(self, entry: TreeEntry[T]) -> Tree[T]: ...


class PatchError(RuntimeError):

    pass


def no_log(*args, **kwargs):
    pass


class Patch(Object, interface=True):

    __slots__ = ('name', 'reverse')

    name: Annotated[str, field()]
    reverse: Annotated[Optional['Patch'], field(
        default=None
    )]
    is_reversible: Annotated[ClassVar[bool], field(
        ignore=True,
    )] = False
    kind: Annotated[ClassVar[str], field(
        ignore=True,
    )]

    def _lazy_name(self) -> str:
        return f'{self.kind}()'

    def _lazy_reverse(self) -> Optional['Patch']:
        return None

    def _repr_args(self, **options) -> str:
        return self.name

    # noinspection PyShadowingNames
    def apply(self, root: Tree) -> None:
        raise NotImplementedError()

    @staticmethod
    def compose(*patches: 'Patch', name: str = None) -> 'Patch':
        if patches:
            if len(patches) == 1:
                return Patch.coerce(patches[0])

            return SequencePatch(*(Patch.coerce(patch) for patch in patches))
            # patches = tuple(Patch.coerce(patch) for patch in patches)
            #
            # def patch_compose(entry: TreeEntry) -> None:
            #     for patch in patches:
            #         patch(entry)
            #
            # return Patch.with_function(patch_compose, name=name or 'compose(' + ', '.join(Patch.name_for(p) for p in patches) + ')')
        raise ValueError('No patches provided!')

    @classmethod
    def _coerce_from_sequence(cls, spec: Sequence, /, **kwargs):
        return Patch.compose(*spec, **kwargs)

    @classmethod
    def _coerce_callable(cls, spec: Callable, /, **kwargs):
        return Patch.with_function(spec)

    def name_for(self: PatchFunction) -> str:
        if isinstance(self, Patch):
            return self.name
        return self.__qualname__

    _single_key_kind = True


@provides(Patch, 'sequence')
class SequencePatch(Patch):

    __slots__ = ['patches']

    patches: tuple[Patch, ...]

    def _coerce_patches(self, patches: Sequence[Patch]) -> tuple[Patch, ...]:
        if isinstance(patches, Iterable):
            patches = tuple(Patch.coerce(p) for p in patches)
            if patches: return patches
        raise ValueError(f'Could not coerce {patches} to a sequence of Patches')

    def _lazy_name(self) -> str:
        return 'sequence[' + ', '.join(Patch.name_for(p) for p in self.patches) + ']'

    def build_patch(self) -> PatchFunction:
        patches = self.patches

        def patch(entry: TreeEntry) -> None:
            for p in patches:
                p.patch(entry)

        return patch


class WherePatch(Patch):

    __slots__ = ('where',)

    where: Annotated[Predicate[TreeEntry], field(
        doc='The predicate to filter the tree entries by',
        coerce=True,
    )]

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        return {'where': self.where.describe("entry")}


@provides(Patch,'freeze-module')
class FreezeModulePatch(WherePatch):

    __slots__ = ('keys', 'recurse')

    keys: Annotated[Optional[list[str]], field(
        doc='The keys to freeze',
    )]
    recurse: Annotated[bool, field(
        doc='Whether to freeze the children of the module',
        default=False,
    )]

    def _lazy_reverse(self) -> Optional['Patch']:
        return UnfreezeModulePatch(where=self.where, keys=self.keys, recurse=self.recurse, reverse=self,
                                   name=f'-{self.name}')

    def apply(self, root: Tree) -> None:
        def freeze(e: TreeEntry):
            if isinstance(e.value, Module):
                print(f'Freezing {e.path}: {self.keys}')
                e.value.freeze(keys=self.keys, recurse=self.recurse)

        apply_to_modules(root, freeze, include=self.where)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        kwargs = super()._repr_kwargs(**options)
        kwargs.update(interface=self.interface, spec=self.spec)
        return kwargs

    is_reversible = True


@provides(Patch,'unfreeze-module')
class UnfreezeModulePatch(WherePatch):

    __slots__ = ('keys', 'recurse')

    keys: Annotated[Optional[list[str]], field(
        doc='The keys to freeze',
    )]
    recurse: Annotated[bool, field(
        doc='Whether to freeze the children of the module',
        default=False,
    )]

    def _lazy_reverse(self) -> Optional['Patch']:
        return FreezeModulePatch(where=self.where, keys=self.keys, recurse=self.recurse, reverse=self,
                                 name=f'-{self.name}')

    def apply(self, root: Tree) -> None:
        def unfreeze(e: TreeEntry):
            if isinstance(e.value, Module):
                print(f'Unfreezing {e.path}: {self.keys}')
                e.value.unfreeze(keys=self.keys, recurse=self.recurse)

        apply_to_modules(root, unfreeze, include=self.where)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        kwargs = super()._repr_kwargs(**options)
        kwargs.update(interface=self.interface, spec=self.spec)
        return kwargs

    is_reversible = True


@provides(Patch,'replace-module')
class ReplaceModulePatch(WherePatch):

    __slots__ = ('interface', 'spec')

    interface: type[Module]
    spec: Any

    def _coerce_interface(self, spec: str|type) -> type[Module]:
        if m := meta.for_spec(spec):
            if cls := m.get_class(Module):
                return cls
        raise ValueError(f'Could not find Module class for {spec}')

    def apply(self, root: Tree) -> None:
        spec = self.spec
        cls = self.interface

        def replace(e: TreeEntry):
            args = e.value.args.args_like(spec)
            new_module = cls.from_args(args)
            print(f'Replacing {e.path} with {new_module}')
            e.replace(new_module)

        apply_to_modules(root, replace, include=self.where)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        kwargs = super()._repr_kwargs(**options)
        kwargs.update(interface=self.interface, spec=self.spec)
        return kwargs


@provides(Patch, 'add-instrument')
class AddInstrumentPatch(WherePatch):

    __slots__ = ('instrument',)

    instrument: Annotated[Instrument, field(
        doc='The instrument to add to the modules',
        coerce=True,
    )]

    def _lazy_reverse(self) -> Optional['Patch']:
        return RemoveInstrumentPatch(where=self.where, instrument=self.instrument, reverse=self,
                                 name=f'-{self.name}')

    def apply(self, root: Tree) -> None:

        def add_instrument(e: TreeEntry):
            if isinstance(e.value, Module):
                print(f'Adding instrument to {e.path}: {self.instrument}')
                e.value.set_instrument(self.instrument)

        apply_to_modules(root, add_instrument, include=self.where)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        kwargs = super()._repr_kwargs(**options)
        kwargs.update(instrument=self.instrument)
        return kwargs

    is_reversible = True


@provides(Patch, 'remove-instrument')
class RemoveInstrumentPatch(WherePatch):

    __slots__ = ('instrument',)

    instrument: Annotated[Instrument, field(
        doc='The instrument to add to the modules',
        coerce=True,
    )]

    def _lazy_reverse(self) -> Optional['Patch']:
        return AddInstrumentPatch(where=self.where, instrument=self.instrument, reverse=self,
                                 name=f'-{self.name}')

    def apply(self, root: Tree) -> None:
        def remove_instrument(e: TreeEntry):
            if isinstance(e.value, Module):
                print(f'Removing instrument from {e.path}: {self.instrument}')
                e.value.remove_instrument(self.instrument)
        apply_to_modules(root, remove_instrument, include=self.where)

    def _repr_kwargs(self, **options) -> Optional[dict[Optional[str], Any]]:
        kwargs = super()._repr_kwargs(**options)
        kwargs.update(instrument=self.instrument)
        return kwargs

    is_reversible = True

