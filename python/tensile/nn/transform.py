#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from .common import *
from .module import Module


class ModuleTransform(Protocol):

    def __call__(self, module: Module, /, **options) -> Optional[Module]: ...


class ModuleTransformCategory(Object):

    __slots__ = ('name', 'transforms', 'default_transform', 'search_mro')

    name: Annotated[str, field(
        required=True,
    )]
    transforms: Annotated[dict[type[Module], ModuleTransform], field(
        default_factory=dict,
    )]
    default_transform: Annotated[Optional[ModuleTransform], field(
        default=None,
    )]
    search_mro: Annotated[bool, field(
        default=False
    )]

    def get_transform(self, module: Module) -> Optional[ModuleTransform]:
        cls = type(module)
        if txf := self.transforms.get(cls):
            return txf
        if self.search_mro:
            for cls in cls.mro()[1:]:
                if issubclass(cls, Module):
                    if txf := self.transforms.get(cls):
                        return txf

        return self.default_transform

    def register_transform(self, module_class: type[Module], transform: ModuleTransform):
        self.transforms[module_class] = transform

    def set_default(self, default_transform: ModuleTransform):
        self.default_transform = default_transform


class ModuleTransforms(Object):

    __slots__ = ()

    categories: ClassVar[dict[str, ModuleTransformCategory]] = {}

    @staticmethod
    def get_category(name: str, create: bool = False) -> ModuleTransformCategory:
        category = ModuleTransforms.categories.get(name)
        if category is None:
            if create:
                category = ModuleTransformCategory(name=name)
                ModuleTransforms.categories[name] = category
            else:
                raise ValueError(f'No module transform category named: [{name}]')
        return category



