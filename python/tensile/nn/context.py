#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import contextlib

from .common import ten, Object, field, Annotated, Any, ClassVar, Optional, Self, TypeVar, TYPE_CHECKING


if TYPE_CHECKING:
    import tensile.nn.module


F = TypeVar('F', bound='ForwardContext')


class ForwardContext(Object):

    __slots__ = ('parent', 'model', 'params', 'stream', 'debugging')

    parent: Annotated[Optional['ForwardContext'], field(
        doc='The parent context.'
    )]
    model: Annotated[Optional['tensile.nn.module.Module'], field(
        doc='The model for which this context is active.'
    )]
    params: Annotated[dict[str, Any], field(
        doc='The parameters for this context.',
    )]
    stream: Annotated[Optional[ten.Stream], field(
        doc='The stream to use for this forward pass'
    )]
    debugging: Annotated[bool, field(
        doc='Whether to enable debug logging.',
        default=False,
    )]

    def _lazy_params(self) -> dict[str, Any]:
        return {}

    def start(self):
        pass

    def finish(self):
        pass

    def set_param(self, key: str, value: Any = None) -> None:
        self.params[key] = value

    def update_params(self, **params):
        self.params.update(params)

    def default_params(self, **defaults):
        params = self.params
        for k, v in defaults.items():
            if k not in params: params[k] = v

    def get_param(self, key: str, default: Any = None) -> Any:
        val = self.params.get(key, ...)
        if val is ...:
            if parent := self.parent: parent.get_param(key, default)
            return default
        return val

    def cast(self, cls: type[F]) -> Optional[F]:
        return self if isinstance(self, cls) else None

    # @classmethod
    # def coerce(cls, kind: str = None, parent: 'ForwardContext' = None, **kwargs) -> 'ForwardContext':
    #     return cls(kind=kind, parent=parent, params=kwargs)

    def first_of_type(self, cls: type[F]) -> Optional[F]:
        if isinstance(self, cls): return self
        if parent := self.parent:
            return parent.first_of_type(cls)
        return None

    def set_attr(self, name: str, value: Any) -> Any:
        try:
            setattr(self, name, value)
        except Exception as e:
            raise AttributeError(f'Failed to set attribute {name} on {self}: {e}') from e

    def push_attr(self, name: str, value: Any) -> Any:
        old = getattr(self, name)
        self.set_attr(name, value)
        return old

    def push_params(self, params: dict[str, Any]) -> dict[str, Any]:
        old = self.params
        if params:
            if old:
                self.params = old.copy()
                self.update_params()
            else:
                self.params = params
        return old

    def push_state(self, params: dict[str, Any] = None, **kwargs) -> dict[str, Any]:
        current = {}
        push_attr = self.push_attr
        for k, v in kwargs.items():
            current[k] = push_attr(k, v)
        if params:
            current['params'] = self.push_params(params)
        return current

    def pop_state(self, state: dict[str, Any] = None, /, **kwargs) -> None:
        set_attr = self.set_attr
        if state:
            for k, v in state.items():
                set_attr(k, v)
        if kwargs:
            for k, v in kwargs.items():
                set_attr(k, v)

    def push(self, **kwargs):
        return forward_context(self, **kwargs)

    def create_child(self, kind: str = None, parent: 'ForwardContext' = None, **kwargs) -> 'ForwardContext':
        if kind is None: kind = self.child_kind or default_context_kind
        if parent is not None: raise ValueError('Cannot specify a parent.')
        ctx = self._create_child(kind=kind, parent=self, **kwargs)
        return ctx

    def _create_child(self, **kwargs) -> 'ForwardContext':
        return ForwardContext.coerce(**kwargs)

    @classmethod
    def get_current(cls, allow_parent: bool = True) -> Optional[Self]:
        if forward_ctx is None:
            return None
        return forward_ctx.first_of_type(cls) if allow_parent else forward_ctx.cast(cls)

    @classmethod
    def get_training(cls) -> Any:
        ctx = cls.get_current()
        return ctx.training if ctx is not None else None

    @classmethod
    def open(cls, ctx: 'ForwardContext', **kwargs):
        return forward_context(ctx, **kwargs)

    child_kind: ClassVar[Optional[str]] = None


forward_ctx: Optional[ForwardContext] = None


default_context_kind: str = 'default'


def create_default_context(kind: str = None, **kwargs) -> ForwardContext:
    if kind is None: kind = default_context_kind
    return ForwardContext.coerce(kind=kind, **kwargs)


def get_forward_context(cls: type[F] = None) -> Optional[F]:
    return None if forward_ctx is None else forward_ctx.cast(cls)


@contextlib.contextmanager
def forward_context(ctx: ForwardContext = None, /, parent = None, **kwargs):
    if parent is not None: raise ValueError('Cannot specify a parent.')
    global forward_ctx
    old_ctx = forward_ctx
    if ctx is None:
        if old_ctx is None:
            ctx = create_default_context(parent=old_ctx, **kwargs)
        else:
            ctx = old_ctx.create_child(**kwargs)

        def begin(c: ForwardContext): c.start()
        def cleanup(c: ForwardContext): c.finish()
    else:
        def begin(c: ForwardContext): pass
        old_parent = ctx.parent
        if ctx is not old_ctx:
            ctx.parent = old_ctx
        if kwargs:
            old_state = ctx.push_state(**kwargs)
            def cleanup(c: ForwardContext): c.pop_state(old_state)
        else:
            def cleanup(c: ForwardContext): c.parent = old_parent

    try:
        forward_ctx = ctx
        begin(ctx)
        yield ctx
    finally:
        try:
            cleanup(ctx)
        except Exception as e:
            ctx.error('Failed to finish context: {}', e)
        finally:
            forward_ctx = old_ctx


@contextlib.contextmanager
def current_context():
    yield forward_context
