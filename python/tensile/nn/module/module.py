#  Copyright (c) 2025. Richard Vermillion. All Rights Reserved.

import textwrap
from pathlib import Path
from typing import final

from ...infrastructure import Object, Predicate, field, meta, tree, predicates
from ...infrastructure.tree import TreeEntry
from ...infrastructure.types import (
    Annotated, Any, Callable, ClassVar, Coercer, Iterable, Mapping, Optional,
    PredicateFunction, Self, Setter, Spec,
    TypeVar, Union, is_protocol
)
from ...infrastructure.util import join_str, name_function, noop, tie_call
from ...shims import Array, Shape, ten

from .args import ModuleArgs
from .context import ForwardContext
from .instrument import Instrument


optional = object()


M = TypeVar('M', bound='Module')


ModuleFilter = Callable[['Module', str, Any], bool]

EntryPredicate = Predicate[TreeEntry]


@predicates.function
def is_module(value: Any) -> bool:
    return isinstance(value, Module)

is_module_entry: EntryPredicate = tree.value_predicate(is_module)

is_parameter: Predicate = tree.is_array

is_parameter_entry: tree.TreeFilter = tree.is_array_entry

@predicates.function
def is_container(value: Any) -> bool:
    return is_module(value) or isinstance(value, (dict, list))

@predicates.function
def is_container_entry(entry: TreeEntry) -> bool:
    return is_container(entry.value)


@predicates.function
def is_trainable_parameter_entry(entry: TreeEntry) -> bool:
    if ten.is_array(entry.value):
        if parent := entry.parent:
            value = parent.value
            if isinstance(value, Module):
                return entry.step not in value._no_grad
        return True
    return False

@predicates.function
def is_leaf_module_entry(entry: TreeEntry) -> bool:
    if is_module_entry(entry):
        children = tree.flatten(entry.value.children())
        return len(children) == 0
    return False


is_leaf_module_traverser = tree.Traverser(include=is_leaf_module_entry)
is_parameter_traverser = tree.Traverser(include=is_parameter_entry)
is_trainable_parameter_traverser = tree.Traverser(include=is_trainable_parameter_entry)
is_module_traverser = tree.Traverser(include=is_module_entry)
child_traverser = tree.Traverser(include=is_module_entry, descend=~is_module_entry)



def fix_apply_fn(apply_fn: Callable, spread: bool = False, just_value: bool = False, just_path: bool = False) -> Callable[[TreeEntry], None]:
    if apply_fn is None:
        return noop
    elif just_value:
        return lambda e: apply_fn(e.value)
    elif spread:
        return lambda e: apply_fn(*e)
    elif just_path:
        return lambda e: apply_fn(e.path)
    else:
        return apply_fn


module_built = False


class ModuleField(meta.Field):

    __slots__ = ()

    slots = meta.Field.slots | set(__slots__)

    @property
    def parameter(self) -> bool:
        return bool(self.get_option('parameter', False))

    @property
    def tree(self) -> bool | None:
        return self.get_option('tree')

    def build_coerce(self, spec: Spec) -> Optional[Coercer]:
        if module_built:
            name = self.name
            required = self.required
            if cls := self.type.cls:
                if issubclass(cls, Module):
                    mcls: type[Module] = cls

                    # noinspection PyShadowingNames
                    def coerce(this: Any, val: Any) -> Any:
                        if val is None:
                            if required:
                                raise ValueError(f'Missing required field [{name}] in {this}')
                            return val
                        if isinstance(val, cls):
                            return val
                        return mcls.coerce(val)
                    return coerce
        return super().build_coerce(spec)

    def build_poke(self, spec: Spec) -> Setter:
        poke = super().build_poke(spec)
        if module_built:
            name = self.name
            if cls := self.type.cls:
                if self.tree is not False:
                    if issubclass(cls, (Module, list, dict)):
                        def new_poke(this: Module, value: Any):
                            this.set_child(name, value)
                            poke(this, value)
                        return new_poke
                    elif issubclass(cls, ten.Array):
                        if self.parameter:
                            def new_poke(this: Module, value: Any):
                                param = None if value is None else ten.parameter(value)
                                this.set_child(name, param)
                                poke(this, param)
                        else:
                            def new_poke(this: Module, value: Any):
                                this.set_child(name, value)
                                poke(this, value)
                        return new_poke
                    elif is_protocol(cls):
                        def new_poke(this: Module, value: Any):
                            if isinstance(value, Module):
                                this.set_child(name, value)
                            poke(this, value)
                        return new_poke
                elif self.parameter:
                    def new_poke(this: Module, value: Any):
                        param = None if value is None else ten.parameter(value)
                        poke(this, param)
                    return new_poke
        return poke


class ModuleMeta(meta.ObjectMeta):

    __slots__ = ()

    Field = ModuleField

    # def slot_names(self, name: str) -> Iterable[str]:
    #     return module_slot(name),


ModuleTreeValue = Union[Array, 'Module']


class Module(Object, tree.TreeNode[ModuleTreeValue]):

    __slots__ = ('args', 'call', 'training', 'instrument',
                 '_children', '_parent', '_name', '_no_grad', '_lifecycle', )

    args: Annotated[ModuleArgs, field(
        doc='Module arguments',
    )]
    call: Annotated[Callable[..., Any], field(
        doc='Module call function',
    )]
    training: Annotated[bool, field(
        doc='Whether the module is in training mode',
        default=False,
    )]
    instrument: Annotated[Optional['Instrument'], field(
        doc='Instrument object',
    )]
    _lifecycle: Annotated[Object.Lifecycle, field(
        doc='Current module lifecycle state',
        default=Object.Lifecycle.unknown,
    )]
    _children: Annotated[dict[str, Any], field(
        doc='Module child nodes',
        readonly=True,
        default_factory=dict,
    )]
    _parent: Annotated[Optional['Module'], field(
        doc='Parent module',
        default=None,
    )]
    _name: Annotated[Optional[str], field(
        doc='Name of the module',
        default=None,
        init_order=0,
    )]
    _no_grad: Annotated[set[str], field(
        doc='Set of child nodes to exclude from gradient computation',
        readonly=True,
        default_factory=set,
    )]
    keep_frozen: ClassVar[Annotated[bool, field(
        doc='Whether to keep frozen child nodes during initialization',
    )]] = False
    frozen_keys: ClassVar[Annotated[Optional[list[str]], field(
        doc='Keys of child nodes to freeze during initialization',
    )]] = None

    def set_lifecycle(self, lifecycle: Object.Lifecycle):
        self._lifecycle = lifecycle

    def get_lifecycle(self) -> Object.Lifecycle:
        return self._lifecycle

    def preinit(self, args: ModuleArgs) -> ModuleArgs:
        self.validate_args(args)
        return args

    def init(self, spec: Spec):
        super().init(spec)
        if args := spec.get('args'):
            self.init_from_args(args)

    def init_from_args(self, args: ModuleArgs):
        pass

    @property
    def input_features(self) -> int:
        return 0

    @property
    def output_features(self) -> int:
        return 0

    @property
    def input_feature_shape(self) -> Shape:
        features = self.input_features
        if features > 0:
            return features,
        return ()

    @property
    def output_feature_shape(self) -> Shape:
        features = self.output_features
        if features > 0:
            return features,
        return ()

    def postinit(self, spec: Spec):
        if self.keep_frozen:
            # Freeze this model's parameters
            self.freeze(recurse=False, keys=self.frozen_keys)
        if self.call is None:
            self._initialize_call()

    def _initialize_call(self):
        self._configure_call(self._generate_call())

    def _configure_call(self, call: Callable):
        if instrument := self.instrument:
            self.call = instrument.wrap_call(self, call, self.training)
        else:
            self.call = call

    def _generate_call(self) -> Callable:
        return self.train_call if self.training else self.eval_call

    def _should_dropout(self, child: 'Module') -> bool:
        return False

    def _with_dropout(self, mod: Callable[..., Array], p: float = 0.1, d: int = 1) -> Callable[..., Array]:
        if p == 0:
            return mod
        if isinstance(mod, Module) and not self._should_dropout(mod):
                return mod
        dropout = meta.for_qname('patchlm.nn.layers.dropout.Dropout').coerce(p=p, d=d)
        if not callable(dropout):
            raise TypeError(f'Dropout must be callable, got {dropout}')

        def call(*args, **kwargs):
            return dropout(mod(*args, **kwargs))
        return call

    def _coerce_instrument(self, spec: Any) -> Optional[Instrument]:
        if spec is None: return None
        return Instrument.coerce(spec)

    def _instrument_changed(self, instrument: 'Instrument', old_instrument: 'Instrument'):
        if instrument is not None or old_instrument is not None:
            self._configure_call(self._generate_call())

    def set_instrument(self, instrument: Instrument | Callable, compose: bool = False) -> None:
        if compose:
            instrument = Instrument.compose(self.instrument, instrument)
        self.instrument = instrument
        # self._configure_call(self._generate_call())

    def train_call(self, *args, **kwargs):
        return self.eval_call(*args, **kwargs)

    def eval_call(self, *args, **kwargs):
        raise NotImplementedError()

    @final
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def _set_training_mode(self, mode: bool) -> None:
        if self.training != mode:
            self.training = mode
            for child in self._children.values():
                if ten.is_array(child):
                    ten.require_grad(child, mode)
            self._initialize_call()
            # self.call = self.train_call if mode else self.eval_call

    def _training_changed(self, mode: bool, old_mode: bool) -> None:
        pass
        # if mode != old_mode:
        #     self.info(f'Training mode changing from {old_mode} to {mode}')

    def set_parent(self, parent: 'Module' = None, name: str = None) -> None:
        if self._parent is None or parent is None:
            self._parent = parent
        if self._name is None or name is None:
            self._name = name

    @property
    def path(self) -> str:
        nm = self._name or ''
        if self._parent is None:
            return nm
        pp = self._parent.path
        return f'{pp}.{nm}' if pp else nm

    def set_child(self, name: str, child: Any) -> None:
        def set_parent(m: Any, p: Module|None, n: str|None):
            if isinstance(m, Module):
                m.set_parent(p, n)
            elif isinstance(m, list):
                for i, c in enumerate(m):
                    set_parent(c, p, f'{name}.{i}')
            elif isinstance(m, dict):
                for k, c in m.items():
                    set_parent(c, p, f'{name}.{k}')

        old = self._children.pop(name, None)
        if old is not child:
            if old is not None:
                set_parent(old, None, None)

            if child is not None:
                set_parent(child, self, name)
                self._children[name] = child

    def train(self, mode: bool = True) -> Self:
        """Set the model in or out of training mode.

        Training mode only applies to certain layers. For example
        :obj:`Dropout` applies a random mask in training mode, but is the
        identity in evaluation mode.

        Args:
            mode (bool): Indicate if the model should be in training or
                evaluation mode. Default: ``True``.
        Returns:
            The module instance after updating the training mode.
        """

        self.apply_to_modules(lambda _, m: m._set_training_mode(mode), spread=True)

        return self

    def eval(self) -> Self:
        """Set the model to evaluation mode.

        See :func:`train`.
        """
        return self.train(False)

    def apply_to_parameters(self, apply_fn: Callable = None, *, spread: bool = False, just_value: bool = False,
                            include: PredicateFunction[TreeEntry] = None) -> Self:
        """Apply a function to all the modules in this instance (including this
        instance).

        Args:
            apply_fn (Callable): The function to apply to the modules which
                takes two parameters. The first parameter is the string path of
                the module (e.g. ``"model.layers.0.linear"``). The second
                parameter is the module object.

        Returns:
            The module instance after updating submodules.
            :param just_pass_module:
            :param spread:
            :param apply_fn:
        """
        fixed_fn = fix_apply_fn(apply_fn, spread=spread, just_value=just_value)
        if include is None:
            tree.apply(self, fixed_fn, traverser=is_module_traverser)
        else:
            tree.apply(self, fixed_fn, include=is_parameter_entry & include)
        return self

    def apply_to_modules(self, apply_fn: Callable = None, spread: bool = False, just_value: bool = False,
                         include: PredicateFunction[TreeEntry] = None) -> Self:
        """Apply a function to all the modules in this instance (including this
        instance).

        Args:
            apply_fn (Callable): The function to apply to the modules which
                takes two parameters. The first parameter is the string path of
                the module (e.g. ``"model.layers.0.linear"``). The second
                parameter is the module object.

        Returns:
            The module instance after updating submodules.
            :param just_pass_module:
            :param spread:
            :param apply_fn:
        """
        fixed_fn = fix_apply_fn(apply_fn, spread=spread, just_value=just_value)
        if include is None:
            tree.apply(self, fixed_fn, traverser=is_module_traverser)
        else:
            tree.apply(self, fixed_fn, include=is_module_entry & include)
        return self

    def update_modules(self, modules: dict, strict: bool = True) -> Self:
        for a, b in tree.join(self, modules, traverser=is_module_traverser):
            new_value = b.value
            if is_module(new_value):
                a.replace(new_value)
            elif strict:
                raise ValueError(
                    f"Received invalid type: {type(new_value).__name__}."
                )
        return self


    def leaf_modules(self) -> tree.Tree['Module']:
        """Return the submodules that do not contain other modules."""
        return tree.filter(self, traverser=is_leaf_module_traverser)

    def modules(self) -> list['Module']:
        """Return a list with all the modules in this instance.

        Returns:
            A list of :class:`mlx.nn.Module` instances.
        """
        modulelist = []
        self.apply_to_modules(lambda e: modulelist.append(e.value))
        return modulelist

    def named_modules(self) -> list[tuple[str, 'Module']]:
        """Return a list with all the modules in this instance and their name
        with dot notation.

        Returns:
            A list of tuples (str, :class:`mlx.nn.Module`).
        """
        modulelist = []
        self.apply_to_modules(lambda e: modulelist.append(tuple(e)))
        return modulelist

    def _validate_keys(self, keys: str|list[str], strict: bool) -> list[str]:
        keys = keys if isinstance(keys, list) else [keys]
        if strict:
            for k in keys:
                if k not in self:
                    raise KeyError(f"Module doesn't contain member {k}.")
        return keys

    def freeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, list[str]]] = None,
        strict: bool = False,
    ) -> Self:
        """Freeze the Module's parameters or some of them. Freezing a parameter means not
        computing gradients for it.

        This function is idempotent i.e. freezing a frozen model is a no-op.

        Example:
            For instance to only train the attention parameters from a Transformer:

            .. code-block:: python

                model = nn.Transformer()
                model.freeze()
                model.apply_to_modules(lambda k, v: v.unfreeze() if k.endswith("attention") else None)

        Args:
            recurse (bool, optional): If True then freeze the parameters of the
                submodules as well. Default: ``True``.
            keys (str or list[str], optional): If provided then only these
                parameters will be frozen otherwise all the parameters of a
                module. For instance freeze all biases by calling
                ``module.freeze(keys="bias")``.
            strict (bool, optional): If set to ``True`` validate that the passed keys exist.
                Default: ``False``.

        Returns:
            The module instance after freezing the parameters.
        """

        def _freeze_impl(path: str, mod: Module):
            local_keys = keys
            if local_keys is None:
                local_keys = tree.flatten(mod, include=is_parameter_entry,
                                          descend=~is_module_entry, force_descend=True)
                local_keys = [k for (k, v) in local_keys]

            local_keys = mod._validate_keys(local_keys, strict)
            mod._no_grad.update(local_keys)

        if recurse:
            self.apply_to_modules(_freeze_impl, spread=True)
        else:
            _freeze_impl("", self)
        return self

    def unfreeze(
        self,
        *,
        recurse: bool = True,
        keys: Optional[Union[str, list[str]]] = None,
        strict: bool = False,
    ) -> Self:

        def _unfreeze_impl(path: str, mod: Module):
            if keys is None:
                mod._no_grad.clear()

            else:
                local_keys = mod._validate_keys(keys, strict)
                mod._no_grad.difference_update(local_keys)

        if recurse:
            self.apply_to_modules(_unfreeze_impl, spread=True)
        else:
            _unfreeze_impl("", self)

        if self.keep_frozen:
            """Wrap unfreeze so that we unfreeze any layers we might contain but
            our parameters will remain frozen."""
            self.freeze(recurse=False, keys=self.frozen_keys)

        return self

    def validate_args(self, args: ModuleArgs) -> None:
        if args is None:
            raise ValueError('args cannot be None')

    def build_proj(self, in_size: int, out_size: int, bias: bool = False, name: str = 'proj') -> 'Module':
        proj_args = self.args.make_args(name, input_dims=in_size, output_dims=out_size, bias=bias)
        return self.build_proj_from_args(proj_args)

    def build_proj_from_args(self, args: ModuleArgs, kind: str = 'linear') -> 'Module':
        return Module.coerce(kind=kind, args=args)

    def items(self) -> Iterable[tuple[str, Any]]:
        return self._children.items()

    def __iter__(self):
        return iter(self._children)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._children[key]

    def parameters(self) -> tree.Tree[ten.Array]:
        """Recursively return all the :class:`mlx.core.array` members of this Module
        as a dict of dicts and lists."""
        return tree.filter(self, traverser=is_parameter_traverser)

    def trainable_parameters(self) -> tree.Tree[ten.Array]:
        """Recursively return all the non frozen :class:`mlx.core.array` members of
        this Module as a dict of dicts and lists."""
        return tree.filter(self, traverser=is_trainable_parameter_traverser)

    def children(self) -> tree.Tree['Module']:
        """Return the direct descendants of this Module instance."""
        return tree.filter(self, traverser=child_traverser, force_descend=True)

    def get_child(self, path: str) -> Any:
        return tree.get(self, path)

    def configure(self, args: ModuleArgs, **kwargs) -> Self:
        for arg, value in kwargs.items():
            if value is None:
                if args is None:
                    raise ValueError(f'Missing required arg: {arg}')
                value = args.get(arg)
                if value is None:
                    raise ValueError(f'Missing required arg: {arg}')
            elif value is optional:
                if args is not None:
                    value = args.get(arg)
            else:
                if args is not None:
                    value = args.get(arg, default=value)
            setattr(self, arg, value)
        return self

    # @contextlib.contextmanager
    # def trainer(self, trainer: 'patchlm.train.trainer.Trainer'):
    #     try:
    #         yield self
    #     finally:
    #         pass

    def _repr_args(self, **options) -> str:
        in_dim, out_dim = self.input_features, self.output_features
        extra = self._extra_structure()
        if in_dim == 0 or out_dim == 0:
            dims = ''
        else:
            dims = f'[{"*" if in_dim < 0 else in_dim} -> {"*" if out_dim < 0 else out_dim}]'
        return join_str(self.path, extra, dims)

    def build_forward_context(self, model: 'Module' = None, **kwargs) -> ForwardContext:
        if model is None: model = self
        return self.ForwardContext.coerce(model=model, **kwargs)

    def update(self, parameters: dict, strict: bool = True) -> Self:
        """Replace the parameters of this Module with the provided ones in the
        dict of dicts and lists.

        Commonly used by the optimizer to change the model to the updated
        (optimized) parameters. Also used by the :meth:`mlx.nn.value_and_grad` to set the
        tracers in the model in order to compute gradients.

        The passed in parameters dictionary need not be a full dictionary
        similar to :meth:`parameters`. Only the provided locations will be
        updated.

        Args:
            parameters (dict): A complete or partial dictionary of the modules
                parameters.
            strict (bool): If ``True`` checks that ``parameters`` is a
                subset of the module's parameters. Default: ``True``.
        Returns:
            The module instance after updating the parameters.
        """

        for a, b in tree.join(self, parameters, include=is_parameter_entry):
            new_value = b.value
            if ten.is_array(new_value):
                a.replace(new_value)
            elif strict:
                raise ValueError(
                    f"Received invalid type: {type(new_value).__name__}."
                )

        return self

    def load_weights(
        self,
        file_or_weights: str|Path|list[tuple[str, ten.Array]],
        strict: bool = True,
    ) -> 'Module':
        """
        Update the model's weights from a ``.npz``, a ``.safetensors`` file, or a list.

        Args:
            file_or_weights (str or list(tuple(str, mx.array))): The path to
                the weights ``.npz`` file (``.npz`` or ``.safetensors``) or a list
                of pairs of parameter names and arrays.
            strict (bool, optional): If ``True`` then checks that the provided
              weights exactly match the parameters of the model. Otherwise,
              only the weights actually contained in the model are loaded and
              shapes are not checked. Default: ``True``.

        Returns:
            The module instance after updating the weights.

        Example:

            .. code-block:: python

                import mlx.core as mx
                import mlx.nn as nn
                model = nn.Linear(10, 10)

                # Load from file
                model.load_weights("weights.npz")

                # Load from .safetensors file
                model.load_weights("weights.safetensors")

                # Load from list
                weights = [
                    ("weight", mx.random.uniform(shape=(10, 10))),
                    ("bias",  mx.zeros((10,))),
                ]
                model.load_weights(weights)

                # Missing weight
                weights = [
                    ("weight", mx.random.uniform(shape=(10, 10))),
                ]

                # Raises a ValueError exception
                model.load_weights(weights)

                # Ok, only updates the weight but not the bias
                model.load_weights(weights, strict=False)
        """

        if isinstance(file_or_weights, Path):
            file_or_weights = str(file_or_weights)

        if isinstance(file_or_weights, str):

            if file_or_weights.endswith(".safetensors"):
                weight_files = [file_or_weights]
            else:
                import glob

                model_path = Path(file_or_weights)
                weight_files = glob.glob(str(model_path / "model*.safetensors"))

                if not weight_files:
                    # Try weight for back-compat
                    weight_files = glob.glob(str(model_path / "weight*.safetensors"))

                if not weight_files:
                    self.error(f"No safetensors found in {model_path}")
                    raise FileNotFoundError(f"No safetensors found in {model_path}")

            weights = {}
            for wf in weight_files:
                weights.update(ten.load_tensors(wf))
            weights = list(weights.items())
        else:
            weights = file_or_weights

        if strict:
            new_weights = dict(weights)
            curr_weights = tree.flatdict(self.parameters())  #, destination={})
            if extras := (new_weights.keys() - curr_weights.keys()):
                num_extra = len(extras)
                extras = ",\n".join(sorted(extras))
                # ten.eval(new_weights, curr_weights)
                raise ValueError(
                    f"Received {num_extra} parameters not in model: \n{extras}."
                )
            if missing := (curr_weights.keys() - new_weights.keys()):
                num_missing = len(missing)
                missing = ",\n".join(sorted(missing))
                raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
            for k, v in curr_weights.items():
                v_new = new_weights[k]
                if not ten.is_array(v_new):
                    raise ValueError(
                        "Expected mx.array but received "
                        f"{type(v_new)} for parameter {k}"
                    )
                if v_new.shape != v.shape:
                    raise ValueError(
                        f"Expected shape {v.shape} but received "
                        f"shape {v_new.shape} for parameter {k}"
                    )

        if len(weights) != 0:
            self.update(tree.unflatten(weights), strict=False)
        return self

    def migrate(self, **kwargs) -> Self:
        return self

    def save_weights(self, file: str|Path):
        """
        Save the model's weights to a file. The saving method is determined by the file extension:
        - ``.npz`` will use :func:`mx.savez`
        - ``.safetensors`` will use :func:`mx.save_safetensors`
        """

        if isinstance(file, Path):
            file = str(file)

        params_dict = tree.flatdict(self.parameters())

        ten.save_tensors(file, params_dict)


    def copy(self) -> Self:
        return self.__class__(args=self.args)

    def unwrap(self, cls: type[M]) -> M:
        if isinstance(self, cls):
            return self
        raise TypeError(f'Cannot unwrap a {cls} from a {type(self)}')

    def structure(self) -> str:
        children = tree.flatten(self.children(), traverser=child_traverser)
        value = f"{type(self).__name__}({self._extra_structure()}"
        for k, v in children:
            value += "\n"
            value += textwrap.indent(f"({k}): {v.structure()}", prefix="  ")
        if children:
            value += "\n"
        value += ")"
        return value

    def _extra_structure(self) -> str:
        return ''

    @classmethod
    def from_args(cls, args: ModuleArgs = None, **kwargs) -> Self:
        if args is None:
            args = cls.Args.from_dict(kwargs)
        elif kwargs:
            args = cls.Args.combine_args(args, kwargs)

        if kind := args.kind:
            # if cls.handles_kind(kind):
            #     pass
            # else:
            return cls.meta.coerce(args=args, kind=kind)

        impl = cls.refine_implementation(args)

        return impl(args=args)

    @classmethod
    def handles_kind(cls, kind: str) -> bool:
        return kind == 'default'

    @classmethod
    def refine_implementation(cls, args: ModuleArgs) -> type[Self]:
        return cls

    @classmethod
    def provide_from(cls, spec: Mapping[str, Any], args: ModuleArgs = None, **kwargs) -> Self:
        if args is None:
            args = spec.get('args')
            if args is None:
                if spec:
                    kwargs.update(spec)
                args = cls.Args.from_dict(kwargs)
            elif isinstance(args, dict):
                args = cls.Args.from_dict(args)
        if not isinstance(args, cls.Args):
            args = cls.Args.from_args(args)

        impl = cls.refine_implementation(args)

        return impl(args=args)

    @classmethod
    def _coerce_from_str(cls, spec: str, /, **kwargs):
        return cls.from_args(cls.Args.from_dict({}), kind=spec, **kwargs)

    ForwardContext: ClassVar[type[ForwardContext]] = ForwardContext
    Args: ClassVar[type[ModuleArgs]] = ModuleArgs
    Meta = ModuleMeta


# This hack saves a dereference to the call property.
# WARNING: you won't be able to set a breakpoint in the Module.__call__ function above
# unless you comment this out. And if you override the `call` property or add a new slot for it
# you'll get weird behavior.
tie_call(Module, 'call')

# noinspection PyRedeclaration
module_built = True

meta.for_class(Module).configure_registry(
    modules='patchlm.nn.layers',
    append_kind=True,
)



class CompiledModule(Module):

    __slots__ = ('train_call', 'eval_call')

    train_call: Callable
    eval_call: Callable

    def _initialize_call(self):
        self.compile(self.training)

    def _generate_call(self) -> Callable:
        def call(*args, **kwargs):
            self.compile(self.training)
            return self.call(*args, **kwargs)
        return call

    def _instrument_changed(self, instrument: 'Instrument', old_instrument: 'Instrument'):
        if instrument is not None or old_instrument is not None:
            self.call = self._generate_call()

    def _lazy_train_call(self) -> Callable:
        self.compile(train=True)
        return self.train_call

    def _lazy_eval_call(self) -> Callable:
        self.compile(train=False)
        return self.eval_call

    def compile(self, train: bool = None):
        if train is None: train = self.training
        mode = 'train' if train else 'eval'
        self.debug('compiling {} call for {}', mode, self)
        try:
            if train:
                self.train_call = self.build_train_call()
                if self.training:
                    self._configure_call(self.train_call)
            else:
                self.eval_call = self.build_eval_call()
                if not self.training:
                    self._configure_call(self.eval_call)
            self.debug('compiled {} call for {}', mode, self)
        except Exception as e:
            self.error('failed to compile {} call for {}: {}', mode, self, e)
            raise e

    def build_call(self, train: bool = False, **options) -> Callable:
        raise NotImplementedError()

    def build_train_call(self) -> Callable:
        call = self.build_call(True)
        return name_function(call, self.path + '.train_call')

    def build_eval_call(self) -> Callable:
        call = self.build_call(False)
        return name_function(call, self.path + '.eval_call')

    @staticmethod
    def recompile(module: 'CompiledModule', value: Any, old: Any):
        if value != old and module.get_lifecycle().is_ready():
            module._configure_call(module._generate_call())


class DelegatingModule(CompiledModule):

    __slots__ = ('delegate',)

    delegate: Module

    def build_call(self, train: bool = False, **options) -> Callable:
        delegate = self.delegate
        return self.wrap_delegate_call(delegate.train_call if train else delegate.eval_call)

    def wrap_delegate_call(self, call: Callable) -> Callable:
        return call

    def validate_args(self, args: ModuleArgs):
        pass

    def unwrap(self, cls: type[M]) -> M:
        delegate = self.delegate
        if isinstance(delegate, cls):
            return delegate
        raise TypeError(f'Cannot unwrap a {cls} from a {type(delegate)}')



__all__ = [
    'DelegatingModule',
    'Module',
    'ModuleArgs',
    'CompiledModule',
]

