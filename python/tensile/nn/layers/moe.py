#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

from ..common import *
from ..module import FunctionModule, Module, CompiledModule
from .mlp import MLP, MLPArgs
from .switch import SwitchModule


class SparseMoEArgs(MLPArgs):

    num_experts: int
    top_k: int = 5

    switch_mlp: MLPArgs
    router: Module.Args

    shared_expert: Optional[MLPArgs] = None
    shared_expert_gate: Optional[Module.Args] = None



@provides(MLP, 'sparse-moe')
class SparseMoE(MLP):

    __slots__ = ('num_experts', 'switch_mlp', 'router', 'shared_expert', 'shared_expert_gate', 'top_k', 'norm_topk_prob')

    num_experts: Annotated[int, field()]
    top_k: Annotated[int, field()]
    switch_mlp: Annotated[SwitchModule, field()]
    router: Annotated[FunctionModule, field()]
    shared_expert: Annotated[Optional[MLP], field()]
    shared_expert_gate: Annotated[Optional[FunctionModule], field()]

    def init_from_args(self, args: SparseMoEArgs):
        super().init_from_args(args)
        self.num_experts = args.num_experts
        self.top_k = args.top_k

        self.switch_mlp = SwitchModule.from_args(args.switch_mlp.set_defaults(
            num_experts=self.num_experts,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            kind='glu',
        ))
        self.router = FunctionModule.from_args(args.router.set_defaults(
            in_dim=self.in_dim,
            out_dim=self.num_experts,
            kind='linear',
        ))
        if shared_expert_args := args.shared_expert:
            self.shared_expert = MLP.from_args(shared_expert_args.set_defaults(
                in_dim=self.in_dim,
                hidden_dim=self.hidden_dim,
                out_dim=self.out_dim,
                kind='glu',
            ))
            if shared_expert_gate_args := args.shared_expert_gate:
                self.shared_expert_gate = FunctionModule.from_args(shared_expert_gate_args.set_defaults(
                    in_dim=self.in_dim,
                    out_dim=1,
                    kind='linear',
                ))
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def build_call(self, mode: CompiledModule.Mode, **options) -> Callable:
        k = self.top_k
        norm_topk_prob = self.norm_topk_prob
        router = self.router
        switch_mlp = self.switch_mlp

        if shared_expert := self.shared_expert:
            if shared_expert_gate := self.shared_expert_gate:
                shared_gate_activation = ten.sigmoid
                def add_shared_expert(x: Array, y: Array) -> Array:
                    shared_y = shared_expert(x)
                    shared_y = shared_gate_activation(shared_expert_gate(x)) * shared_y
                    return y + shared_y
            else:
                def add_shared_expert(x: Array, y: Array) -> Array:
                    return y + shared_expert(x)

        else:
            def add_shared_expert(x: Array, y: Array) -> Array:
                return y

        add_shared_expert = self.subinstrument('add_shared_expert', add_shared_expert, mode)

        def call(x: Array, /) -> Array:

            gates = router(x)
            gates = ten.softmax(gates, axis=-1, precise=True)

            inds = ten.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = ten.take_along_axis(gates, inds, axis=-1)
            if norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)

            y = switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

            y = add_shared_expert(x, y)

            return y

        return call

    Args = SparseMoEArgs
