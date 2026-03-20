#  Copyright (c) 2026. Richard Vermillion. All Rights Reserved.

import requests

from ..infra.types import *
from ..infra import Object, field, provides
from ..repo import Repo


class Skip:
    __slots__ = ()
    def __bool__(self):
        return False

skip = Skip()

def add(config: dict, name: str, value: Any) -> None:
    if value is not skip:
        while (dot := name.find('.')) >= 0:
            config = config.setdefault(name[:dot], {})
            name = name[dot+1:]
        config[name] = value

def get(config: dict, name: str, default: Any = None) -> Any:
    while (dot := name.find('.')) >= 0:
        conf = config.get(name[:dot])
        if conf is None:
            return default
        if isinstance(conf, dict):
            config = conf
        else:
            raise ValueError(f"Cannot get {name}")
        name = name[dot+1:]
    return config.get(name, default)



class Architecture(Object):

    __slots__ = ('state', 'value', 'stack', 'config')

    state: dict
    value: Any
    config: Annotated[dict, field(
        required=True
    )]
    stack: Annotated[list[dict], field(
        default_factory=list,
    )]

    model_kind = 'default'

    def add(self, name: str, value: Any) -> None:
        add(self.state, name, value)

    def update(self, **kwargs):
        state = self.state
        for k, v in kwargs.items():
            add(state, k, v)

    def build(self, key: str, method: str = None):
        self.push()
        build = getattr(self, method or f'build_{key}')
        build()
        self.pop(key)

    def push(self):
        self.stack.append(self.state)
        self.state = {}
        self.value = self.state

    def pop(self, key: str):
        value = self.value
        self.state = self.stack.pop()
        if value is not skip:
            self.state[key] = value
        self.value = self.state

    def get(self, name: str, default: Any = None) -> Any:
        return get(self.config, name, default)

    def skip(self):
        self.value = skip

    def replace(self, value: Any):
        self.value = value

    def convert(self, name: str, repo: str, org: str = None) -> dict:
        self.state = {}
        self.update(
            name=name,
            org=org or skip,
            kind=self.model_kind,
            repo=repo
        )
        self.build_main()
        return self.state

    def build_main(self) -> None:
        self.update(
            model_type=self.get('model_type', skip),
        )

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]
        return object.__getattribute__(self, item)

    @classmethod
    def from_config(cls, config: dict) -> 'Architecture':
        if architectures := config.get('architectures'):
           architecture = architectures[0]
           return Architecture.coerce(kind=architecture, config=config)
        raise ValueError(f"Missing 'architectures' key in config: {config}")

    @classmethod
    def from_hf(cls, name: str) -> 'Architecture':
        repo = Repo.coerce(name)
        config = repo.fetch_config()
        return cls.from_config(config)

    @classmethod
    def convert_hf(cls, repo: str) -> dict:
        arch = cls.from_hf(repo)
        org, name = repo.split('/', maxsplit=1)
        return arch.convert(name, f'hf:{org}/{name}')


class CausalLM(Architecture):

    __slots__ = ()

    lm_kind = skip
    norm_kind = 'rms'
    norm_eps = 1e-6
    norm_dims_key = 'hidden_size'
    norm_eps_key = 'rms_norm_eps'
    layer_kind = 'transformer'
    mlp_kind = 'mlp.glu'
    mlp_in_dim_key = 'hidden_size'
    mlp_hidden_dim_key = 'intermediate_size'
    mlp_out_dim_key = 'hidden_size'
    mlp_bias = False
    attention_kind = 'attention.standard'
    attention_bias = True
    position_encoder_kind = 'rope'
    lm_head_kind = 'linear'
    lm_head_bias = False


    def build_norm(self):
        self.update(
            kind=self.norm_kind,
            dims=self.get(self.norm_dims_key, skip),
            eps=self.get(self.norm_eps_key, self.norm_eps),
        )

    def build_attention_pre_norm(self):
        self.build_norm()

    def build_attention_post_norm(self):
        self.skip()

    def build_mlp_pre_norm(self):
        self.build_norm()

    def build_mlp_post_norm(self):
        self.skip()

    @property
    def num_layers(self):
        return self.get('num_hidden_layers')

    @property
    def num_attention_heads(self):
        return self.get('num_attention_heads', skip)

    @property
    def head_dim(self):
        if hidden_size := self.get('hidden_size'):
            if num_heads := self.num_attention_heads:
                return hidden_size // num_heads
        return skip

    def build_position_encoder(self):
        self.update(
            kind=self.position_encoder_kind,
            traditional=False,
            dims=self.head_dim,
            max_positions=self.get('max_position_embeddings', skip),
            original_max_positions=self.get('original_max_position_embeddings', skip),
            partial_rotary_factor=self.get('partial_rotary_factor', skip),
            base=self.get('rope_theta'),
        )
        if 'rope_scaling' in self.config:
            self.add('scaling', self.get('rope_scaling'))
        if rope_type := self.get('rope_scaling.rope_type'):
            self.add('kind', f'rope-{rope_type}')
        elif rope_type := self.get('rope_scaling.type'):
            self.add('kind', f'rope-{rope_type}')

    def build_mlp(self):
        self.update(kind='normed')
        self.build('pre_norm', 'build_mlp_pre_norm')
        self.build('body', 'build_mlp_body')
        self.build('post_norm', 'build_mlp_post_norm')

    def build_mlp_body(self):
        self.update(
            kind=self.mlp_kind,
            activation=self.get('hidden_act'),
            bias=self.get('mlp_bias', self.mlp_bias),
            in_dim=self.get(self.mlp_in_dim_key, skip),
            hidden_dim=self.get(self.mlp_hidden_dim_key),
            out_dim=self.get(self.mlp_out_dim_key, skip),
        )

    def build_attention(self):
        self.update(kind='normed')
        self.build('pre_norm', 'build_attention_pre_norm')
        self.build('body', 'build_attention_body')
        self.build('post_norm', 'build_attention_post_norm')

    def build_attention_body(self):
        self.update(
            kind=self.attention_kind,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.get('num_key_value_heads'),
            head_dim=self.head_dim,
            bias=self.get('attention_bias', self.attention_bias),
        )

        self.build('position_encoder')


    def build_layer(self):
        self.update(kind=self.layer_kind)

        self.build('attention')

        self.build('mlp')

    def build_layers(self) -> None:
        self.update(count=self.num_layers)
        self.build('_', 'build_layer')

    def build_lm(self) -> None:
        self.update(
            kind=self.lm_kind,
            vocab_size=self.get( 'vocab_size'),
            hidden_dim=self.get('hidden_size'),
            window_size=self.get('sliding_window', skip),
        )

        self.build('layers')
        self.build('norm')

    @property
    def has_lm_head(self) -> bool:
        return self.get('tie_word_embeddings', False)

    def build_lm_head(self) -> None:
        if self.has_lm_head:
            self.skip()
        else:
            self.update(
                kind=self.lm_head_kind,
                bias=self.lm_head_bias,
            )

    def build_tokenizer(self) -> None:
        self.skip()

    def build_cache(self) -> None:
        window_size = self.get('sliding_window', skip)
        if window_size is skip:
            self.skip()
        else:
            self.update(
                kind='sliding_window',
                window_size=window_size,
            )

    def build_main(self) -> None:
        super().build_main()
        self.update(
            kind='language',
            dtype=self.get('torch_dtype'),
            quantization=self.get('quantization', skip),
        )

        self.build('lm')
        self.build('lm_head')
        self.build('tokenizer')
        self.build('cache')


@provides(Architecture, 'Qwen2ForCausalLM')
class Qwen2ForCausalLM(CausalLM):

    __slots__ = ()

    def build_attention_body(self):
        super().build_attention_body()
        self.add('o_proj.kind', 'linear')
        self.add('o_proj.bias', False)
        # self.update(
        #     o_proj={
        #         'kind': 'linear',
        #         'bias': False
        #     }
        # )


@provides(Architecture, 'LlamaForCausalLM')
class LlamaForCausalLM(CausalLM):

    __slots__ = ()

    # def build_position_encoder(self):
    #     super().build_position_encoder()
    #     if self.get('rope_scaling.rope_type') == 'llama3':
    #         self.add('kind', 'rope.llama3')


@provides(Architecture, 'MistralForCausalLM')
class MistralForCausalLM(CausalLM):

    __slots__ = ()


@provides(Architecture, 'Phi3ForCausalLM')
class Phi3ForCausalLM(CausalLM):

    __slots__ = ()

    attention_kind = 'attention.fused-standard'
    mlp_kind = 'mlp.fused-glu'
