# tensile

A dual-backend tensor framework for writing ML code once and running it on both [MLX](https://github.com/ml-explore/mlx) and [PyTorch](https://pytorch.org/) — including training loops, not just inference. The tensile.nn package also has its own `Module` system that supports declarative model building, patching, and instrumentation. Many common layers are implemented using this Module framework in `tensile.nn.layers`.

## What is this?

If you work across Apple Silicon (MLX) and CUDA (PyTorch), you know the pain: same math, completely different APIs for gradients, optimizer steps, device management, and evaluation semantics. Tensile sits between your code and the backend, providing a unified API that handles the hard differences so you don't have to.

Switch backends with an environment variable:

```bash
TENSILE=mlx python train.py
TENSILE=torch python train.py
```

Same model code. Same training loop. Same results (up to floating-point non-determinism between backends).

## Why not just use one framework?

Because the tradeoffs are real. MLX gives you unified memory and lazy evaluation on Apple Silicon — great for local experimentation on a MacBook Pro. PyTorch gives you CUDA, a massive ecosystem, and battle-tested distributed training. If you're prototyping locally and training at scale in the cloud, you need both. Tensile lets you write code once instead of maintaining two versions.

## What tensile actually abstracts

**Consistent API** rationalizes the API's across both backends. Infills some missing functions.

**Lazy vs. eager evaluation.** MLX is lazy — nothing computes until you force evaluation. PyTorch is eager — everything computes immediately. This isn't a cosmetic difference; it fundamentally changes how training loops, gradient computation, and memory management work. Tensile hides this so your training code doesn't need `if backend == 'mlx':` branches everywhere.

**Gradient computation.** MLX uses functional `value_and_grad`; PyTorch uses `loss.backward()` and `.grad` attributes. Tensile unifies these behind a single optimizer abstraction. You write a `train_fn` that takes a batch and returns a scalar loss. Call `optimizer.stepper(train_fn)` and you get back a step function that works on either backend.

**Optimizer step semantics.** Parameter updates, gradient zeroing, evaluation scheduling — all different between the two frameworks. Tensile's optimizer supports parameter groups with independent hyperparameters *and* independent optimizer algorithms (e.g., AdamW for one group, SGD for another), on both backends.

**Parameter tree manipulation.** Filtering, flattening, and operating on nested parameter trees — the kind of thing you need for gradient clipping, selective freezing, or routing different parameter groups to different optimizers.

**Device and stream management.** Abstracted away. You don't think about it.

## The training abstraction

The core pattern looks like this:

```python
from tensile.optim import Optimizer, TrainFunction

# Your model code — no backend-specific logic
def train_fn(batch):
    logits = model(batch.x)
    return loss_fn(logits, batch.y)

# Get a step function from the optimizer
step = optimizer.stepper(
    train_fn,
    grad_handlers=[clip_grad_norm(1.0)],  # optional hooks
)

# Training loop
for batch in dataloader:
    loss = step(batch)
```

The `stepper` pattern composes cleanly with hooks for gradient handling, step-start/end callbacks, and (on MLX) configurable evaluation frequency.

## Data-driven configuration

Tensile uses a provider/registry pattern where concrete implementations register themselves by **kind**. Model architectures, optimizers, schedulers, and other components can be instantiated from YAML or dictionary configs without your code needing to know the specific implementation class. For example, to create a model you can simply call `Model.coerce` with the dictionary read from a YAML file:

```python
from tensile.models import Model

model = Model.coerce(yaml.safe_load(model_filename))
```

To create a model with the architecture to run `qwen2.5-7b-instruct-8bit`, you can use the following YAML:

```yaml
kind: language
model:
  vocab_size: 152064
  hidden_size: 3584
  layers:
    count: 28
    _:
      kind: transformer
      input_layernorm:
        kind: rms
        eps: 1e-06
      attention:
        kind: standard
        num_attention_heads: 28
        num_key_value_heads: 4
        bias: true
        o_proj:
          bias: false
        position_encoder:
          kind: rope
          traditional: false
          max_positions: 32768
          base: 1000000.0
      mlp:
        kind: glu
        activation: silu
        bias: false
        hidden_dim: 18944
      post_attention_layernorm:
        kind: rms
        eps: 1e-06
  norm:
    kind: rms
    eps: 1e-06
lm_head:
  kind: linear
  bias: false
```

There is no need to write boilerplate code for each model type. The model is constructed with the write architecture based on the config.  For a Llama 3 model, you would change the parameters, including the position_encoder to be:

```yaml
        position_encoder:
          kind: rope.llama3
          traditional: false
          base: 500000.0
          max_positions: 131072
          scaling:
            factor: 32.0
            high_freq_factor: 4.0
            low_freq_factor: 1.0
            original_max_positions: 8192
```

And that's it, no re-implementing every part of the model to make sure that the right RoPE implementation is used.

The same pattern works for optimizers:

```python
from tensile.optim import Optimizer

optimizer = Optimizer.coerce({
    "config": {
        "kind": "adamw",
        "lr": 1e-4,
        "weight_decay": 0.01
    }
})
```

This makes experiment configuration declarative and keeps implementation details out of your training scripts.

## Instruments

The tensile `Module` system also supports first class instrumentation. You can add an instrument to any model and it has a chance to wrap the call to the model without changing the model's tree structure.  An instrument can grab activations for logging or even run inner optimization loops during a forward pass (see `tensile.extra.instrument.head_precondition` for an example). Instruments are easy to write (just extend the `Instrument` class and implement one method) and are easy to add, remove, and compose.

## Patches

The tensile `Module` system also supports patching models, which let's use change the structure or algorithm and add instrumentation declaratively.  You can declare patches in YAML files, just like everything else:

```yaml
patches:
  # Replaces every module in the tree that matches the path `**.attend` with an implementation of 
  # the `Attend` interface that has kind `sink`.
  add_sink_attend:
    replace-module:
      interface: tensile.nn.attention.attend.Attend
      spec:
        kind: sink
      where:
        tree.path: '**.attend'
  # Freezes every module in the tree that matches the path `model.layers.*` and whose step 
  # matches the lambda function (meaning only odd layers are frozen)
  freeze_odd_layers:
    freeze-module:
      where:
        - tree.path: model.layers.*
        - tree.step:
            lambda: x % 2 == 1
  # Adds an instrument to every module in the tree that matches the path `**.q_proj` that 
  # logs the activations of that projection layer every 5 batches.
  add_q_proj_logging:
    add-instrument:
      instrument:
        kind: log-activation
        schedule:
          kind: every
          attr: batch
          n: 5
      where:
        tree.path: '**.q_proj'
```

There is a complete composible predicate system that lets you pick which modules to apply patches to. And you can write you own patches to do whatever you want (just register them as a new patch kind).

## What else is in here

**tensile.infa** an infrastructure library for writing objects with smart fields that take part in a registry that uses a provides/coerce pattern.

**tensile.nn** a new base Module class that supports compiling modules and easily instrumenting them.

**NumPy backend** (partial). A subset of the API works with NumPy arrays, useful for testing and non-training workloads.

**Predicate library.** A composable predicate system used internally for parameter filtering, tree operations, and configuration logic. Supports logical composition (and, or, xor, not) with proper implication dispatch.

**Graph-mode tensors** (experimental). An alternative tensor implementation with the same API that builds computation graphs and supports event-based delta propagation — update a tensor and changed regions propagate through broadcasts, reshapes, and matmuls. Early stage, but useful for incremental computation over tensor graphs.

## Status

Tensile is under active development. It works — [patchlm](https://github.com/TODO) uses it to train and run transformer models on both backends — but the API is not yet stable and documentation is sparse. Expect rough edges.

## Requirements

- Python 3.11+
- For MLX backend: `mlx` (Apple Silicon Mac)
- For PyTorch backend: `torch`

## License

[MIT](LICENSE)

## Acknowledgments

Tensile was built to support research on composable, behavior-preserving architectural modifications to pretrained language models. If that sounds interesting, take a look at [patchlm](https://github.com/TODO).