# tensile

A dual-backend tensor framework for writing ML code once and running it on both [MLX](https://github.com/ml-explore/mlx) and [PyTorch](https://pytorch.org/) — including training loops, not just inference.

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

Tensile uses a provider/registry pattern where concrete implementations register themselves by kind. Model architectures, optimizers, schedulers, and other components can be instantiated from YAML or dictionary configs without your code needing to know the specific implementation class:

```python
optimizer = Optimizer.coerce({"config": { "kind": "adamw", "lr": 1e-4, "weight_decay": 0.01}})
```

This makes experiment configuration declarative and keeps implementation details out of your training scripts.

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