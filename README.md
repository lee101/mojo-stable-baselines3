# mojo-stable-baselines3

The rollout and replay buffers of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
ported to Mojo.

This is a port of `stable_baselines3/common/buffers.py` (SB3 v2.9.0): the Mojo
kernels in `src/ported.mojo` are a line-for-line rendering of that file's
compute-bound bodies, and `mojosb3` exposes the same classes, method names,
argument order and defaults, so an existing caller can swap the import and keep
working. The buffers hold NumPy arrays instead of PyTorch tensors, which is the
one API-level difference and is listed under [Divergences](#divergences).

The tests run the real upstream package side by side with this one and assert
element-for-element equality, not just "it did not crash".

## What is covered

| Upstream symbol | Mojo kernel | Notes |
| --- | --- | --- |
| `BaseBuffer.swap_and_flatten` | `swap_and_flatten` | dtype preserving, 1/2/4/8 byte items |
| `BaseBuffer.size` / `reset` / `extend` / `sample` | pure Python | control flow only, no kernel |
| `ReplayBuffer.add` | `replay_add` | returns the updated `pos`, as upstream's wrap does |
| `ReplayBuffer._get_samples` | `replay_get_samples` | including the `optimize_memory_usage` variant |
| `ReplayBuffer._maybe_cast_dtype` | pure Python | |
| `RolloutBuffer.compute_returns_and_advantage` | `compute_returns_and_advantage` | GAE(lambda), bit-identical to upstream below the SIMD width |
| `RolloutBuffer.add` | `rollout_add` | |
| `RolloutBuffer.get` / `_get_samples` | `swap_and_flatten` + `rollout_get_samples` | |
| `DictReplayBuffer.add` | `dict_copy_row` | one call per key, as upstream loops per key |
| `DictReplayBuffer._get_samples` | `dict_replay_gather_key` + `dict_replay_get_samples` | |
| `DictRolloutBuffer.add` | `dict_copy_row` + `rollout_add` | |
| `DictRolloutBuffer.get` / `_get_samples` | `swap_and_flatten` + `dict_flat_gather` | |
| `NStepReplayBuffer._get_samples` | `n_step_get_samples` | |
| `get_obs_shape` / `get_action_dim` | pure Python | ported from `common/preprocessing.py` |
| `ReplayBufferSamples`, `RolloutBufferSamples`, `DictReplayBufferSamples`, `DictRolloutBufferSamples` | pure Python | same field names and order |

## What is not covered

- `stable_baselines3/her/replay_buffer.py` (`HerReplayBuffer`) and
  `prioritized_replay_buffer.py` (`PrioritizedReplayBuffer`).
- The algorithms themselves (PPO, SAC, DQN, ...). Only the buffers are ported;
  they are drop-in for the buffer part of a loop.
- `VecNormalize`. `_normalize_obs` / `_normalize_reward` are ported and call
  `env.normalize_obs` / `env.normalize_reward` exactly as upstream does, but
  the `VecNormalize` class itself is not part of this port.
- GPU. The kernels are CPU, and nothing in this port was benchmarked on a GPU.
  They are too far below 2 flops per byte to want it; the reasoning is in
  [Why there is no GPU path](#why-there-is-no-gpu-path).

## Install

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles `src/capi.mojo` into
`dist/libmojo-stable-baselines3.so` with `mojo build --emit shared-lib`. The
Python package is importable through the `PYTHONPATH = "python"` activation
entry, so `pixi run python ...` is all that is needed; outside pixi, put
`python/` on `PYTHONPATH` yourself. `mojosb3._lib.build()` also rebuilds the
library on demand if the sources are newer than the `.so`.

Pinned toolchain: `mojo == 1.2.0.dev2026092605` and, for
`max.algorithm.parallelize`, `max == 26.7.0.dev2026092605`; the two ship in
lockstep and the `max` package carries the `.mojoc` files the `std` package no
longer has. Upstream `stable-baselines3 == 2.9.0` is a dependency so the parity
tests can compare against it.

## Usage

```python
import gymnasium as gym
import numpy as np
import mojosb3

observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

buffer = mojosb3.RolloutBuffer(
    8, observation_space, action_space, gae_lambda=0.95, gamma=0.99, n_envs=3
)
for _ in range(8):
    buffer.add(
        obs=np.zeros((3, 4), np.float32),
        action=np.zeros((3, 2), np.float32),
        reward=np.ones(3, np.float32),
        episode_start=np.zeros(3, np.float32),
        value=np.zeros(3, np.float32),
        log_prob=np.zeros(3, np.float32),
    )
buffer.compute_returns_and_advantage(
    last_values=np.zeros(3, np.float32), dones=np.zeros(3, np.float32)
)
for batch in buffer.get(batch_size=8):
    ...  # batch.observations, batch.actions, batch.returns, ...
```

A fuller version of the same thing, including a replay buffer, is
`examples/gae_rollout.py`:

```console
$ pixi run python examples/gae_rollout.py
collected 8 steps, full=True
observations (8, 3, 4)  actions (8, 3, 2)
advantages mean -0.900106
returns    mean -0.822588
returns == advantages + values: True
minibatch 0: observations (8, 4), actions (8, 2), returns (8,)
minibatch 1: observations (8, 4), actions (8, 2), returns (8,)
minibatch 2: observations (8, 4), actions (8, 2), returns (8,)
replay sample: observations (5, 4), rewards (5, 1), dones (5, 1)
```

## Benchmarks

Measured with `pixi run bench` (which holds a machine-wide `flock`, so a
concurrent job on the box cannot distort the numbers). Five repeats, best of.

Machine: **Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86_64**
(`os.cpu_count()` reports 72 logical cores; the process is confined to 36 of
them), Python 3.13.14, NumPy 2.5.1, PyTorch 2.13.0, stable-baselines3 2.9.0.
This is a shared box, so the millisecond figures move between runs; the ratios
and the faster/slower verdicts are stable.

| case | mojo-stable-baselines3 | stable-baselines3 | speedup | |
| --- | --- | --- | --- | --- |
| `RolloutBuffer.compute_returns_and_advantage` (2048 x 8 envs, 4d) | 0.06 ms | 16.84 ms | 266.98x | faster |
| `RolloutBuffer.compute_returns_and_advantage` (2048 x 64 envs, 4d) | 0.18 ms | 17.87 ms | 99.92x | faster |
| `RolloutBuffer.get()` one full pass (2048 x 8 envs, 4d) | 0.57 ms | 1.36 ms | 2.37x | faster |
| `ReplayBuffer.sample(4096)` (100k buffer, 4d obs) | 0.41 ms | 0.64 ms | 1.55x | faster |
| `ReplayBuffer.sample(2048)` (50k buffer, 84x84x3 uint8 obs) | 29.83 ms | 57.92 ms | 1.94x | faster |
| `ReplayBuffer.add` x 20000 (1 env, 4d obs) | 246.03 ms | 153.07 ms | 0.62x | **slower** |
| `ReplayBuffer.sample(4096)` optimize_memory_usage (100k, 4d obs) | 0.33 ms | 0.74 ms | 2.27x | faster |
| `NStepReplayBuffer.sample(4096)` n_steps=4 (100k buffer, 4d obs) | 0.67 ms | 2.03 ms | 3.03x | faster |
| `DictReplayBuffer.sample(4096)` (100k buffer, 4d + discrete) | 0.33 ms | 0.78 ms | 2.36x | faster |
| `BaseBuffer.swap_and_flatten` (200000 x 4 envs x 64d) | 36.36 ms | 119.67 ms | 3.29x | faster |
| **total** | **314.8 ms** | **370.9 ms** | **1.18x** | |

Reading the table:

- **GAE is the reason this port exists.** Upstream runs one Python loop
  iteration per rollout step, so its cost is linear in `buffer_size` with a large
  constant; the Mojo kernel walks the same recurrence with the `n_envs` axis in
  SIMD registers and no per-step interpreter overhead. 2048 steps cost 0.06 ms.
- **The batched samplers win roughly 2x.** One call replaces a handful of NumPy
  fancy indexing operations, and the per-sample loop turns into a vectorised
  gather.
- **`ReplayBuffer.add` is still about 2x slower, and it is the boundary, not the
  kernel.** The kernel itself moves a few dozen bytes for that call. What is
  left is Python: building the six row arrays, extracting twelve addresses and
  crossing into ctypes, about 12.3 us per call (246.03 ms / 20000) against
  upstream's 7.7 us of pure NumPy row assignment (153.07 ms / 20000). The
  address memoisation and the packed argument block below have taken two rounds
  out of it; the remainder is the cost of crossing into C at all, and upstream
  does not cross.
- Upstream's `_get_samples` ends in `to_torch`, so **its** sampling numbers
  include PyTorch tensor construction. That is what a caller of upstream pays,
  and it is why the samplers are a small multiple of upstream here rather than
  an order of magnitude.

### Parallelism: where it pays and where it does not

`max.algorithm.parallelize` is used by two kernels, both of which walk a loop of
independent rows and write only into their own slot:

| kernel | split over | serial branch | parallel branch |
| --- | --- | --- | --- |
| `swap_and_flatten` (200000 x 4 envs x 64d, 205 MB) | steps | 92.67 ms | 35.65 ms |
| `ReplayBuffer.sample(4096)` (4d obs, 160 KB moved) | batch rows | 0.26 ms | 5.83 ms |

Both branches are the same kernel; they were measured by rebuilding with
`PAR_MIN_BYTES` (`src/ported.mojo`) set to a value above the payload (serial)
and to 1 byte (always parallel). The first row is a 205 MB move, which pays for
the launch twice over. The second moves 160 KB, which does not pay for one:
forcing that case across the line is a 22x regression. `_workers()` -- one
worker per megabyte moved, at most 16, and a single worker (the serial loop)
below the line -- puts the crossover between the two.

The other row loops (`rollout_get_samples`, `dict_replay_gather_key`,
`dict_flat_gather`, `n_step_get_samples`) are left serial deliberately. At the
sizes the benchmark and real callers pass they all land under the threshold --
`RolloutBuffer.get()`'s full pass moves 640 KiB and `NStepReplayBuffer.sample`
160 KB -- and the second row above is what forcing them over it costs. They
keep the single-loop form; there is no parallel branch to switch on, so nothing
here is paying for a path that is never taken.

### Why there is no GPU path

There is no GPU path in this port, and no GPU code was benchmarked: the
argument for not writing one is arithmetic intensity, which can be read off the
kernels.

Every kernel here is a copy, or very nearly one:

| kernel | flops per byte moved |
| --- | --- |
| `swap_and_flatten` | 0 |
| `replay_get_samples` / `dict_*_gather` / `rollout_get_samples` | 0 |
| `replay_add` / `rollout_add` | 0 |
| `compute_returns_and_advantage` | under 1 |

All of them are far below the roughly 2 flops per byte where a GPU pays for its
transfer. On this workload the data has to be copied host to device, the
kernel launched, and the result copied back, and a kernel that does no
arithmetic cannot amortise that round trip. The honest answer for this library
is that it is a memory-movement library, and the right answer for that is a
parallel SIMD CPU path, which is what the table above measures.


## How it works

### FFI strategy

`mojo build --emit shared-lib src/capi.mojo` produces one shared library with
thirteen `abi("C")` entry points, loaded with `ctypes`.

Two constraints shape the boundary, and both are properties of `@export` rather
than of this port:

- `@export` rejects pointers and inferred origins, so **every buffer crosses as
  an `Int` address** and is re-typed inside the kernel. Float scalars cross as
  `Float64`. GAE's `gamma` and `gae_lambda` are narrowed to float32 at the edge,
  because upstream's recurrence runs in float32 and that narrowing is what
  makes the two bit-identical; `n_step_get_samples` keeps its `gamma` in float64,
  the width upstream's `gamma ** n` uses.
- `@export` rejects parametric functions, so **the dtype of an observation or
  action array crosses as an item size in bytes** (1, 2, 4 or 8). Kernels
  dispatch on it internally; a 4-byte array takes a float32 SIMD path, anything
  else takes a byte path. That is the only reason the byte path exists.

One place needs more than the item size. `rollout_get_samples` casts actions to
float32 the way upstream does, and a 2- or 8-byte action array is read through
an integer pointer of that width -- right for int16 and int64, wrong for
float16 and float64, whose bit patterns are not integers. The item size alone
cannot tell them apart, so `RolloutBuffer._get_samples` packs a float action
array of either width into float32 first and hands the kernel 4 bytes. Every
other dtype crosses as itself.

Both live in the kernel *signature*. Inside the body, the branch structure and
the arithmetic are upstream's, in upstream's order. `src/ported.mojo` can be
read next to `buffers.py`; the comment above each kernel names the upstream
statement it is executing.

The Python side (`python/mojosb3/`) is a thin, faithful mirror of
`buffers.py`: same classes, same method bodies, with the NumPy expression that
upstream writes replaced by a call into the matching kernel. `to_device(array,
copy=True)` replaces `to_torch`: it copies like `th.tensor` does and returns a
NumPy array, because the port has no PyTorch dependency.

Two things on the boundary are about the cost of crossing it, not about what the
kernels do. Both were measured on this machine, and both are worth a large
share of a `ReplayBuffer.add` call, which totals 12.3 us:

- **`addr()` is memoised on the array's identity.** `ndarray.ctypes.data` builds
  a fresh helper object on every access, which `add` needs twelve times.
  `mojosb3._lib.addr` caches the address against `id(a)` and validates the entry
  with a weak reference, so a collected array can never answer for whatever the
  allocator hands out next; the table is dropped wholesale at 8192 entries so a
  long run that passes fresh arrays every step cannot grow it. Measured
  best-of-seven over 100000 lookups: 1.72 us -> 0.31 us.
- **`sb3_replay_add` takes one argument instead of twenty-two.** Measured on the
  same loop, a ctypes call with 22 int64 arguments costs 5.80 us against 0.83 us
  for one, so the arguments were 5.0 us of a call that now costs 12.3 us in
  total. The buffer keeps a 22-element int64 block, writes the arguments into
  it in `sb3_replay_add`'s order and passes its address; `src/capi.mojo` unpacks
  it and forwards to a `replay_add` that is unchanged. The batch entry points
  keep their explicit signatures: they are called once per batch, where 5 us is
  noise, and one convention that is easy to read beats two that are fast.

### Memory layout

Every buffer array is C-contiguous with the same shape upstream uses, so
`addr(array)` is the kernel's base pointer and upstream's indexing translates
directly. The one public entry point that can be handed a view rather than a
fresh array, `BaseBuffer.swap_and_flatten`, calls `np.ascontiguousarray` on its
argument first, so a strided or transposed input gets the answer upstream gives
instead of a silently wrong one.

| array | shape | dtype |
| --- | --- | --- |
| `observations` | `(buffer_size, n_envs, *obs_shape)` | the observation space's dtype |
| `actions` | `(buffer_size, n_envs, action_dim)` | `action_space.dtype` (`_maybe_cast_dtype` for `ReplayBuffer`, which upstream also narrows; `RolloutBuffer` keeps the space's dtype and casts to float32 on the way out) |
| `rewards`, `dones`, `timeouts`, `episode_starts`, `values`, `log_probs`, `advantages`, `returns` | `(buffer_size, n_envs)` | `float32` |

A row is `n_envs * prod(obs_shape)` elements; the gathers index
`batch_inds[b] * n_envs + env_indices[b]`, which is upstream's
`arr[batch_inds, env_indices, :]` in one multiply-add.

The one storage difference: `rewards`, `values`, `episode_starts` and
`advantages` are allocated with `simd_width()` spare float32 elements past the
end, and the arrays the port hands back are views onto them. The backward GAE
sweep reads a whole float32 vector even when `n_envs` is narrower than one, and
the spare elements keep that read inside the allocation. The views behave
exactly like `np.zeros((buffer_size, n_envs))`; `simd_width()` is queried from
the library (`sb3_simd_width`) so it tracks the build.

### GAE without a loop-carried register

`compute_returns_and_advantage` is a sequential recurrence over steps, so it
cannot be reordered; what it can do is vectorise over the `n_envs` axis. The
kernel keeps the carried `last_gae_lam` in the `advantages` array it is already
writing rather than in a register: the value for step *t* is exactly the row the
sweep wrote for step *t+1*, and reading it back keeps every SIMD lane attached
to the right environment when the row is split into blocks. Two details follow
from that and are commented in the source:

- When `n_envs` is not a multiple of the SIMD width, the trailing block
  **overlaps** the previous one. Each overlapped lane is recomputed from
  identical inputs and an identical carried value, so the overlapping store
  writes the same bits.
- When `n_envs` is **narrower** than one vector, a single padded block is
  computed and stored lane by lane; storing the whole vector would clobber the
  row the sweep has already finalised.

## Divergences

Everything below is deliberate and is the complete list.

1. **Samples are NumPy, not `torch.Tensor`.** `ReplayBufferSamples` and
   `RolloutBufferSamples` keep upstream's field names and order, but the fields
   hold NumPy arrays. `to_torch` is replaced by `to_device(array, copy=True)`,
   which copies like `th.tensor` does. A caller that needs tensors can wrap with
   `th.as_tensor`.
   Inside `_get_samples` the port calls it with `copy=False`: the arrays it
   passes are the `np.empty` destinations the kernel just filled, unaliased and
   owned by nobody else, so the copy upstream's `to_torch` makes would
   duplicate the whole batch for nothing. On an Atari-sized sample that second
   copy was the larger half of the call. A caller who wants the copy still gets
   it from `to_device(array)`.
2. **`device` is accepted and ignored.** The constructors keep
   `device="auto"` in the signature and store it, but there is no device to
   resolve: the port is CPU only.
3. **`get_obs_shape` / `get_action_dim` dispatch on the space class name**
   instead of `isinstance(observation_space, spaces.Box)`, so importing
   `mojosb3` does not require gymnasium. The result is identical for every
   gymnasium space, and the tests assert that against upstream.
4. **`replay_add` and `rollout_add` take a `write_obs` flag.** The `Dict`
   subclasses store their observations key by key through `dict_copy_row` and
   then call the same packed `replay_add` entry point with `write_obs=0`, so
   the kernel skips the single-array observation store. The non-dict path
   always passes 1 and is otherwise identical to upstream.
5. **`NStepReplayBuffer._get_samples` folds the index array into the loop.**
   Upstream materialises `indices = (batch_inds[:, None] + steps) % buffer_size`
   as a `[batch, n_steps]` array; the kernel recomputes `(batch_inds[b] + j) %
   buffer_size` inline instead and materialises only `rewards_seq` in a caller
   supplied scratch buffer. The arithmetic and the ordering of every operation
   are upstream's; only the index array's storage is gone.
6. **The `Dict` per-key loops run in Python.** Upstream's
   `for key in self.observations.keys()` and its per-key dict comprehensions
   stay in Python and call a straight-line kernel once per key. This is forced
   by the toolchain, not chosen: constructing a `Pointer` from a loop-varying
   address silently miscompiles on mojo 1.2.0.dev2026092605 (see
   `MOJO_NOTES.md` 3a), so a kernel that looped over per-key addresses itself
   would be wrong. The number of keys is small and the loop is upstream's.

### Numerical parity

Every array is compared element-for-element against upstream in
`tests/test_parity.py`, and every comparison is exact (`assert_array_equal`)
with one documented exception:

- **`compute_returns_and_advantage` at or above the SIMD width** agrees with
  upstream to within 1e-6 absolute (a few float32 ulps) rather than bit for
  bit. LLVM contracts the vectorised `a + b * c` into an FMA where NumPy does
  not. Below the width, where the kernel runs one padded block, the result is
  bit-identical; the test asserts exact equality there and a tight
  `rtol=1e-5, atol=1e-6` above it. Both implementations are equidistant from a
  float64 reference, so this is rounding, not a wrong result.

## Layout

```
src/ported.mojo        kernels, in upstream source order
src/capi.mojo          the thirteen abi("C") entry points
build/build.sh         mojo build --emit shared-lib -> dist/libmojo-stable-baselines3.so
python/mojosb3/        ctypes glue, buffers.py, type_aliases.py, preprocessing.py
tests/                 pytest; parity against upstream stable-baselines3
bench/bench.py         benchmarks against upstream, prints a markdown table
examples/gae_rollout.py
```

## License

MIT, as upstream. See `LICENSE`.
