"""Behaviour of the ported API: names, signatures, edge cases, dtypes.

These do not need upstream to agree numerically; they check that the ported
classes behave the way the upstream ones document, and that the non-float32
dtype paths (images, discrete spaces) are wired correctly.
"""

from __future__ import annotations

import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch as th

import mojosb3
from mojosb3 import _lib
from stable_baselines3.common import buffers as upstream
from stable_baselines3.common import preprocessing as upstream_preprocessing
from conftest import as_numpy

BOX = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
ACT = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


# ------------------------------------------------------------------- API surface


@pytest.mark.parametrize(
    "name",
    ["BaseBuffer", "ReplayBuffer", "RolloutBuffer", "DictReplayBuffer", "DictRolloutBuffer", "NStepReplayBuffer"],
)
def test_every_buffer_class_is_exported(name):
    assert hasattr(mojosb3, name)
    assert hasattr(upstream, name)


@pytest.mark.parametrize("name", ["get_obs_shape", "get_action_dim"])
def test_preprocessing_helpers_match_upstream(name):
    spaces = [
        gym.spaces.Box(-1.0, 1.0, shape=(3, 4), dtype=np.float32),
        gym.spaces.Discrete(7),
        gym.spaces.MultiDiscrete([2, 3, 4]),
        gym.spaces.MultiBinary(5),
    ]
    if name == "get_obs_shape":
        spaces.append(
            gym.spaces.Dict({"a": gym.spaces.Box(-1, 1, (2,), np.float32), "b": gym.spaces.Discrete(3)})
        )
    for space in spaces:
        expected = getattr(upstream_preprocessing, name)(space)
        got = getattr(mojosb3, name)(space)
        if isinstance(expected, dict):
            assert got == expected
        else:
            assert got == expected


@pytest.mark.parametrize(
    "cls,method",
    [
        ("BaseBuffer", "swap_and_flatten"),
        ("BaseBuffer", "size"),
        ("BaseBuffer", "extend"),
        ("BaseBuffer", "reset"),
        ("BaseBuffer", "sample"),
        ("ReplayBuffer", "add"),
        ("ReplayBuffer", "sample"),
        ("ReplayBuffer", "_get_samples"),
        ("RolloutBuffer", "compute_returns_and_advantage"),
        ("RolloutBuffer", "add"),
        ("RolloutBuffer", "get"),
        ("RolloutBuffer", "_get_samples"),
    ],
)
def test_method_signatures_match_upstream(cls, method):
    ours = inspect.signature(getattr(getattr(mojosb3, cls), method))
    theirs = inspect.signature(getattr(getattr(upstream, cls), method))
    assert list(ours.parameters) == list(theirs.parameters)
    for name, parameter in theirs.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            assert ours.parameters[name].default == parameter.default


def test_buffer_constructor_keywords_match_upstream():
    for cls in ("ReplayBuffer", "RolloutBuffer", "DictReplayBuffer", "DictRolloutBuffer"):
        ours = inspect.signature(getattr(mojosb3, cls).__init__).parameters
        theirs = inspect.signature(getattr(upstream, cls).__init__).parameters
        assert list(ours) == list(theirs)
        assert ours["device"].default == theirs["device"].default
        assert ours["n_envs"].default == theirs["n_envs"].default
    roll = inspect.signature(mojosb3.RolloutBuffer.__init__).parameters
    assert roll["gae_lambda"].default == 1
    assert roll["gamma"].default == 0.99


# ------------------------------------------------------------------- buffer basics


def test_buffer_size_is_divided_by_n_envs_like_upstream():
    assert mojosb3.ReplayBuffer(100, BOX, ACT, n_envs=8).buffer_size == 12
    assert mojosb3.ReplayBuffer(4, BOX, ACT, n_envs=8).buffer_size == 1


def test_size_reset_and_full_track_position():
    buffer = mojosb3.RolloutBuffer(3, BOX, ACT, n_envs=2)
    assert buffer.size() == 0
    z = np.zeros((2, 4), np.float32)
    a = np.zeros((2, 2), np.float32)
    for step in range(3):
        buffer.add(z, a, z[:, 0], z[:, 0], z[:, 0], z[:, 0])
        assert buffer.size() == min(step + 1, 3)
    assert buffer.full
    buffer.reset()
    assert not buffer.full and buffer.pos == 0 and buffer.size() == 0


def test_extend_adds_a_whole_batch():
    n, t = 2, 4
    buffer = mojosb3.RolloutBuffer(t, BOX, ACT, n_envs=n)
    obs = np.zeros((t, n, 4), np.float32)
    act = np.zeros((t, n, 2), np.float32)
    rew = np.ones((t, n), np.float32)
    start = np.zeros((t, n), np.float32)
    value = np.zeros((t, n), np.float32)
    log_prob = np.zeros((t, n), np.float32)
    buffer.extend(obs, act, rew, start, value, log_prob)
    assert buffer.pos == t and buffer.full
    np.testing.assert_array_equal(buffer.rewards, np.ones((t, n), np.float32))


def test_optimize_memory_usage_conflicts_with_timeout_handling():
    with pytest.raises(ValueError, match="optimize_memory_usage"):
        mojosb3.ReplayBuffer(4, BOX, ACT, optimize_memory_usage=True, handle_timeout_termination=True)


def test_maybe_cast_dtype_matches_upstream():
    for dtype in (np.float64, np.float32, np.int64, np.int8):
        assert mojosb3.ReplayBuffer._maybe_cast_dtype(dtype) == upstream.ReplayBuffer._maybe_cast_dtype(dtype)


def test_n_step_rejects_optimize_memory_usage():
    with pytest.raises(NotImplementedError):
        mojosb3.NStepReplayBuffer(
            4, BOX, ACT, optimize_memory_usage=True, handle_timeout_termination=False, n_steps=2
        )


def test_dict_buffers_reject_optimize_memory_usage():
    space = gym.spaces.Dict({"a": BOX})
    with pytest.raises(AssertionError):
        mojosb3.DictReplayBuffer(4, space, ACT, optimize_memory_usage=True)


# ------------------------------------------------------------------- dtypes


def test_uint8_image_observations_round_trip_through_the_replay_buffer():
    space = gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
    n, t = 2, 5
    ours = mojosb3.ReplayBuffer(t, space, ACT, n_envs=n)
    rng = np.random.default_rng(0)
    for _ in range(t):
        obs = rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
        next_obs = rng.integers(0, 256, size=(n, 8, 8, 3), dtype=np.uint8)
        ours.add(
            obs, next_obs,
            rng.normal(size=(n, 2)).astype(np.float32),
            rng.normal(size=n).astype(np.float32),
            np.zeros(n, np.float32), [{} for _ in range(n)],
        )
    inds = np.array([0, 1])
    np.random.seed(0)
    sample = ours._get_samples(inds)
    np.random.seed(0)
    envs = np.random.randint(0, high=ours.n_envs, size=len(inds))
    assert sample.observations.dtype == np.uint8
    assert sample.observations.shape == (2, 8, 8, 3)
    for b, (index, env) in enumerate(zip(inds, envs)):
        np.testing.assert_array_equal(sample.observations[b], ours.observations[index, env])
        np.testing.assert_array_equal(sample.next_observations[b], ours.next_observations[index, env])


def test_discrete_observation_space_is_reshaped_to_one_column():
    space = gym.spaces.Discrete(4)
    n = 3
    buffer = mojosb3.ReplayBuffer(3, space, ACT, n_envs=n)
    obs = np.array([0, 1, 2])
    buffer.add(obs, obs, np.zeros((n, 2), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), [{}] * n)
    np.testing.assert_array_equal(buffer.observations[:, :, 0].ravel(), [0, 1, 2])


def test_discrete_action_space_is_cast_to_float32_on_get():
    space = gym.spaces.Discrete(5)
    n, t = 2, 3
    buffer = mojosb3.RolloutBuffer(t, BOX, space, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    for step in range(t):
        buffer.add(z, np.array([[1], [3]], np.int64), z[:, 0], z[:, 0], z[:, 0], z[:, 0])
    assert buffer.actions.dtype == np.int64
    sample = next(iter(buffer.get()))
    assert sample.actions.dtype == np.float32
    # `get` shuffles, so compare the multiset rather than the order.
    assert sorted(sample.actions.ravel().tolist()) == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]


def test_multi_discrete_action_space_keeps_its_width():
    space = gym.spaces.MultiDiscrete([2, 3])
    n, t = 2, 3
    buffer = mojosb3.RolloutBuffer(t, BOX, space, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    for _ in range(t):
        buffer.add(z, np.array([[1, 2], [2, 0]], np.int64), z[:, 0], z[:, 0], z[:, 0], z[:, 0])
    assert buffer.action_dim == 2
    sample = next(iter(buffer.get()))
    assert sample.actions.shape == (t * n, 2)


def test_float64_observations_keep_their_dtype():
    space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float64)
    n = 2
    buffer = mojosb3.ReplayBuffer(3, space, ACT, n_envs=n)
    obs = np.arange(8, dtype=np.float64).reshape(n, 4)
    buffer.add(obs, obs, np.zeros((n, 2), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), [{}] * n)
    assert buffer.observations.dtype == np.float64
    np.testing.assert_array_equal(buffer.observations[0], obs)


def test_float64_action_space_is_downcast_to_float32_on_construction():
    space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
    buffer = mojosb3.ReplayBuffer(3, BOX, space, n_envs=1)
    assert buffer.actions.dtype == np.float32


# ------------------------------------------------------------------- sampling


def test_sample_matches_upstream_under_a_fixed_seed():
    n, t = 2, 8
    pair = (
        upstream.ReplayBuffer(t, BOX, ACT, n_envs=n),
        mojosb3.ReplayBuffer(t, BOX, ACT, n_envs=n),
    )
    rng = np.random.default_rng(1)
    for _ in range(t):
        transition = (
            rng.normal(size=(n, 4)).astype(np.float32),
            rng.normal(size=(n, 4)).astype(np.float32),
            rng.normal(size=(n, 2)).astype(np.float32),
            rng.normal(size=n).astype(np.float32),
            np.zeros(n, np.float32),
            [{} for _ in range(n)],
        )
        for buffer in pair:
            buffer.add(*transition)
    np.random.seed(123)
    theirs = pair[0].sample(5)
    np.random.seed(123)
    ours = pair[1].sample(5)
    for name in theirs._fields:
        expected = getattr(theirs, name)
        got = getattr(ours, name)
        if expected is None:
            assert got is None
        else:
            np.testing.assert_array_equal(got, expected.numpy())


@pytest.mark.parametrize("dtype", [np.float64, np.float16])
def test_float_action_spaces_of_width_other_than_four_survive_the_cast(dtype):
    """`_cast_row` reads a 2- or 8-byte array through an integer pointer of the
    same width.  That is right for int16/int64 and silently wrong for float16
    and float64, so the port packs those to float32 before the kernel sees
    them.  Upstream keeps `action_space.dtype` in the buffer and casts on the
    way out, and the values have to come back intact."""
    n, t = 2, 3
    space = gym.spaces.Box(-10.0, 10.0, (3,), dtype=dtype)
    pair = (
        upstream.RolloutBuffer(t, BOX, space, n_envs=n),
        mojosb3.RolloutBuffer(t, BOX, space, n_envs=n),
    )
    assert pair[0].actions.dtype == pair[1].actions.dtype == dtype
    rng = np.random.default_rng(41)
    z = np.zeros((n, 4), np.float32)
    for _ in range(t):
        action = rng.uniform(-10.0, 10.0, size=(n, 3)).astype(dtype)
        for i, buffer in enumerate(pair):
            buffer.add(
                z,
                action,
                z[:, 0],
                z[:, 0],
                th.as_tensor(z[:, 0]) if i == 0 else z[:, 0],
                th.as_tensor(z[:, 0]) if i == 0 else z[:, 0],
            )
    for buffer in pair:
        next(iter(buffer.get()))
    inds = np.arange(t * n)
    theirs = as_numpy(pair[0]._get_samples(inds))
    ours = as_numpy(pair[1]._get_samples(inds))
    assert ours["actions"].dtype == np.float32
    np.testing.assert_allclose(ours["actions"], theirs["actions"], rtol=1e-3)


def test_sample_of_an_empty_buffer_raises_like_upstream():
    buffer = mojosb3.ReplayBuffer(4, BOX, ACT, n_envs=1)
    with pytest.raises(ValueError):
        buffer.sample(2)


def test_optimize_memory_usage_sample_never_returns_the_write_position():
    n, t = 1, 6
    buffer = mojosb3.ReplayBuffer(
        t, BOX, ACT, n_envs=n, optimize_memory_usage=True, handle_timeout_termination=False
    )
    z = np.zeros((n, 4), np.float32)
    for step in range(t):
        buffer.add(z + step, z, np.zeros((n, 2), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32), [{}])
    assert buffer.full and buffer.pos == 0
    # The transition at index `pos` overlaps two episodes, so sampling it is
    # what optimize_memory_usage exists to avoid.
    sample = buffer._get_samples(np.array([1, 2, 3]))
    np.testing.assert_array_equal(sample.observations[:, 0], buffer.observations[1:4, 0, 0])


# ------------------------------------------------------------------- misc


def test_to_device_copies_by_default():
    source = np.arange(4, dtype=np.float32)
    buffer = mojosb3.RolloutBuffer(2, BOX, ACT)
    copied = buffer.to_device(source)
    copied[0] = 99.0
    assert source[0] == 0.0
    assert buffer.to_device(source, copy=False) is not None



def test_swap_and_flatten_accepts_a_strided_view():
    """The kernel walks one flat C-contiguous run, so a view must be packed
    first; without that the address points at the wrong elements."""
    base = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    view = base[::2, :, 1:4]
    assert not view.flags.c_contiguous
    np.testing.assert_array_equal(
        mojosb3.BaseBuffer.swap_and_flatten(view),
        upstream.BaseBuffer.swap_and_flatten(view),
    )


def test_dict_replay_add_stores_rows_built_from_temporaries():
    """`DictReplayBuffer.add` packs the reward and done rows into the argument
    block as bare addresses.  If the arrays behind those addresses are not
    still alive when the kernel runs, the values stored are whatever the
    allocator left in them, so churn the heap between the adds and check every
    row.
    """
    n, t = 2, 6
    space = gym.spaces.Dict({"a": BOX})
    buffer = mojosb3.DictReplayBuffer(t * n, space, ACT, n_envs=n)
    rng = np.random.default_rng(31)
    expected = []
    for _ in range(t):
        reward = rng.normal(size=n).astype(np.float32)
        done = (rng.random(n) < 0.5).astype(np.float32)
        expected.append((reward.copy(), done.copy()))
        buffer.add(
            {"a": rng.normal(size=(n, 4)).astype(np.float32)},
            {"a": rng.normal(size=(n, 4)).astype(np.float32)},
            rng.normal(size=(n, 2)).astype(np.float32),
            reward,
            done,
            [{} for _ in range(n)],
        )
        # Force the allocator to hand the freed blocks back to someone else.
        churn = [rng.normal(size=1 << 12) for _ in range(8)]
        del churn
    np.testing.assert_array_equal(
        buffer.rewards, np.stack([row[0] for row in expected])
    )
    np.testing.assert_array_equal(buffer.dones, np.stack([row[1] for row in expected]))


def test_get_flattens_only_once():
    n, t = 2, 4
    buffer = mojosb3.RolloutBuffer(t, BOX, ACT, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    a = np.zeros((n, 2), np.float32)
    for _ in range(t):
        buffer.add(z, a, z[:, 0], z[:, 0], z[:, 0], z[:, 0])
    list(buffer.get())
    assert buffer.generator_ready
    assert buffer.observations.shape == (t * n, 4)
    first = buffer.observations
    list(buffer.get())
    assert buffer.observations is first


def test_swap_and_flatten_is_a_staticmethod_on_the_base_class():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(
        mojosb3.BaseBuffer.swap_and_flatten(arr), upstream.BaseBuffer.swap_and_flatten(arr)
    )


def test_library_exposes_the_kernels_the_glue_calls_for():
    library = _lib.lib()
    for name in _lib._SIGNATURES:
        assert hasattr(library, name)
    assert _lib.simd_width() >= 4


def test_a_replay_buffer_of_size_one_still_works():
    n = 1
    buffer = mojosb3.ReplayBuffer(1, BOX, ACT, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    buffer.add(z, z, np.zeros((n, 2), np.float32), np.array([1.0], np.float32), np.zeros(n, np.float32), [{}])
    assert buffer.full and buffer.pos == 0
    np.testing.assert_array_equal(buffer.rewards[0], [1.0])


def test_normalization_hooks_are_called_with_the_env():
    class FakeVecNormalize:
        def __init__(self):
            self.obs_calls = 0
            self.reward_calls = 0

        def normalize_obs(self, obs):
            self.obs_calls += 1
            return obs

        def normalize_reward(self, reward):
            self.reward_calls += 1
            return reward.astype(np.float32)

    env = FakeVecNormalize()
    n, t = 1, 2
    buffer = mojosb3.ReplayBuffer(t, BOX, ACT, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    a = np.zeros((n, 2), np.float32)
    for _ in range(t):
        buffer.add(z, z, a, z[:, 0], z[:, 0], [{}])
    buffer._get_samples(np.array([0]), env=env)
    # upstream normalises observations and next_observations separately
    assert env.obs_calls == 2
    assert env.reward_calls == 1


# ------------------------------------------------------------- the FFI boundary


def test_addr_returns_the_live_data_pointer_for_every_layout():
    for array in (
        np.zeros((3, 4), np.float32),
        np.zeros(5, np.uint8),
        np.zeros(6, np.int64),
        np.arange(12.0, dtype=np.float64).reshape(3, 4)[::2],
        np.zeros((2, 2), np.float32).T,
    ):
        assert _lib.addr(array) == array.ctypes.data
    read_only = np.zeros(4, np.float32)
    read_only.flags.writeable = False
    assert _lib.addr(read_only) == read_only.ctypes.data


def test_addr_never_serves_a_recycled_id():
    """`addr` memoises on `id()`, so a freed array must not answer for its
    replacement.  The weak reference in the cache entry is what makes that
    safe; this pins the behaviour down by allocating over the same id."""
    seen = []
    for _ in range(64):
        array = np.zeros(1 << 16, np.float32)
        seen.append((id(array), _lib.addr(array), array.ctypes.data))
        del array
    assert all(cached == live for _, cached, live in seen)
    # Same identity, same answer: that is the whole point of the cache.
    array = np.zeros(1 << 16, np.float32)
    assert _lib.addr(array) == _lib.addr(array) == array.ctypes.data


def test_replay_add_takes_one_packed_argument_block():
    """`sb3_replay_add` crosses as a single int64 block.

    The block is written per buffer and rewritten on every add, so a field left
    over from the previous call would show up as a stale value in the buffer.
    """
    buffer = mojosb3.ReplayBuffer(4, BOX, ACT, n_envs=2)
    assert buffer._add_args.shape == (22,)
    assert buffer._add_args_addr == _lib.addr(buffer._add_args)
    n = 2
    obs = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    action = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    reward = np.array([3.0, 4.0], np.float32)
    done = np.array([1.0, 0.0], np.float32)
    buffer.add(obs, obs, action, reward, done, [{"TimeLimit.truncated": True}, {}])
    np.testing.assert_array_equal(buffer.observations[0], obs)
    np.testing.assert_array_equal(buffer.actions[0], action)
    np.testing.assert_array_equal(buffer.rewards[0], reward)
    np.testing.assert_array_equal(buffer.dones[0], done)
    # The zero-row fast path must clear the previous call's timeout flag.
    assert buffer.timeouts[0].tolist() == [1.0, 0.0]
    buffer.add(obs + 1, obs + 1, action, reward, done, [{}, {}])
    assert buffer.timeouts[1].tolist() == [0.0, 0.0]


def test_static_add_arg_slots_match_the_live_buffer():
    """The slots `add` no longer rewrites must still describe this buffer.

    `ReplayBuffer.add` only writes the six row addresses and `pos`; the rest of
    the block is filled once in `__init__` from arrays that are never
    reassigned.  A future change that swaps a buffer array after construction
    would leave the block pointing at freed memory, and nothing else in the
    suite would notice.
    """
    buffer = mojosb3.ReplayBuffer(4, BOX, ACT, n_envs=2)
    args = buffer._add_args
    assert args[0] == _lib.addr(buffer.observations)
    assert args[1] == _lib.addr(buffer.next_observations)
    assert args[2] == _lib.addr(buffer.actions)
    assert args[3] == _lib.addr(buffer.rewards)
    assert args[4] == _lib.addr(buffer.dones)
    assert args[5] == _lib.addr(buffer.timeouts)
    assert args[12] == buffer.buffer_size
    assert args[13] == buffer.n_envs
    assert args[14] == 4
    assert args[15] == buffer.action_dim
    assert args[16] == 1
    assert args[17] == buffer.observations.dtype.itemsize
    assert args[18] == buffer.actions.dtype.itemsize
    assert args[20] == 0
    assert args[21] == 1


def test_static_add_arg_slots_follow_the_optimize_memory_usage_variant():
    """The memory-optimising variant shares one observation array, so the
    `next_observations` slot is zero and the flag is set, from `__init__`."""
    buffer = mojosb3.ReplayBuffer(
        4, BOX, ACT, n_envs=2, optimize_memory_usage=True, handle_timeout_termination=False
    )
    args = buffer._add_args
    assert args[0] == _lib.addr(buffer.observations)
    assert args[1] == 0
    assert args[20] == 1
    assert args[21] == 0
    # The shared array is what the kernel stores the next row into.
    obs = np.arange(8, dtype=np.float32).reshape(2, 4)
    action = np.arange(4, dtype=np.float32).reshape(2, 2)
    reward = np.array([3.0, 4.0], np.float32)
    done = np.array([1.0, 0.0], np.float32)
    buffer.add(obs, obs + 1, action, reward, done, [{}, {}])
    np.testing.assert_array_equal(buffer.observations[0], obs)
    np.testing.assert_array_equal(buffer.observations[1], obs + 1)


def test_to_device_is_not_called_with_a_copy_on_a_freshly_gathered_batch():
    """`_get_samples` owns the arrays it returns, so the port must not duplicate
    the batch on the way out; `to_device` still copies when a caller asks."""
    buffer = mojosb3.ReplayBuffer(4, BOX, ACT, n_envs=1)
    z = np.zeros((1, 4), np.float32)
    a = np.zeros((1, 2), np.float32)
    for _ in range(4):
        buffer.add(z, z, a, z[:, 0], z[:, 0], [{}])
    sample = buffer._get_samples(np.arange(4))
    # The batch the kernel filled is handed back as it stands: a view onto the
    # destination it already owns, not a second copy of it.  Writing through
    # the sample must not reach the buffer.
    assert sample.observations.base is not None
    sample.observations[0, 0] = 42.0
    assert buffer.observations[0, 0, 0] == 0.0
    source = np.arange(4, dtype=np.float32)
    assert buffer.to_device(source) is not source
    assert buffer.to_device(source, copy=False) is source
