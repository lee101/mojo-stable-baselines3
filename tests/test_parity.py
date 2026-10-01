"""Numerical parity against the real upstream ``stable_baselines3`` package.

Every test here fills an upstream buffer and a mojosb3 buffer with the same
data and asserts the arrays are equal.  Where upstream returns
``torch.Tensor`` the test compares ``.numpy()``; that is the only adaptation
and it is stated in the README.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th

import mojosb3
from mojosb3._lib import simd_width
from stable_baselines3.common import buffers as upstream

from conftest import as_numpy


def fill_pair(buffers, data, last_values, dones):
    obs, action, reward, episode_start, value, log_prob = data
    for step in range(len(obs)):
        for i, buffer in enumerate(buffers):
            buffer.add(
                obs[step],
                action[step],
                reward[step],
                episode_start[step],
                th.as_tensor(value[step]) if i == 0 else value[step],
                th.as_tensor(log_prob[step]) if i == 0 else log_prob[step],
            )
    for i, buffer in enumerate(buffers):
        if i == 0:
            buffer.compute_returns_and_advantage(th.as_tensor(last_values), dones)
        else:
            buffer.compute_returns_and_advantage(last_values, dones)


# ------------------------------------------------------------- compute_returns_and_advantage


@pytest.mark.parametrize("gamma,gae_lambda", [(0.99, 0.95), (0.9, 1.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)])
def test_gae_matches_upstream(obs_space, act_space, rollout_data, last_step, gamma, gae_lambda):
    t, n = rollout_data[0].shape[:2]
    last_values, dones = last_step
    pair = (
        upstream.RolloutBuffer(t, obs_space, act_space, gae_lambda=gae_lambda, gamma=gamma, n_envs=n),
        mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=gae_lambda, gamma=gamma, n_envs=n),
    )
    fill_pair(pair, rollout_data, last_values, dones)

    np.testing.assert_array_equal(pair[0].advantages, pair[1].advantages)
    np.testing.assert_array_equal(pair[0].returns, pair[1].returns)


@pytest.mark.parametrize("n_envs", [1, 2, 3, 7, 8, 9, 15, 16, 17, 32])
def test_gae_matches_upstream_for_every_simd_lane_count(obs_space, act_space, n_envs):
    """`n_envs` straddles the SIMD width in every direction: below, at, above.

    Below the width the kernel is bit-identical to upstream.  At or above it the
    vector loop is FMA-contracted by LLVM where NumPy is not, which moves the
    result by a few float32 ulps; the bound here is deliberately tight and the
    measured error is well under 1e-6.
    """
    t = 5
    rng = np.random.default_rng(n_envs)
    data = (
        rng.normal(size=(t, n_envs, 4)).astype(np.float32),
        rng.normal(size=(t, n_envs, 2)).astype(np.float32),
        rng.normal(size=(t, n_envs)).astype(np.float32),
        (rng.random((t, n_envs)) < 0.4).astype(np.float32),
        rng.normal(size=(t, n_envs)).astype(np.float32),
        rng.normal(size=(t, n_envs)).astype(np.float32),
    )
    last_values = rng.normal(size=n_envs).astype(np.float32)
    dones = (rng.random(n_envs) < 0.5).astype(np.float32)
    pair = (
        upstream.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n_envs),
        mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n_envs),
    )
    fill_pair(pair, data, last_values, dones)
    width = simd_width()
    if n_envs < width:
        np.testing.assert_array_equal(pair[0].advantages, pair[1].advantages)
        np.testing.assert_array_equal(pair[0].returns, pair[1].returns)
    else:
        np.testing.assert_allclose(pair[1].advantages, pair[0].advantages, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pair[1].returns, pair[0].returns, rtol=1e-5, atol=1e-6)


def test_gae_matches_hand_computed_vector():
    """A two-step rollout worked through by hand, gamma=0.5, lambda=1.

    step 1: delta = 2 + 0.5*0.1 - 0.25            = 1.8
    step 0: delta = 1 + 0.5*0.25 - 0.5           = 0.625
            A[0]  = 0.625 + 0.5 * 1.8            = 1.525
    """
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    buffer = mojosb3.RolloutBuffer(2, obs_space, act_space, gae_lambda=1.0, gamma=0.5, n_envs=1)
    z = np.zeros((1, 1), dtype=np.float32)
    buffer.add(z, z, np.array([1.0], np.float32), z, np.array([0.5], np.float32), z)
    buffer.add(z, z, np.array([2.0], np.float32), z, np.array([0.25], np.float32), z)
    buffer.compute_returns_and_advantage(np.array([0.1], np.float32), np.zeros(1, np.float32))

    np.testing.assert_allclose(buffer.advantages.ravel(), [1.525, 1.8], rtol=1e-6)
    np.testing.assert_allclose(buffer.returns.ravel(), [2.025, 2.05], rtol=1e-6)


def test_gae_td0_is_one_step(obs_space, act_space):
    """gae_lambda=0 is TD(0): r_t + gamma * v(s_{t+1}) - v(s_t), no carry."""
    n, t = 1, 3
    buffer = mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.0, gamma=0.9, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    action = np.zeros((n, 2), np.float32)
    rewards = np.array([1.0, 2.0, 3.0], np.float32).reshape(t, n)
    values = np.array([0.5, 0.25, 0.125], np.float32).reshape(t, n)
    for step in range(t):
        buffer.add(z, action, rewards[step], z, values[step], z)
    buffer.compute_returns_and_advantage(np.array([0.0], np.float32), np.zeros(n, np.float32))

    expected = np.array([1.0 + 0.9 * 0.25 - 0.5, 2.0 + 0.9 * 0.125 - 0.25, 3.0 - 0.125], np.float32)
    np.testing.assert_allclose(buffer.advantages.ravel(), expected, rtol=1e-6)


def test_gae_terminal_step_uses_dones(obs_space, act_space):
    """A done on the last step must cut the bootstrap, not just scale it."""
    n, t = 1, 2
    buffer = mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=1.0, gamma=0.9, n_envs=n)
    z = np.zeros((n, 4), np.float32)
    action = np.zeros((n, 2), np.float32)
    buffer.add(z, action, np.array([1.0], np.float32), z, np.array([0.0], np.float32), z)
    buffer.add(z, action, np.array([1.0], np.float32), z, np.array([0.0], np.float32), z)
    buffer.compute_returns_and_advantage(np.array([10.0], np.float32), np.array([1.0], np.float32))
    # last step: delta = 1 + 0.9*10*0 - 0 = 1
    # step 0:    delta = 1 + 0.9*0*1 - 0 = 1;  A[0] = 1 + 0.9*1*1 = 1.9
    np.testing.assert_allclose(buffer.advantages.ravel(), [1.9, 1.0], rtol=1e-6)


# ------------------------------------------------------------------------ RolloutBuffer


def test_rollout_add_matches_upstream(obs_space, act_space, rollout_data):
    t, n = rollout_data[0].shape[:2]
    pair = (
        upstream.RolloutBuffer(t, obs_space, act_space, n_envs=n),
        mojosb3.RolloutBuffer(t, obs_space, act_space, n_envs=n),
    )
    fill_pair(pair, rollout_data, np.zeros(n, np.float32), np.zeros(n, np.float32))
    for name in ("observations", "actions", "rewards", "episode_starts", "values", "log_probs"):
        np.testing.assert_array_equal(getattr(pair[0], name), getattr(pair[1], name))


def test_rollout_get_samples_matches_upstream(obs_space, act_space, rollout_data, last_step):
    t, n = rollout_data[0].shape[:2]
    pair = (
        upstream.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
        mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
    )
    fill_pair(pair, rollout_data, *last_step)
    next(iter(pair[0].get()))
    next(iter(pair[1].get()))

    inds = np.random.default_rng(3).permutation(t * n)
    theirs = as_numpy(pair[0]._get_samples(inds))
    ours = as_numpy(pair[1]._get_samples(inds))
    for name in theirs:
        if ours[name] is None:
            assert theirs[name] is None
            continue
        assert ours[name].shape == theirs[name].shape
        np.testing.assert_array_equal(ours[name], theirs[name])


def test_rollout_get_minibatches_partition_the_buffer(obs_space, act_space, rollout_data, last_step):
    t, n = rollout_data[0].shape[:2]
    buffer = mojosb3.RolloutBuffer(t, obs_space, act_space, n_envs=n)
    fill_pair((buffer,), rollout_data, *last_step)
    batches = list(buffer.get(batch_size=5))
    assert [len(b.observations) for b in batches] == [5, 5, 5, 5, 4]
    assert sum(len(b.returns) for b in batches) == t * n


# ----------------------------------------------------------------------- swap_and_flatten


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4, 2, 3), np.float32),
        ((4, 2, 3), np.float64),
        ((4, 2, 3), np.int64),
        ((4, 2, 3), np.uint8),
        ((4, 2), np.float32),
        ((4, 2, 16), np.float32),
        ((4, 2, 17), np.float32),
        ((3, 5, 1), np.float32),
    ],
)
def test_swap_and_flatten_matches_upstream(shape, dtype):
    rng = np.random.default_rng(sum(shape))
    arr = (rng.random(shape) * 100).astype(dtype)
    np.testing.assert_array_equal(
        mojosb3.BaseBuffer.swap_and_flatten(arr),
        upstream.BaseBuffer.swap_and_flatten(arr),
    )


def test_swap_and_flatten_preserves_the_env_major_order(obs_space, act_space):
    n, t = 3, 4
    arr = np.arange(t * n, dtype=np.float32).reshape(t, n)
    out = mojosb3.BaseBuffer.swap_and_flatten(arr)
    for step in range(t):
        for env in range(n):
            assert out[env * t + step] == arr[step, env]


# ------------------------------------------------------------------------ ReplayBuffer


def _fill_replay(buffers, rng, obs_shape, act_size, n_envs):
    """One `add` of the same transition into every buffer in `buffers`.

    `obs_shape` is a feature count or the observation space's own shape, so the
    same helper fills a flat `(4,)` box and a `(16, 16, 3)` one.
    """
    shape = (n_envs, obs_shape) if isinstance(obs_shape, int) else obs_shape
    data = (
        rng.normal(size=shape).astype(np.float32),
        rng.normal(size=shape).astype(np.float32),
        rng.normal(size=(n_envs, act_size)).astype(np.float32),
        rng.normal(size=n_envs).astype(np.float32),
        (rng.random(n_envs) < 0.3).astype(np.float32),
        [{"TimeLimit.truncated": bool(x)} for x in rng.random(n_envs) < 0.3],
    )
    for buffer in buffers:
        buffer.add(*data)
    return data


def test_replay_add_matches_upstream(obs_space, act_space):
    n_envs, obs_size, act_size = 3, 4, 2
    pair = (
        upstream.ReplayBuffer(12, obs_space, act_space, n_envs=n_envs),
        mojosb3.ReplayBuffer(12, obs_space, act_space, n_envs=n_envs),
    )
    rng = np.random.default_rng(5)
    for _ in range(12):
        _fill_replay(pair, rng, obs_size, act_size, n_envs)
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        np.testing.assert_array_equal(getattr(pair[0], name), getattr(pair[1], name))
    assert pair[0].pos == pair[1].pos
    assert pair[0].full == pair[1].full
    assert pair[0].size() == pair[1].size()


def test_replay_get_samples_matches_upstream(obs_space, act_space):
    n_envs, obs_size, act_size = 3, 4, 2
    pair = (
        upstream.ReplayBuffer(12, obs_space, act_space, n_envs=n_envs),
        mojosb3.ReplayBuffer(12, obs_space, act_space, n_envs=n_envs),
    )
    rng = np.random.default_rng(6)
    for _ in range(12):
        _fill_replay(pair, rng, obs_size, act_size, n_envs)

    inds = np.arange(pair[0].size())
    np.random.seed(0)
    theirs = as_numpy(pair[0]._get_samples(inds))
    np.random.seed(0)
    ours = as_numpy(pair[1]._get_samples(inds))
    for name in theirs:
        if ours[name] is None:
            assert theirs[name] is None
            continue
        assert ours[name].shape == theirs[name].shape
        np.testing.assert_array_equal(ours[name], theirs[name])


def test_replay_optimize_memory_usage_matches_upstream(obs_space, act_space):
    n_envs, obs_size, act_size = 3, 4, 2
    pair = (
        upstream.ReplayBuffer(
            12, obs_space, act_space, n_envs=n_envs,
            optimize_memory_usage=True, handle_timeout_termination=False,
        ),
        mojosb3.ReplayBuffer(
            12, obs_space, act_space, n_envs=n_envs,
            optimize_memory_usage=True, handle_timeout_termination=False,
        ),
    )
    rng = np.random.default_rng(7)
    for _ in range(12):
        _fill_replay(pair, rng, obs_size, act_size, n_envs)
    np.testing.assert_array_equal(pair[0].observations, pair[1].observations)

    inds = np.arange(pair[0].size())
    np.random.seed(1)
    theirs = as_numpy(pair[0]._get_samples(inds))
    np.random.seed(1)
    ours = as_numpy(pair[1]._get_samples(inds))
    for name in theirs:
        np.testing.assert_array_equal(ours[name], theirs[name])


def test_replay_wraparound_keeps_the_write_position(obs_space, act_space):
    buffer = mojosb3.ReplayBuffer(6, obs_space, act_space, n_envs=2)
    rng = np.random.default_rng(8)
    seen = []
    for step in range(10):
        _fill_replay((buffer,), rng, 4, 2, 2)
        seen.append((buffer.pos, buffer.full))
    # buffer_size is 6 // 2 == 3, so pos cycles 0,1,2 and `full` latches on the
    # third add, exactly as upstream does.
    assert seen == [(1, False), (2, False), (0, True), (1, True), (2, True),
                    (0, True), (1, True), (2, True), (0, True), (1, True)]


def test_replay_timeout_flag_is_recorded(obs_space, act_space):
    buffer = mojosb3.ReplayBuffer(4, obs_space, act_space, n_envs=2)
    obs = np.zeros((2, 4), np.float32)
    buffer.add(obs, obs, obs[:, :2], np.zeros(2, np.float32), np.ones(2, np.float32),
               [{"TimeLimit.truncated": True}, {}])
    assert buffer.timeouts[0].tolist() == [1.0, 0.0]
    assert buffer.dones[0].tolist() == [1.0, 1.0]


# -------------------------------------------------------------------- Dict buffers


@pytest.fixture
def dict_space():
    return gym.spaces.Dict(
        {
            "vec": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            "disc": gym.spaces.Discrete(5),
        }
    )


def _fill_dict_rollout(buffers, t, n, rng, last_values, dones):
    for step in range(t):
        obs = {
            "vec": rng.normal(size=(n, 3)).astype(np.float32),
            "disc": rng.integers(0, 5, size=n),
        }
        action = rng.normal(size=(n, 2)).astype(np.float32)
        reward = rng.normal(size=n).astype(np.float32)
        start = (rng.random(n) < 0.25).astype(np.float32)
        value = rng.normal(size=n).astype(np.float32)
        log_prob = rng.normal(size=n).astype(np.float32)
        for i, buffer in enumerate(buffers):
            buffer.add(
                obs, action, reward, start,
                th.as_tensor(value) if i == 0 else value,
                th.as_tensor(log_prob) if i == 0 else log_prob,
            )
    for i, buffer in enumerate(buffers):
        if i == 0:
            buffer.compute_returns_and_advantage(th.as_tensor(last_values), dones)
        else:
            buffer.compute_returns_and_advantage(last_values, dones)


def test_dict_rollout_matches_upstream(dict_space, act_space):
    t, n = 6, 3
    rng = np.random.default_rng(9)
    last_values = rng.normal(size=n).astype(np.float32)
    dones = np.array([0.0, 1.0, 0.0], np.float32)
    pair = (
        upstream.DictRolloutBuffer(t, dict_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
        mojosb3.DictRolloutBuffer(t, dict_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
    )
    _fill_dict_rollout(pair, t, n, rng, last_values, dones)

    for key in ("vec", "disc"):
        np.testing.assert_array_equal(pair[0].observations[key], pair[1].observations[key])
    for name in ("actions", "rewards", "episode_starts", "values", "log_probs"):
        np.testing.assert_array_equal(getattr(pair[0], name), getattr(pair[1], name))
    np.testing.assert_array_equal(pair[0].advantages, pair[1].advantages)
    np.testing.assert_array_equal(pair[0].returns, pair[1].returns)


def test_dict_rollout_get_samples_matches_upstream(dict_space, act_space):
    t, n = 6, 3
    rng = np.random.default_rng(10)
    last_values = rng.normal(size=n).astype(np.float32)
    pair = (
        upstream.DictRolloutBuffer(t, dict_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
        mojosb3.DictRolloutBuffer(t, dict_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n),
    )
    _fill_dict_rollout(pair, t, n, rng, last_values, np.zeros(n, np.float32))
    next(iter(pair[0].get()))
    next(iter(pair[1].get()))

    inds = np.random.default_rng(4).permutation(t * n)
    theirs = as_numpy(pair[0]._get_samples(inds))
    ours = as_numpy(pair[1]._get_samples(inds))
    for key in ("vec", "disc"):
        np.testing.assert_array_equal(ours["observations"][key], theirs["observations"][key])
    for name in ("actions", "old_values", "old_log_prob", "advantages", "returns"):
        np.testing.assert_array_equal(ours[name], theirs[name])


def test_dict_replay_matches_upstream(dict_space, act_space):
    n = 3
    pair = (
        upstream.DictReplayBuffer(9, dict_space, act_space, n_envs=n),
        mojosb3.DictReplayBuffer(9, dict_space, act_space, n_envs=n),
    )
    rng = np.random.default_rng(12)
    for _ in range(9):
        obs = {"vec": rng.normal(size=(n, 3)).astype(np.float32), "disc": rng.integers(0, 5, size=n)}
        next_obs = {"vec": rng.normal(size=(n, 3)).astype(np.float32), "disc": rng.integers(0, 5, size=n)}
        action = rng.normal(size=(n, 2)).astype(np.float32)
        reward = rng.normal(size=n).astype(np.float32)
        done = (rng.random(n) < 0.3).astype(np.float32)
        for buffer in pair:
            buffer.add(obs, next_obs, action, reward, done, [{} for _ in range(n)])

    for key in ("vec", "disc"):
        np.testing.assert_array_equal(pair[0].observations[key], pair[1].observations[key])
        np.testing.assert_array_equal(pair[0].next_observations[key], pair[1].next_observations[key])
    np.testing.assert_array_equal(pair[0].actions, pair[1].actions)

    inds = np.arange(pair[0].size())
    np.random.seed(2)
    theirs = as_numpy(pair[0]._get_samples(inds))
    np.random.seed(2)
    ours = as_numpy(pair[1]._get_samples(inds))
    for key in ("vec", "disc"):
        np.testing.assert_array_equal(ours["observations"][key], theirs["observations"][key])
        np.testing.assert_array_equal(ours["next_observations"][key], theirs["next_observations"][key])
    for name in ("actions", "dones", "rewards"):
        np.testing.assert_array_equal(ours[name], theirs[name])


# -------------------------------------------------------------------- NStepReplayBuffer


@pytest.mark.parametrize("n_steps", [1, 3, 4, 8])
def test_n_step_replay_matches_upstream(obs_space, act_space, n_steps):
    n = 3
    pair = (
        upstream.NStepReplayBuffer(9, obs_space, act_space, n_envs=n, n_steps=n_steps, gamma=0.97),
        mojosb3.NStepReplayBuffer(9, obs_space, act_space, n_envs=n, n_steps=n_steps, gamma=0.97),
    )
    rng = np.random.default_rng(100 + n_steps)
    for _ in range(9):
        _fill_replay(pair, rng, 4, 2, n)

    inds = np.arange(pair[0].size())
    np.random.seed(3)
    theirs = as_numpy(pair[0]._get_samples(inds))
    np.random.seed(3)
    ours = as_numpy(pair[1]._get_samples(inds))
    for name in theirs:
        np.testing.assert_allclose(ours[name], theirs[name], rtol=1e-6, atol=1e-6)


def test_n_step_return_is_discounted_until_the_first_done(obs_space, act_space):
    """With n_steps=3, gamma=0.5 and a done on the first step, the return is r0."""
    n = 1
    buffer = mojosb3.NStepReplayBuffer(4, obs_space, act_space, n_envs=n, n_steps=3, gamma=0.5)
    z = np.zeros((n, 4), np.float32)
    action = np.zeros((n, 2), np.float32)
    rewards = [1.0, 2.0, 4.0, 8.0]
    for step, reward in enumerate(rewards):
        done = np.array([1.0 if step == 0 else 0.0], np.float32)
        buffer.add(z, z, action, np.array([reward], np.float32), done, [{}])
    sample = buffer._get_samples(np.array([0]))
    assert sample.rewards[0, 0] == pytest.approx(1.0)
    assert sample.discounts[0, 0] == pytest.approx(0.5)
    assert sample.dones[0, 0] == pytest.approx(1.0)


def test_n_step_discount_is_gamma_to_the_power_of_the_mask_length(obs_space, act_space):
    n = 1
    buffer = mojosb3.NStepReplayBuffer(4, obs_space, act_space, n_envs=n, n_steps=3, gamma=0.5)
    z = np.zeros((n, 4), np.float32)
    action = np.zeros((n, 2), np.float32)
    for reward in (1.0, 2.0, 4.0, 8.0):
        buffer.add(z, z, action, np.array([reward], np.float32), np.zeros(n, np.float32), [{}])
    sample = buffer._get_samples(np.array([0]))
    # No done anywhere in the window, so the mask is all three steps.
    assert sample.discounts[0, 0] == pytest.approx(0.125)
    assert sample.rewards[0, 0] == pytest.approx(1.0 + 2.0 * 0.5 + 4.0 * 0.25)



# ------------------------------------------------- the parallel row-loop paths


# `ported.mojo` splits a row loop across `max.algorithm.parallelize` workers
# once the batch moves more than a megabyte, so the sizes below are chosen to
# land above that line and to leave a partial last chunk: the split must not
# change a single element and the last worker must still cover the tail.  Every
# comparison is against upstream on the same data.


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((4096, 3, 64), np.float32),
        ((1000, 7, 100), np.float32),
        ((900, 4, 800), np.uint8),
        ((513, 5, 256), np.float32),
    ],
)
def test_swap_and_flatten_matches_upstream_above_the_parallel_threshold(shape, dtype):
    rng = np.random.default_rng(sum(shape))
    arr = (rng.random(shape) * 100).astype(dtype)
    assert arr.nbytes > 2 << 20
    np.testing.assert_array_equal(
        mojosb3.BaseBuffer.swap_and_flatten(arr),
        upstream.BaseBuffer.swap_and_flatten(arr),
    )


def test_replay_get_samples_matches_upstream_above_the_parallel_threshold(act_space):
    space = gym.spaces.Box(-1.0, 1.0, (16, 16, 3), np.float32)
    size, n_envs, n_batch = 1024, 1, 512
    pair = (
        upstream.ReplayBuffer(size, space, act_space, n_envs=n_envs),
        mojosb3.ReplayBuffer(size, space, act_space, n_envs=n_envs),
    )
    rng = np.random.default_rng(21)
    for _ in range(size):
        _fill_replay(pair, rng, (n_envs, 16, 16, 3), 2, n_envs)
    inds = rng.integers(0, size, size=n_batch)
    assert pair[1].observations.nbytes > 2 << 20
    theirs = as_numpy(pair[0]._get_samples(inds))
    ours = as_numpy(pair[1]._get_samples(inds))
    for name in theirs:
        np.testing.assert_array_equal(ours[name], theirs[name])


def test_replay_get_samples_parallel_tail_covers_every_row(act_space):
    """A batch whose length is not a multiple of the worker count.

    `_workers` asks for one worker per megabyte, so this batch splits into
    chunks of 168 with 503 rows and the last worker takes 167.  Every row must
    be written exactly once, in order.
    """
    space = gym.spaces.Box(-1.0, 1.0, (16, 16, 3), np.float32)
    size, n_envs, n_batch = 1024, 1, 503
    buffer = mojosb3.ReplayBuffer(size, space, act_space, n_envs=n_envs)
    rng = np.random.default_rng(22)
    for _ in range(size):
        _fill_replay((buffer,), rng, (n_envs, 16, 16, 3), 2, n_envs)
    inds = rng.integers(0, size, size=n_batch)
    sample = buffer._get_samples(inds)
    np.testing.assert_array_equal(sample.observations, buffer.observations[inds, 0])
    np.testing.assert_array_equal(sample.next_observations, buffer.next_observations[inds, 0])
    np.testing.assert_array_equal(sample.actions, buffer.actions[inds, 0])
    np.testing.assert_array_equal(sample.rewards.reshape(-1), buffer.rewards[inds, 0])
    # `dones` come back with the timeout rows folded out, exactly as upstream.
    expected_dones = buffer.dones[inds, 0] * (1.0 - buffer.timeouts[inds, 0])
    np.testing.assert_array_equal(sample.dones.reshape(-1), expected_dones)


def test_swap_and_flatten_below_the_parallel_threshold_still_matches(obs_space, act_space):
    """The serial branch of the same kernel, for the sizes callers really pass."""
    rng = np.random.default_rng(23)
    arr = rng.random((32, 3, 64)).astype(np.float32)
    assert arr.nbytes < 2 << 20
    np.testing.assert_array_equal(
        mojosb3.BaseBuffer.swap_and_flatten(arr),
        upstream.BaseBuffer.swap_and_flatten(arr),
    )
