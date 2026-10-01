"""mojo-stable-baselines3 against upstream stable-baselines3 on the same data.

    pixi run bench

Reports the honest number, including the cases upstream wins.  Note that
upstream's `_get_samples` ends in `to_torch`, so its sampling numbers include
PyTorch tensor construction; that is what a caller of upstream actually pays,
and it is called out in the table where it matters.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import gymnasium as gym  # noqa: E402
import torch as th  # noqa: E402

import mojosb3  # noqa: E402
from stable_baselines3.common import buffers as upstream  # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine() -> str:
    cores = os.cpu_count()
    return f"{cpu_model()}, {cores} logical cores, {platform.system()} {platform.machine()}"


CASES = []


def case(name: str):
    def decorate(fn):
        CASES.append((name, fn))
        return fn
    return decorate


# ---------------------------------------------------------------- rollout / GAE


def fill_rollout(buffer, t, n, obs_size, act_size, rng, as_torch):
    obs = rng.normal(size=(n, obs_size)).astype(np.float32)
    action = rng.normal(size=(n, act_size)).astype(np.float32)
    reward = rng.normal(size=n).astype(np.float32)
    start = (rng.random(n) < 0.05).astype(np.float32)
    value = rng.normal(size=n).astype(np.float32)
    log_prob = rng.normal(size=n).astype(np.float32)
    for _ in range(t):
        buffer.add(
            obs, action, reward, start,
            th.as_tensor(value) if as_torch else value,
            th.as_tensor(log_prob) if as_torch else log_prob,
        )
    return rng.normal(size=n).astype(np.float32), (rng.random(n) < 0.02).astype(np.float32)


@case("RolloutBuffer.compute_returns_and_advantage (2048 x 8 envs, 4d)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    t, n = 2048, 8
    rng = np.random.default_rng(0)
    theirs = upstream.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n)
    ours = mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n)
    last_values, dones = fill_rollout(theirs, t, n, 4, 2, rng, True)
    fill_rollout(ours, t, n, 4, 2, np.random.default_rng(0), False)
    last_t = th.as_tensor(last_values)
    return (
        lambda: ours.compute_returns_and_advantage(last_values, dones),
        lambda: theirs.compute_returns_and_advantage(last_t, dones),
    )


@case("RolloutBuffer.compute_returns_and_advantage (2048 x 64 envs, 4d)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    t, n = 2048, 64
    rng = np.random.default_rng(1)
    theirs = upstream.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n)
    ours = mojosb3.RolloutBuffer(t, obs_space, act_space, gae_lambda=0.95, gamma=0.99, n_envs=n)
    last_values, dones = fill_rollout(theirs, t, n, 4, 2, rng, True)
    fill_rollout(ours, t, n, 4, 2, np.random.default_rng(1), False)
    last_t = th.as_tensor(last_values)
    return (
        lambda: ours.compute_returns_and_advantage(last_values, dones),
        lambda: theirs.compute_returns_and_advantage(last_t, dones),
    )


@case("RolloutBuffer.get() one full pass (2048 x 8 envs, 4d)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    t, n = 2048, 8
    rng = np.random.default_rng(2)
    theirs = upstream.RolloutBuffer(t, obs_space, act_space, n_envs=n)
    ours = mojosb3.RolloutBuffer(t, obs_space, act_space, n_envs=n)
    last_values, dones = fill_rollout(theirs, t, n, 4, 2, rng, True)
    fill_rollout(ours, t, n, 4, 2, np.random.default_rng(2), False)
    theirs.compute_returns_and_advantage(th.as_tensor(last_values), dones)
    ours.compute_returns_and_advantage(last_values, dones)
    return (
        lambda: [b for b in ours.get()],
        lambda: [b for b in theirs.get()],
    )


# ---------------------------------------------------------------- replay buffer


def fill_replay(buffer, size, n, obs_space_shape, act_size, rng, obs_dtype=np.float32):
    obs = rng.random((n, *obs_space_shape)).astype(obs_dtype)
    next_obs = rng.random((n, *obs_space_shape)).astype(obs_dtype)
    action = rng.normal(size=(n, act_size)).astype(np.float32)
    reward = rng.normal(size=n).astype(np.float32)
    done = (rng.random(n) < 0.02).astype(np.float32)
    for _ in range(size):
        buffer.add(obs, next_obs, action, reward, done, [{} for _ in range(n)])


@case("ReplayBuffer.sample(4096) (100k buffer, 4d obs)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    size, n = 100_000, 1
    rng = np.random.default_rng(3)
    theirs = upstream.ReplayBuffer(size, obs_space, act_space, n_envs=n)
    ours = mojosb3.ReplayBuffer(size, obs_space, act_space, n_envs=n)
    fill_replay(theirs, size, n, (4,), 2, rng)
    fill_replay(ours, size, n, (4,), 2, np.random.default_rng(3))

    def run_theirs():
        np.random.seed(0)
        return theirs.sample(4096)

    def run_ours():
        np.random.seed(0)
        return ours.sample(4096)

    return run_ours, run_theirs


@case("ReplayBuffer.sample(2048) (50k buffer, 84x84x3 uint8 obs)")
def _():
    obs_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
    act_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    size, n = 50_000, 1
    rng = np.random.default_rng(4)
    theirs = upstream.ReplayBuffer(size, obs_space, act_space, n_envs=n)
    ours = mojosb3.ReplayBuffer(size, obs_space, act_space, n_envs=n)
    fill_replay(theirs, size, n, (84, 84, 3), 1, rng, np.uint8)
    fill_replay(ours, size, n, (84, 84, 3), 1, np.random.default_rng(4), np.uint8)

    def run_theirs():
        np.random.seed(0)
        return theirs.sample(2048)

    def run_ours():
        np.random.seed(0)
        return ours.sample(2048)

    return run_ours, run_theirs


@case("ReplayBuffer.add x 20000 (1 env, 4d obs)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    n = 1
    rng = np.random.default_rng(5)
    theirs = upstream.ReplayBuffer(40_000, obs_space, act_space, n_envs=n)
    ours = mojosb3.ReplayBuffer(40_000, obs_space, act_space, n_envs=n)
    obs = rng.normal(size=(n, 4)).astype(np.float32)
    next_obs = rng.normal(size=(n, 4)).astype(np.float32)
    action = rng.normal(size=(n, 2)).astype(np.float32)
    reward = rng.normal(size=n).astype(np.float32)
    done = np.zeros(n, np.float32)
    infos = [{}]

    def run_theirs():
        pos = 0
        for _ in range(20_000):
            theirs.add(obs, next_obs, action, reward, done, infos)
            theirs.pos = pos
            pos = (pos + 1) % theirs.buffer_size

    def run_ours():
        pos = 0
        for _ in range(20_000):
            ours.add(obs, next_obs, action, reward, done, infos)
            ours.pos = pos
            pos = (pos + 1) % ours.buffer_size

    return run_ours, run_theirs


@case("ReplayBuffer.sample(4096) optimize_memory_usage (100k, 4d obs)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    size, n = 100_000, 1
    rng = np.random.default_rng(6)
    kwargs = dict(optimize_memory_usage=True, handle_timeout_termination=False)
    theirs = upstream.ReplayBuffer(size, obs_space, act_space, n_envs=n, **kwargs)
    ours = mojosb3.ReplayBuffer(size, obs_space, act_space, n_envs=n, **kwargs)
    fill_replay(theirs, size, n, (4,), 2, rng)
    fill_replay(ours, size, n, (4,), 2, np.random.default_rng(6))

    def run_theirs():
        np.random.seed(0)
        return theirs.sample(4096)

    def run_ours():
        np.random.seed(0)
        return ours.sample(4096)

    return run_ours, run_theirs


@case("NStepReplayBuffer.sample(4096) n_steps=4 (100k buffer, 4d obs)")
def _():
    obs_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    size, n = 100_000, 1
    rng = np.random.default_rng(7)
    kwargs = dict(n_steps=4, gamma=0.99)
    theirs = upstream.NStepReplayBuffer(size, obs_space, act_space, n_envs=n, **kwargs)
    ours = mojosb3.NStepReplayBuffer(size, obs_space, act_space, n_envs=n, **kwargs)
    fill_replay(theirs, size, n, (4,), 2, rng)
    fill_replay(ours, size, n, (4,), 2, np.random.default_rng(7))

    def run_theirs():
        np.random.seed(0)
        return theirs.sample(4096)

    def run_ours():
        np.random.seed(0)
        return ours.sample(4096)

    return run_ours, run_theirs


# ---------------------------------------------------------------- dict buffers


@case("DictReplayBuffer.sample(4096) (100k buffer, 4d + discrete)")
def _():
    space = gym.spaces.Dict(
        {"vec": gym.spaces.Box(-1, 1, (4,), np.float32), "disc": gym.spaces.Discrete(16)}
    )
    act_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    size, n = 100_000, 1
    rng = np.random.default_rng(8)
    theirs = upstream.DictReplayBuffer(size, space, act_space, n_envs=n)
    ours = mojosb3.DictReplayBuffer(size, space, act_space, n_envs=n)
    for _ in range(size):
        obs = {"vec": rng.normal(size=(n, 4)).astype(np.float32), "disc": rng.integers(0, 16, size=n)}
        next_obs = {"vec": rng.normal(size=(n, 4)).astype(np.float32), "disc": rng.integers(0, 16, size=n)}
        action = rng.normal(size=(n, 2)).astype(np.float32)
        reward = rng.normal(size=n).astype(np.float32)
        done = np.zeros(n, np.float32)
        theirs.add(obs, next_obs, action, reward, done, [{}])
        ours.add(obs, next_obs, action, reward, done, [{}])

    def run_theirs():
        np.random.seed(0)
        return theirs.sample(4096)

    def run_ours():
        np.random.seed(0)
        return ours.sample(4096)

    return run_ours, run_theirs


# ---------------------------------------------------------------- primitives


@case("BaseBuffer.swap_and_flatten (200000 x 4 envs x 64d)")
def _():
    arr = np.ascontiguousarray(np.random.default_rng(9).normal(size=(200_000, 4, 64)), dtype=np.float32)
    return (
        lambda: mojosb3.BaseBuffer.swap_and_flatten(arr),
        lambda: upstream.BaseBuffer.swap_and_flatten(arr),
    )


def main() -> None:
    print(f"machine: {machine()}")
    print(f"python: {platform.python_version()}   numpy: {np.__version__}   torch: {th.__version__}")
    print(f"upstream stable-baselines3: {__import__('stable_baselines3').__version__}")
    print()
    print(f"| {'case':<58} | {'mojo-stable-baselines3':>20} | {'stable-baselines3':>18} | {'speedup':>8} |")
    print(f"|{'-' * 60}|{'-' * 22}|{'-' * 20}|{'-' * 10}|")
    total_ours = 0.0
    total_theirs = 0.0
    for name, build in CASES:
        ours, theirs = build()
        ours()
        theirs()
        a = timeit(ours)
        b = timeit(theirs)
        total_ours += a
        total_theirs += b
        verdict = "faster" if a < b else "SLOWER"
        print(f"| {name:<58} | {a * 1e3:>17.2f} ms | {b * 1e3:>15.2f} ms | {b / a:>7.2f}x | {verdict}")
    print()
    print(f"total: mojo-stable-baselines3 {total_ours * 1e3:.1f} ms, "
          f"stable-baselines3 {total_theirs * 1e3:.1f} ms, {total_theirs / total_ours:.2f}x")


if __name__ == "__main__":
    main()
