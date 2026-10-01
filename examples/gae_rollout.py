"""Fill a RolloutBuffer, compute GAE, and read minibatches back.

    pixi run python examples/gae_rollout.py

Mirrors what an on-policy loop such as PPO does between rollouts: collect
`buffer_size` transitions per environment, run
`compute_returns_and_advantage`, then iterate the buffer in minibatches.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import mojosb3

BUFFER_SIZE = 8
N_ENVS = 3
GAMMA = 0.99
GAE_LAMBDA = 0.95


def main() -> None:
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    buffer = mojosb3.RolloutBuffer(
        BUFFER_SIZE,
        observation_space,
        action_space,
        gae_lambda=GAE_LAMBDA,
        gamma=GAMMA,
        n_envs=N_ENVS,
    )

    rng = np.random.default_rng(0)
    for _ in range(BUFFER_SIZE):
        buffer.add(
            obs=rng.normal(size=(N_ENVS, 4)).astype(np.float32),
            action=rng.normal(size=(N_ENVS, 2)).astype(np.float32),
            reward=rng.normal(size=N_ENVS).astype(np.float32),
            episode_start=np.zeros(N_ENVS, dtype=np.float32),
            value=rng.normal(size=N_ENVS).astype(np.float32),
            log_prob=rng.normal(size=N_ENVS).astype(np.float32),
        )

    print(f"collected {buffer.size()} steps, full={buffer.full}")
    print(f"observations {buffer.observations.shape}  actions {buffer.actions.shape}")

    last_values = rng.normal(size=N_ENVS).astype(np.float32)
    dones = np.zeros(N_ENVS, dtype=np.float32)
    buffer.compute_returns_and_advantage(last_values, dones)

    print(f"advantages mean {buffer.advantages.mean(): .6f}")
    print(f"returns    mean {buffer.returns.mean(): .6f}")
    print(f"returns == advantages + values: "
          f"{np.allclose(buffer.returns, buffer.advantages + buffer.values)}")

    for index, batch in enumerate(buffer.get(batch_size=8)):
        print(
            f"minibatch {index}: observations {batch.observations.shape}, "
            f"actions {batch.actions.shape}, returns {batch.returns.shape}"
        )

    # The same buffer, sampled as a replay buffer would be.
    replay = mojosb3.ReplayBuffer(16, observation_space, action_space, n_envs=N_ENVS)
    for step in range(BUFFER_SIZE):
        obs = rng.normal(size=(N_ENVS, 4)).astype(np.float32)
        next_obs = rng.normal(size=(N_ENVS, 4)).astype(np.float32)
        replay.add(
            obs,
            next_obs,
            rng.normal(size=(N_ENVS, 2)).astype(np.float32),
            rng.normal(size=N_ENVS).astype(np.float32),
            np.zeros(N_ENVS, dtype=np.float32),
            [{} for _ in range(N_ENVS)],
        )
    sample = replay.sample(5)
    print(f"replay sample: observations {sample.observations.shape}, "
          f"rewards {sample.rewards.shape}, dones {sample.dones.shape}")


if __name__ == "__main__":
    main()
