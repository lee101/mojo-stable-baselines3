"""Ported from ``stable_baselines3/common/buffers.py`` (SB3 v2.9.0).

Read this module top to bottom next to the upstream file: the classes, their
method names, their argument order and their defaults are upstream's, and the
compute-bound bodies call the Mojo kernels of the same name.  What differs is
documented in the README under "Divergences"; the short version is that the
buffers and the sample containers hold NumPy arrays instead of PyTorch tensors.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import numpy as np

from ._lib import addr, f32, itemsize, lib, simd_width
from .preprocessing import get_action_dim, get_obs_shape
from .type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


def _padded(shape, dtype=np.float32) -> np.ndarray:
    """``np.zeros(shape)`` with a SIMD width of spare elements past the end.

    The backward GAE sweep reads a whole float32 vector even when ``n_envs`` is
    narrower than one; the spare elements keep that read inside the allocation.
    The returned array is a view, so it still behaves like ``np.zeros(shape)``.
    """
    flat = np.zeros(int(np.prod(shape)) + simd_width(), dtype=dtype)
    return flat[: int(np.prod(shape))].reshape(shape)


def _padded_like(arr: np.ndarray) -> np.ndarray:
    """A padded copy of a flat float32 array; see :func:`_padded`."""
    out = _padded(arr.shape, arr.dtype)
    out[:] = arr
    return out


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _is_f32_row(value) -> bool:
    """True when `value` is already a row the kernels can read without a copy.

    The replay kernels take a flat C-contiguous float32 buffer, so an input
    that is already one is used as it stands.
    """
    return (
        isinstance(value, np.ndarray)
        and value.dtype == np.float32
        and value.flags.c_contiguous
    )


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: Device the values would be converted to.
        Kept for signature compatibility with upstream; this port is CPU only
        and the samples come back as NumPy arrays.
    :param n_envs: Number of parallel environments
    """

    observation_space: Any
    obs_shape: Any

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Any = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        # The kernel reads the source as one flat C-contiguous run, so a
        # strided view (which upstream would accept) has to be packed first.
        arr = np.ascontiguousarray(arr)
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        out = np.empty((shape[0] * shape[1], *shape[2:]), dtype=arr.dtype)
        lib().sb3_swap_and_flatten(
            addr(arr),
            addr(out),
            shape[0],
            shape[1],
            _prod(shape[2:]),
            itemsize(arr),
        )
        return out

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args, strict=True):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env=None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray, env=None):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_device(self, array: np.ndarray, copy: bool = True) -> np.ndarray:
        """
        Convert an array to the buffer's working array.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference).
        :return:
        """
        if copy:
            return np.array(array)
        return np.asarray(array)

    @staticmethod
    def _normalize_obs(obs, env=None):
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env=None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: see :class:`BaseBuffer`
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Any = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Per-buffer scratch for the packed `sb3_replay_add` boundary: the
        # argument block is rewritten on every add and the zero row stands in
        # for a timeout signal no env raised.
        self._add_args = np.zeros(22, dtype=np.int64)
        self._add_args_addr = addr(self._add_args)
        self._zero_row = np.zeros(self.n_envs, dtype=np.float32)
        self._fill_static_add_args()

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def _fill_static_add_args(self) -> None:
        """Fill the `sb3_replay_add` slots that a step can never change.

        The buffer arrays are allocated once in `__init__` and are never
        reassigned, and every scalar below describes the buffer rather than the
        transition, so these fifteen slots are written once instead of on every
        call.  `add` then writes only the seven slots a step can move: the six
        row addresses and `pos`.  Rebuilding all twenty-two per call cost more
        than the kernel it feeds -- `add` runs once per environment step, so it
        is the one entry point where the boundary has to be free.
        """
        args = self._add_args
        args[0] = addr(self.observations)
        args[1] = addr(self.next_observations) if not self.optimize_memory_usage else 0
        args[2] = addr(self.actions)
        args[3] = addr(self.rewards)
        args[4] = addr(self.dones)
        args[5] = addr(self.timeouts)
        args[12] = self.buffer_size
        args[13] = self.n_envs
        args[14] = _prod(self.obs_shape)
        args[15] = self.action_dim
        args[16] = 1
        args[17] = itemsize(self.observations)
        args[18] = itemsize(self.actions)
        args[20] = int(self.optimize_memory_usage)
        args[21] = int(self.handle_timeout_termination)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if type(self.observation_space).__name__ == "Discrete":
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        obs_row = np.ascontiguousarray(obs, dtype=self.observations.dtype)
        next_obs_row = np.ascontiguousarray(next_obs, dtype=self.observations.dtype)
        action_row = np.ascontiguousarray(action, dtype=self.actions.dtype)
        # `f32(...reshape(-1))` is a no-op on a row that is already contiguous
        # float32, and paying for it built a fresh view object every step, so
        # the address cache missed on each one.  The kernel only reads these
        # two rows, so a usable input is passed straight through.
        reward_row = reward if _is_f32_row(reward) else f32(np.asarray(reward).reshape(-1))
        done_row = done if _is_f32_row(done) else f32(np.asarray(done).reshape(-1))
        # The timeouts row is a per-call allocation for a signal that is almost
        # always absent; when no env truncated, hand over the buffer's own zero
        # row, which the identity cache in `addr` then resolves for free.
        if self.handle_timeout_termination:
            flags = [info.get("TimeLimit.truncated", False) for info in infos]
            timeout_row = f32(flags) if any(flags) else self._zero_row
        else:
            timeout_row = self._zero_row

        # The static half of the block was filled in `__init__`; only these
        # seven slots move with the step.
        args = self._add_args
        args[6] = addr(obs_row)
        args[7] = addr(next_obs_row)
        args[8] = addr(action_row)
        args[9] = addr(reward_row)
        args[10] = addr(done_row)
        args[11] = addr(timeout_row)
        args[19] = self.pos
        new_pos = lib().sb3_replay_add(self._add_args_addr)
        self.full = self.full or new_pos == 0
        self.pos = new_pos

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        inds = np.ascontiguousarray(batch_inds, dtype=np.int64).reshape(-1)
        envs = np.ascontiguousarray(env_indices, dtype=np.int64).reshape(-1)
        obs_size = _prod(self.obs_shape)
        n_batch = len(inds)

        dst_obs = np.empty((n_batch, obs_size), dtype=self.observations.dtype)
        dst_next_obs = np.empty((n_batch, obs_size), dtype=self.observations.dtype)
        dst_actions = np.empty((n_batch, self.action_dim), dtype=self.actions.dtype)
        dst_dones = np.empty((n_batch, 1), dtype=np.float32)
        dst_rewards = np.empty((n_batch, 1), dtype=np.float32)

        lib().sb3_replay_get_samples(
            addr(self.observations),
            addr(self.next_observations) if not self.optimize_memory_usage else addr(self.observations),
            addr(self.actions),
            addr(self.rewards),
            addr(self.dones),
            addr(self.timeouts),
            addr(inds),
            addr(envs),
            n_batch,
            addr(dst_obs),
            addr(dst_next_obs),
            addr(dst_actions),
            addr(dst_dones),
            addr(dst_rewards),
            self.buffer_size,
            self.n_envs,
            obs_size,
            self.action_dim,
            itemsize(self.observations),
            itemsize(self.actions),
            int(self.optimize_memory_usage),
        )

        # Every array below was allocated by the call above and is unaliased, so
        # `copy=False` hands back the array the kernel just filled instead of
        # duplicating the whole batch.  On an Atari-sized batch that second copy
        # was the larger half of `sample()`.
        data = (
            self._normalize_obs(dst_obs.reshape((n_batch, *self.obs_shape)), env),
            dst_actions,
            self._normalize_obs(dst_next_obs.reshape((n_batch, *self.obs_shape)), env),
            dst_dones,
            self._normalize_reward(dst_rewards, env),
        )
        return ReplayBufferSamples(*(self.to_device(d, copy=False) for d in data))

    @staticmethod
    def _maybe_cast_dtype(dtype):
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: see :class:`BaseBuffer`
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Any = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = _padded((self.buffer_size, self.n_envs))
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = _padded((self.buffer_size, self.n_envs))
        self.values = _padded((self.buffer_size, self.n_envs))
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = _padded((self.buffer_size, self.n_envs))
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = _padded_like(f32(np.asarray(last_values).reshape(-1)))
        dones = _padded_like(f32(np.asarray(dones).reshape(-1)))

        lib().sb3_compute_returns_and_advantage(
            addr(self.rewards),
            addr(self.values),
            addr(self.episode_starts),
            addr(self.advantages),
            addr(self.returns),
            addr(last_values),
            addr(dones),
            self.buffer_size,
            self.n_envs,
            float(self.gamma),
            float(self.gae_lambda),
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if np.ndim(log_prob) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = np.reshape(log_prob, (-1, 1))

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if type(self.observation_space).__name__ == "Discrete":
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        obs_row = np.ascontiguousarray(obs, dtype=self.observations.dtype)
        action_row = np.ascontiguousarray(action, dtype=self.actions.dtype)

        self.pos = lib().sb3_rollout_add(
            addr(self.observations),
            addr(self.actions),
            addr(self.rewards),
            addr(self.episode_starts),
            addr(self.values),
            addr(self.log_probs),
            addr(obs_row),
            addr(action_row),
            addr(f32(reward).reshape(-1)),
            addr(f32(episode_start).reshape(-1)),
            addr(f32(np.asarray(value).reshape(-1))),
            addr(f32(np.asarray(log_prob).reshape(-1))),
            self.n_envs,
            _prod(self.obs_shape),
            1,
            self.action_dim,
            4,
            itemsize(self.actions),
            self.pos,
            self.buffer_size,
        )
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            n = min(batch_size, self.buffer_size * self.n_envs - start_idx)
            yield self._get_samples(indices[start_idx : start_idx + n])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> RolloutBufferSamples:
        inds = np.ascontiguousarray(batch_inds, dtype=np.int64).reshape(-1)
        n_batch = len(inds)
        obs_size = _prod(self.obs_shape)

        dst_obs = np.empty((n_batch, obs_size), dtype=self.observations.dtype)
        dst_actions = np.empty((n_batch, self.action_dim), dtype=np.float32)
        dst_values = np.empty(n_batch, dtype=np.float32)
        dst_log_probs = np.empty(n_batch, dtype=np.float32)
        dst_advantages = np.empty(n_batch, dtype=np.float32)
        dst_returns = np.empty(n_batch, dtype=np.float32)

        # `_cast_row` reads a 2- or 8-byte action array through an integer
        # pointer of the same width, which is right for int16/int64 and wrong
        # for the float dtypes of those widths.  The buffer keeps upstream's
        # `action_space.dtype`, so pack those two into float32 first and hand
        # the kernel a 4-byte array; every other dtype goes as it is.
        actions = self.actions
        action_itemsize = itemsize(actions)
        if actions.dtype.kind == "f" and action_itemsize != 4:
            actions = np.ascontiguousarray(actions, dtype=np.float32)
            action_itemsize = 4
        lib().sb3_rollout_get_samples(
            addr(self.observations),
            addr(actions),
            addr(self.values),
            addr(self.log_probs),
            addr(self.advantages),
            addr(self.returns),
            addr(inds),
            n_batch,
            addr(dst_obs),
            addr(dst_actions),
            addr(dst_values),
            addr(dst_log_probs),
            addr(dst_advantages),
            addr(dst_returns),
            obs_size,
            self.action_dim,
            itemsize(self.observations),
            action_itemsize,
        )

        # All six arrays were allocated by the call above; `copy=False` avoids
        # duplicating the batch, which the kernel has just written.
        data = (
            dst_obs.reshape((n_batch, *self.obs_shape)),
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            dst_actions,
            dst_values,
            dst_log_probs,
            dst_advantages,
            dst_returns,
        )
        return RolloutBufferSamples(*(self.to_device(d, copy=False) for d in data))


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: see :class:`BaseBuffer`
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Any = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Packed `sb3_replay_add` boundary scratch; see `ReplayBuffer.__init__`.
        self._add_args = np.zeros(22, dtype=np.int64)
        self._add_args_addr = addr(self._add_args)
        self._zero_row = np.zeros(self.n_envs, dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if type(self.observation_space.spaces[key]).__name__ == "Discrete":
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])

        for key in self.next_observations.keys():
            if type(self.observation_space.spaces[key]).__name__ == "Discrete":
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # `dict_copy_row` is the kernel behind
        # `self.observations[key][self.pos] = np.array(obs[key])`, run once per
        # key exactly as the loop above does.
        for key in self.observations:
            src = np.ascontiguousarray(obs[key], dtype=self.observations[key].dtype)
            lib().sb3_dict_copy_row(
                addr(self.observations[key]),
                addr(src),
                _prod(self.obs_shape[key]) * self.n_envs,
                self.observations[key].dtype.itemsize,
                self.pos,
            )
            del src

        for key in self.next_observations:
            src = np.ascontiguousarray(next_obs[key], dtype=self.next_observations[key].dtype)
            lib().sb3_dict_copy_row(
                addr(self.next_observations[key]),
                addr(src),
                _prod(self.obs_shape[key]) * self.n_envs,
                self.next_observations[key].dtype.itemsize,
                self.pos,
            )
            del src

        action_row = np.ascontiguousarray(action, dtype=self.actions.dtype)
        if self.handle_timeout_termination:
            flags = [info.get("TimeLimit.truncated", False) for info in infos]
            timeout_row = f32(flags) if any(flags) else self._zero_row
        else:
            timeout_row = self._zero_row

        # These rows must outlive the kernel call: the tuple below only stores
        # their addresses, so an inline `f32(...)` temporary would be freed at
        # the end of the assignment and the kernel would read released memory.
        reward_row = f32(np.asarray(reward).reshape(-1))
        done_row = f32(np.asarray(done).reshape(-1))

        # The order here is `sb3_replay_add`'s in `src/capi.mojo`.
        self._add_args[:] = (
            0,
            0,
            addr(self.actions),
            addr(self.rewards),
            addr(self.dones),
            addr(self.timeouts),
            0,
            0,
            addr(action_row),
            addr(reward_row),
            addr(done_row),
            addr(timeout_row),
            self.buffer_size,
            self.n_envs,
            0,
            self.action_dim,
            0,
            0,
            self.actions.dtype.itemsize,
            self.pos,
            0,
            int(self.handle_timeout_termination),
        )
        new_pos = lib().sb3_replay_add(self._add_args_addr)
        self.full = self.full or new_pos == 0
        self.pos = new_pos

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env=None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        inds = np.ascontiguousarray(batch_inds, dtype=np.int64).reshape(-1)
        envs = np.ascontiguousarray(env_indices, dtype=np.int64).reshape(-1)
        n_batch = len(inds)

        obs_dst = {
            key: np.empty((n_batch, _prod(self.obs_shape[key])), dtype=self.observations[key].dtype)
            for key in self.observations
        }
        next_obs_dst = {
            key: np.empty((n_batch, _prod(self.obs_shape[key])), dtype=self.next_observations[key].dtype)
            for key in self.next_observations
        }
        dst_actions = np.empty((n_batch, self.action_dim), dtype=self.actions.dtype)
        dst_dones = np.empty((n_batch, 1), dtype=np.float32)
        dst_rewards = np.empty((n_batch, 1), dtype=np.float32)

        for key in self.observations:
            lib().sb3_dict_replay_gather_key(
                addr(self.observations[key]),
                addr(obs_dst[key]),
                _prod(self.obs_shape[key]),
                self.observations[key].dtype.itemsize,
                addr(inds),
                addr(envs),
                n_batch,
                self.buffer_size,
                self.n_envs,
            )
        for key in self.next_observations:
            lib().sb3_dict_replay_gather_key(
                addr(self.next_observations[key]),
                addr(next_obs_dst[key]),
                _prod(self.obs_shape[key]),
                self.next_observations[key].dtype.itemsize,
                addr(inds),
                addr(envs),
                n_batch,
                self.buffer_size,
                self.n_envs,
            )
        lib().sb3_dict_replay_get_samples(
            addr(self.actions),
            addr(self.rewards),
            addr(self.dones),
            addr(self.timeouts),
            addr(inds),
            addr(envs),
            n_batch,
            self.buffer_size,
            self.n_envs,
            self.action_dim,
            self.actions.dtype.itemsize,
            addr(dst_actions),
            addr(dst_dones),
            addr(dst_rewards),
        )

        obs_ = self._normalize_obs(
            {key: obs.reshape((n_batch, *self.obs_shape[key])) for key, obs in obs_dst.items()}, env
        )
        next_obs_ = self._normalize_obs(
            {key: obs.reshape((n_batch, *self.obs_shape[key])) for key, obs in next_obs_dst.items()}, env
        )
        # Every array here comes from a `np.empty` above or from the caller's
        # fancy indexing, so `copy=False` returns it unaliased.
        observations = {key: self.to_device(obs, copy=False) for key, obs in obs_.items()}
        next_observations = {key: self.to_device(obs, copy=False) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_device(dst_actions, copy=False),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_device(dst_dones, copy=False),
            rewards=self.to_device(self._normalize_reward(dst_rewards, env), copy=False),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: see :class:`BaseBuffer`
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Any = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape), dtype=self.observation_space[key].dtype
            )
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = _padded((self.buffer_size, self.n_envs))
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = _padded((self.buffer_size, self.n_envs))
        self.values = _padded((self.buffer_size, self.n_envs))
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = _padded((self.buffer_size, self.n_envs))
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if np.ndim(log_prob) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = np.reshape(log_prob, (-1, 1))

        # `dict_copy_row` is the kernel behind
        # `self.observations[key][self.pos] = obs_`, run once per key exactly as
        # the loop above does.
        for key in self.observations:
            src = np.ascontiguousarray(obs[key], dtype=self.observations[key].dtype)
            lib().sb3_dict_copy_row(
                addr(self.observations[key]),
                addr(src),
                _prod(self.obs_shape[key]) * self.n_envs,
                self.observations[key].dtype.itemsize,
                self.pos,
            )
            del src

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.pos = lib().sb3_rollout_add(
            0,
            addr(self.actions),
            addr(self.rewards),
            addr(self.episode_starts),
            addr(self.values),
            addr(self.log_probs),
            0,
            addr(np.ascontiguousarray(action, dtype=self.actions.dtype)),
            addr(f32(reward).reshape(-1)),
            addr(f32(episode_start).reshape(-1)),
            addr(f32(np.asarray(value).reshape(-1))),
            addr(f32(np.asarray(log_prob).reshape(-1))),
            self.n_envs,
            0,
            0,
            self.action_dim,
            4,
            self.actions.dtype.itemsize,
            self.pos,
            self.buffer_size,
        )
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: int | None = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            n = min(batch_size, self.buffer_size * self.n_envs - start_idx)
            yield self._get_samples(indices[start_idx : start_idx + n])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env=None,
    ) -> DictRolloutBufferSamples:
        inds = np.ascontiguousarray(batch_inds, dtype=np.int64).reshape(-1)
        n_batch = len(inds)
        obs_dst = {
            key: np.empty((n_batch, _prod(self.obs_shape[key])), dtype=self.observations[key].dtype)
            for key in self.observations
        }
        for key in self.observations:
            lib().sb3_dict_flat_gather(
                addr(self.observations[key]),
                addr(obs_dst[key]),
                _prod(self.obs_shape[key]),
                self.observations[key].dtype.itemsize,
                addr(inds),
                n_batch,
            )

        # `obs_dst` was allocated above and the `[batch_inds]` fancy indexing
        # already returns fresh arrays, so `copy=False` never aliases a buffer.
        return DictRolloutBufferSamples(
            observations={
                key: self.to_device(obs.reshape((n_batch, *self.obs_shape[key])), copy=False)
                for (key, obs) in obs_dst.items()
            },
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            actions=self.to_device(self.actions[batch_inds].astype(np.float32, copy=False), copy=False),
            old_values=self.to_device(self.values[batch_inds].flatten(), copy=False),
            old_log_prob=self.to_device(self.log_probs[batch_inds].flatten(), copy=False),
            advantages=self.to_device(self.advantages[batch_inds].flatten(), copy=False),
            returns=self.to_device(self.returns[batch_inds].flatten(), copy=False),
        )


class NStepReplayBuffer(ReplayBuffer):
    """
    Replay buffer used for computing n-step returns in off-policy algorithms like SAC/DQN.

    The n-step return combines multiple steps of future rewards,
    discounted by the discount factor gamma.
    This can help improve sample efficiency and credit assignment.

    This implementation uses the same storage space as a normal replay buffer,
    and vectorized operations at sampling time to efficiently compute the
    n-step return, without requiring extra memory.

    This implementation is inspired by:
    - https://github.com/younggyoseo/FastTD3
    - https://github.com/DLR-RM/stable-baselines3/pull/81

    It avoids potential issues such as:
    - https://github.com/younggyoseo/FastTD3/issues/6

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: see :class:`BaseBuffer`
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_steps: Number of steps to accumulate rewards for n-step returns
    :param gamma: Discount factor for future rewards
    """

    def __init__(self, *args, n_steps: int = 3, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        if self.optimize_memory_usage:
            raise NotImplementedError("NStepReplayBuffer doesn't support optimize_memory_usage=True")

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions and compute n-step returns.

        For each sampled transition, the method computes the cumulative discounted reward over
        the next `n_steps`, properly handling episode termination and timeouts.
        The next observation and done flag correspond to the last transition in the computed n-step trajectory.

        :param batch_inds: Indices of samples to retrieve
        :param env: Optional VecNormalize environment for normalizing observations/rewards
        :return: A batch of samples with n-step returns and corresponding observations/actions
        """
        # Randomly choose env indices for each sample
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)

        # Note: the self.pos index is dangerous (will overlap two different episodes when buffer is full)
        # so we set self.pos-1 to truncated=True (temporarily) if done=False and truncated=False
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(original_timeout_values, np.logical_not(self.dones[last_valid_index]))

        inds = np.ascontiguousarray(batch_inds, dtype=np.int64).reshape(-1)
        envs = np.ascontiguousarray(env_indices, dtype=np.int64).reshape(-1)
        n_batch = len(inds)
        obs_size = _prod(self.obs_shape)

        dst_obs = np.empty((n_batch, obs_size), dtype=self.observations.dtype)
        dst_next_obs = np.empty((n_batch, obs_size), dtype=self.observations.dtype)
        dst_actions = np.empty((n_batch, self.action_dim), dtype=self.actions.dtype)
        dst_dones = np.empty((n_batch, 1), dtype=np.float32)
        dst_rewards = np.empty((n_batch, 1), dtype=np.float32)
        dst_discounts = np.empty((n_batch, 1), dtype=np.float32)
        scratch = np.empty(n_batch * self.n_steps, dtype=np.float32)

        lib().sb3_n_step_get_samples(
            addr(self.observations),
            addr(self.next_observations),
            addr(self.actions),
            addr(self.rewards),
            addr(self.dones),
            addr(self.timeouts),
            addr(inds),
            addr(envs),
            n_batch,
            self.n_steps,
            self.buffer_size,
            self.n_envs,
            obs_size,
            self.action_dim,
            itemsize(self.observations),
            itemsize(self.actions),
            float(self.gamma),
            addr(dst_obs),
            addr(dst_next_obs),
            addr(dst_actions),
            addr(dst_dones),
            addr(dst_rewards),
            addr(dst_discounts),
            addr(scratch),
        )

        # Revert back tmp changes to avoid sampling across episodes
        self.timeouts[last_valid_index] = original_timeout_values

        obs = self._normalize_obs(dst_obs.reshape((n_batch, *self.obs_shape)), env)
        next_obs = self._normalize_obs(dst_next_obs.reshape((n_batch, *self.obs_shape)), env)
        actions = dst_actions

        # All six arrays come from a `np.empty` above, so `copy=False` returns
        # the array the kernel filled rather than a second copy of the batch.
        return ReplayBufferSamples(
            observations=self.to_device(obs, copy=False),
            actions=self.to_device(actions, copy=False),
            next_observations=self.to_device(next_obs, copy=False),
            dones=self.to_device(dst_dones, copy=False),
            rewards=self.to_device(dst_rewards, copy=False),
            discounts=self.to_device(dst_discounts, copy=False),
        )
