"""Ported from ``stable_baselines3/common/preprocessing.py``.

Only the two helpers the buffers need.  Upstream dispatches on
``isinstance(observation_space, spaces.Box)`` etc.; the port dispatches on the
space class name so that importing :mod:`mojosb3` does not require gymnasium.
The two are equivalent for every gymnasium space.
"""

from __future__ import annotations

import numpy as np


def _kind(space) -> str:
    return type(space).__name__


def get_obs_shape(
    observation_space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    kind = _kind(observation_space)
    if kind == "Box":
        return observation_space.shape
    elif kind == "Discrete":
        # Observation is an int
        return (1,)
    elif kind == "MultiDiscrete":
        # Number of discrete features
        return (len(observation_space.nvec),)
    elif kind == "MultiBinary":
        # Number of binary features
        return observation_space.shape
    elif kind == "Dict":
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_dim(action_space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    kind = _kind(action_space)
    if kind == "Box":
        return int(np.prod(action_space.shape))
    elif kind == "Discrete":
        # Action is an int
        return 1
    elif kind == "MultiDiscrete":
        # Number of discrete actions
        return len(action_space.nvec)
    elif kind == "MultiBinary":
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
