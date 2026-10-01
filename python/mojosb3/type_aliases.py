"""Common aliases for type hints.

Mirrors ``stable_baselines3/common/type_aliases.py``.  The sample containers keep
upstream's field names and order; the fields hold NumPy arrays instead of
``torch.Tensor`` because the port has no PyTorch dependency (see README
"Divergences").
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

GymEnv = Any
GymObs = Any
GymResetReturn = Any
GymStepReturn = Any
TensorDict = dict[str, np.ndarray]
PyTorchObs = Any


class RolloutBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    old_values: np.ndarray
    old_log_prob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: np.ndarray
    old_values: np.ndarray
    old_log_prob: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class ReplayBufferSamples(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    # For n-step replay buffer
    discounts: np.ndarray | None = None


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: np.ndarray
    next_observations: TensorDict
    dones: np.ndarray
    rewards: np.ndarray
    discounts: np.ndarray | None = None
