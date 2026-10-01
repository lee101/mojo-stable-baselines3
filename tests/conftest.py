import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from mojosb3 import _lib  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def built_library():
    """Force the Mojo shared library to exist before anything imports a symbol."""
    _lib.build()
    return _lib.LIB


@pytest.fixture
def obs_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)


@pytest.fixture
def act_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


@pytest.fixture
def rollout_data():
    """(observations, actions, rewards, episode_starts, values, log_probs)."""
    rng = np.random.default_rng(0)
    t, n = 8, 3
    return (
        rng.normal(size=(t, n, 4)).astype(np.float32),
        rng.normal(size=(t, n, 2)).astype(np.float32),
        rng.normal(size=(t, n)).astype(np.float32),
        (rng.random((t, n)) < 0.25).astype(np.float32),
        rng.normal(size=(t, n)).astype(np.float32),
        rng.normal(size=(t, n)).astype(np.float32),
    )


@pytest.fixture
def last_step():
    rng = np.random.default_rng(11)
    return rng.normal(size=3).astype(np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)


def as_numpy(sample):
    """Upstream samples hold torch tensors; the port's hold NumPy arrays.

    Everything is normalised to NumPy so the two can be compared element for
    element; that normalisation is the only adaptation the tests make.
    """
    out = {}
    for field in sample._fields:
        value = getattr(sample, field)
        if isinstance(value, dict):
            out[field] = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in value.items()}
        elif value is None:
            out[field] = None
        else:
            out[field] = value.numpy() if hasattr(value, "numpy") else value
    return out
