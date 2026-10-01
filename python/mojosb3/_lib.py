"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np
import weakref

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOSB3_LIB") or os.path.join(ROOT, "dist", "libmojo-stable-baselines3.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "sb3_simd_width": ([], I),
    "sb3_replay_add": ([I], I),
    "sb3_swap_and_flatten": ([I, I, I, I, I, I], None),
    "sb3_replay_get_samples": ([I] * 21, I),
    "sb3_compute_returns_and_advantage": ([I] * 9 + [F, F], None),
    "sb3_rollout_add": ([I] * 20, I),
    "sb3_rollout_get_samples": ([I] * 18, I),
    "sb3_n_step_get_samples": ([I] * 16 + [F] + [I] * 7, I),
    "sb3_dict_copy_row": ([I] * 5, None),
    "sb3_dict_replay_gather_key": ([I] * 9, I),
    "sb3_dict_replay_get_samples": ([I] * 14, I),
    "sb3_dict_flat_gather": ([I] * 6, I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOSB3_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOSB3_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-stable-baselines3.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


_ADDR_CACHE: dict[int, tuple[int, "weakref.ref"]] = {}
_ADDR_CACHE_MAX = 8192


def addr(a: np.ndarray) -> int:
    """The base address of `a`, memoised on the array's identity.

    `ndarray.ctypes.data` builds a fresh helper object on every access and costs
    about 1.5 us here, which is more than the kernel it feeds.  Every buffer
    array is looked up on each call with the same object, so the address is
    cached against `id(a)` and validated by a weak reference: if the array has
    been collected the entry can never match a later array, whatever address
    the allocator hands out next.  The table is dropped wholesale once it grows
    past a bound, so a long run that passes fresh arrays every step cannot
    accumulate entries.
    """
    key = id(a)
    hit = _ADDR_CACHE.get(key)
    if hit is not None and hit[1]() is a:
        return hit[0]
    try:
        pointer = ctypes.addressof(ctypes.c_char.from_buffer(a))
    except TypeError:
        # Read-only or non-contiguous: fall back to the always-available path.
        pointer = a.ctypes.data
    if len(_ADDR_CACHE) >= _ADDR_CACHE_MAX:
        _ADDR_CACHE.clear()
    _ADDR_CACHE[key] = (pointer, weakref.ref(a))
    return pointer


def simd_width() -> int:
    """float32 SIMD width the kernels assume; arrays passed to GAE are padded by it."""
    return int(lib().sb3_simd_width())


def f32(a) -> np.ndarray:
    """A C-contiguous float32 view of `a`; used for the scalar arrays only."""
    return np.ascontiguousarray(a, dtype=np.float32)


def itemsize(a: np.ndarray) -> int:
    return a.dtype.itemsize


def main() -> int:
    """`python -m mojosb3._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
