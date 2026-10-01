"""Kernels ported from ``stable_baselines3/common/buffers.py`` (SB3 v2.9.0).

Read this file top to bottom next to the upstream file: the functions appear in
upstream source order, keep upstream's names, argument order and control flow,
and the loops that upstream writes as NumPy whole-array operations are kept as
loops here.  Only the tensor plumbing differs -- buffers arrive as ``Int``
addresses because ``@export`` cannot take a pointer, and the dtype of an
observation/action array arrives as an itemsize because ``@export`` cannot be
parametric.  Both divergences live in the kernel *signature*, not in the kernel
body; the branch structure and the arithmetic follow upstream exactly.

Toolchain invariant this file depends on (see MOJO_NOTES.md 3a): a pointer is
built once, before any loop, and indexed with ``unsafe_offset``.  Building one
from a loop-varying address expression silently miscompiles on mojo
1.2.0.dev2026092605.
"""

from max.algorithm import parallelize
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float32]()
comptime VEC = SIMD[DType.float32, W]
comptime WU = simd_width_of[DType.uint8]()

comptime PtrF = Pointer[Float32, AnyOrigin[mut=True]]
comptime PtrD = Pointer[Float64, AnyOrigin[mut=True]]
comptime PtrH = Pointer[Int16, AnyOrigin[mut=True]]
comptime PtrU = Pointer[UInt8, AnyOrigin[mut=True]]
comptime PtrI = Pointer[Int64, AnyOrigin[mut=True]]

# A row loop over independent units of work: one worker per megabyte moved,
# capped, and only above the threshold -- below it the launch costs more than
# the copy.  `bench/bench.py` is what both numbers come from.
comptime PAR_MIN_BYTES = 1 << 20
comptime PAR_MAX_WORKERS = 16


def _workers(bytes_moved: Int) -> Int:
    var n = bytes_moved // PAR_MIN_BYTES
    if n < 1:
        return 1
    if n > PAR_MAX_WORKERS:
        return PAR_MAX_WORKERS
    return n


# --------------------------------------------------------------- primitives


@always_inline
def _copy_f32(d: PtrF, s: PtrF, count: Int):
    var i = 0
    while i + W <= count:
        d.unsafe_store[width=W](i, s.unsafe_load[width=W](i))
        i += W
    while i < count:
        d.unsafe_store(i, s.unsafe_load(i))
        i += 1


@always_inline
def _copy_u8(d: PtrU, s: PtrU, count: Int):
    var i = 0
    while i + WU <= count:
        d.unsafe_store[width=WU](i, s.unsafe_load[width=WU](i))
        i += WU
    while i < count:
        d.unsafe_store(i, s.unsafe_load(i))
        i += 1


@always_inline
def _gather_row(
    d: PtrF,
    s: PtrF,
    d8: PtrU,
    s8: PtrU,
    doff: Int,
    soff: Int,
    count: Int,
    itemsize: Int,
):
    """``dst[doff:doff + count] = src[soff:soff + count]``, dtype preserving.

    The float32 pair and the byte pair are the same addresses viewed two ways;
    which pair is used is the only thing ``itemsize`` selects, so a 1/2/4/8 byte
    dtype all go through here.
    """
    if itemsize == 4:
        _copy_f32(d.unsafe_offset(doff), s.unsafe_offset(soff), count)
    else:
        _copy_u8(
            d8.unsafe_offset(doff * itemsize),
            s8.unsafe_offset(soff * itemsize),
            count * itemsize,
        )


@always_inline
def _cast_row(
    d: PtrF,
    s32: PtrF,
    s8: Pointer[Int8, AnyOrigin[mut=True]],
    s16: PtrH,
    s64: PtrI,
    soff: Int,
    count: Int,
    itemsize: Int,
):
    """``src[soff:...].astype(np.float32)`` into ``d``.

    The source is read through the *integer* pointer of its width, not by
    reinterpreting the bytes as a float: a float32 constructor applied to a
    float loaded from an int64 buffer is a numeric conversion of the wrong
    number, not a cast of the stored integer.
    """
    if itemsize == 4:
        _copy_f32(d, s32.unsafe_offset(soff), count)
    elif itemsize == 8:
        var src = s64.unsafe_offset(soff)
        for i in range(count):
            d.unsafe_store(i, Float32(src.unsafe_load(i)))
    elif itemsize == 1:
        var src = s8.unsafe_offset(soff)
        for i in range(count):
            d.unsafe_store(i, Float32(src.unsafe_load(i)))
    else:
        var src = s16.unsafe_offset(soff)
        for i in range(count):
            d.unsafe_store(i, Float32(src.unsafe_load(i)))


# -------------------------------------------------------- BaseBuffer.swap_and_flatten


def swap_and_flatten(
    src: Int,
    dst: Int,
    buffer_size: Int,
    n_envs: Int,
    feature_size: Int,
    itemsize: Int,
):
    """``arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])``.

    Source is ``[buffer_size, n_envs, feature_size]``; the destination row order
    is env-major, i.e. destination ``[n, step]`` is source ``[step, n]``.
    """
    var s = PtrF(unsafe_from_address=src)
    var d = PtrF(unsafe_from_address=dst)
    var s8 = PtrU(unsafe_from_address=src)
    var d8 = PtrU(unsafe_from_address=dst)
    # Steps are independent: each writes its own destination rows, so the loop
    # splits into contiguous chunks once the moved bytes justify a launch.
    var n_workers = _workers(buffer_size * n_envs * feature_size * itemsize)
    if n_workers > 1:
        var chunk = (buffer_size + n_workers - 1) // n_workers

        def work(w: Int) {imm}:
            var lo = w * chunk
            var hi = lo + chunk
            if hi > buffer_size:
                hi = buffer_size
            for step in range(lo, hi):
                for n in range(n_envs):
                    _gather_row(
                        d,
                        s,
                        d8,
                        s8,
                        (n * buffer_size + step) * feature_size,
                        (step * n_envs + n) * feature_size,
                        feature_size,
                        itemsize,
                    )

        parallelize(work, n_workers, n_workers)
        return

    for step in range(buffer_size):
        for n in range(n_envs):
            _gather_row(
                d,
                s,
                d8,
                s8,
                (n * buffer_size + step) * feature_size,
                (step * n_envs + n) * feature_size,
                feature_size,
                itemsize,
            )


# ------------------------------------------------------------- ReplayBuffer.add


def replay_add(
    observations: Int,
    next_observations: Int,
    actions: Int,
    rewards: Int,
    dones: Int,
    timeouts: Int,
    obs: Int,
    next_obs: Int,
    action: Int,
    reward: Int,
    done: Int,
    timeout: Int,
    buffer_size: Int,
    n_envs: Int,
    obs_size: Int,
    action_size: Int,
    write_obs: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
    pos: Int,
    optimize_memory_usage: Int,
    handle_timeout_termination: Int,
) -> Int:
    """The body of ``ReplayBuffer.add``; returns the updated ``pos``.

    ``obs``/``next_obs``/``action`` are the ``(n_envs, *shape)`` sources the
    caller has already reshaped and cast, exactly as ``np.array(...)`` does
    before the store.  ``write_obs`` is 0 for the Dict subclass, which stores
    its observations key by key through ``dict_copy_rows``.
    """
    var row = pos * n_envs
    var obs_n = n_envs * obs_size
    var action_n = n_envs * action_size

    if write_obs != 0:
        # self.observations[self.pos] = np.array(obs)
        _gather_row(
            PtrF(unsafe_from_address=observations),
            PtrF(unsafe_from_address=obs),
            PtrU(unsafe_from_address=observations),
            PtrU(unsafe_from_address=obs),
            row * obs_size,
            0,
            obs_n,
            obs_itemsize,
        )

        if optimize_memory_usage:
            # self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
            _gather_row(
                PtrF(unsafe_from_address=observations),
                PtrF(unsafe_from_address=next_obs),
                PtrU(unsafe_from_address=observations),
                PtrU(unsafe_from_address=next_obs),
                ((pos + 1) % buffer_size) * n_envs * obs_size,
                0,
                obs_n,
                obs_itemsize,
            )
        else:
            # self.next_observations[self.pos] = np.array(next_obs)
            _gather_row(
                PtrF(unsafe_from_address=next_observations),
                PtrF(unsafe_from_address=next_obs),
                PtrU(unsafe_from_address=next_observations),
                PtrU(unsafe_from_address=next_obs),
                row * obs_size,
                0,
                obs_n,
                obs_itemsize,
            )

    # self.actions[self.pos] = np.array(action)
    _gather_row(
        PtrF(unsafe_from_address=actions),
        PtrF(unsafe_from_address=action),
        PtrU(unsafe_from_address=actions),
        PtrU(unsafe_from_address=action),
        row * action_size,
        0,
        action_n,
        action_itemsize,
    )

    var r = PtrF(unsafe_from_address=rewards)
    var d = PtrF(unsafe_from_address=dones)
    var t = PtrF(unsafe_from_address=timeouts)
    var src_r = PtrF(unsafe_from_address=reward)
    var src_d = PtrF(unsafe_from_address=done)
    var src_t = PtrF(unsafe_from_address=timeout)
    for n in range(n_envs):
        r.unsafe_store(row + n, src_r.unsafe_load(n))
        d.unsafe_store(row + n, src_d.unsafe_load(n))
        if handle_timeout_termination:
            t.unsafe_store(row + n, src_t.unsafe_load(n))

    var new_pos = pos + 1
    if new_pos == buffer_size:
        new_pos = 0
    return new_pos


# --------------------------------------------------------- ReplayBuffer._get_samples


def replay_get_samples(
    observations: Int,
    next_observations: Int,
    actions: Int,
    rewards: Int,
    dones: Int,
    timeouts: Int,
    batch_inds: Int,
    env_indices: Int,
    n_batch: Int,
    dst_obs: Int,
    dst_next_obs: Int,
    dst_actions: Int,
    dst_dones: Int,
    dst_rewards: Int,
    buffer_size: Int,
    n_envs: Int,
    obs_size: Int,
    action_size: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
    optimize_memory_usage: Int,
) -> Int:
    """The body of ``ReplayBuffer._get_samples``; returns ``n_batch``.

    ``dones`` and ``rewards`` are written already reshaped to ``(-1, 1)``,
    which is the same memory as ``[n_batch]``.
    """
    var inds = PtrI(unsafe_from_address=batch_inds)
    var env_inds = PtrI(unsafe_from_address=env_indices)
    var obs = PtrF(unsafe_from_address=observations)
    var next_obs = PtrF(unsafe_from_address=next_observations)
    var act = PtrF(unsafe_from_address=actions)
    var obs8 = PtrU(unsafe_from_address=observations)
    var next_obs8 = PtrU(unsafe_from_address=next_observations)
    var act8 = PtrU(unsafe_from_address=actions)
    var out_obs = PtrF(unsafe_from_address=dst_obs)
    var out_next_obs = PtrF(unsafe_from_address=dst_next_obs)
    var out_actions = PtrF(unsafe_from_address=dst_actions)
    var out_obs8 = PtrU(unsafe_from_address=dst_obs)
    var out_next_obs8 = PtrU(unsafe_from_address=dst_next_obs)
    var out_actions8 = PtrU(unsafe_from_address=dst_actions)
    var r = PtrF(unsafe_from_address=rewards)
    var d = PtrF(unsafe_from_address=dones)
    var t = PtrF(unsafe_from_address=timeouts)
    var out_d = PtrF(unsafe_from_address=dst_dones)
    var out_r = PtrF(unsafe_from_address=dst_rewards)

    # Rows are independent -- every write lands in `b`'s own slot -- so above the
    # threshold the batch is split into contiguous chunks across workers.
    var row_bytes = (2 * obs_size * obs_itemsize) + (action_size * action_itemsize)
    var n_workers = _workers(n_batch * row_bytes)
    if n_workers > 1:
        var chunk = (n_batch + n_workers - 1) // n_workers

        def work(w: Int) {imm}:
            var lo = w * chunk
            var hi = lo + chunk
            if hi > n_batch:
                hi = n_batch
            for b in range(lo, hi):
                _replay_get_row(
                    b,
                    inds,
                    env_inds,
                    obs,
                    obs8,
                    next_obs,
                    next_obs8,
                    act,
                    act8,
                    out_obs,
                    out_obs8,
                    out_next_obs,
                    out_next_obs8,
                    out_actions,
                    out_actions8,
                    r,
                    d,
                    t,
                    out_d,
                    out_r,
                    n_envs,
                    obs_size,
                    action_size,
                    buffer_size,
                    obs_itemsize,
                    action_itemsize,
                    optimize_memory_usage,
                )

        parallelize(work, n_workers, n_workers)
        return n_batch

    for b in range(n_batch):
        _replay_get_row(
            b,
            inds,
            env_inds,
            obs,
            obs8,
            next_obs,
            next_obs8,
            act,
            act8,
            out_obs,
            out_obs8,
            out_next_obs,
            out_next_obs8,
            out_actions,
            out_actions8,
            r,
            d,
            t,
            out_d,
            out_r,
            n_envs,
            obs_size,
            action_size,
            buffer_size,
            obs_itemsize,
            action_itemsize,
            optimize_memory_usage,
        )
    return n_batch


@always_inline
def _replay_get_row(
    b: Int,
    inds: PtrI,
    env_inds: PtrI,
    obs: PtrF,
    obs8: PtrU,
    next_obs: PtrF,
    next_obs8: PtrU,
    act: PtrF,
    act8: PtrU,
    out_obs: PtrF,
    out_obs8: PtrU,
    out_next_obs: PtrF,
    out_next_obs8: PtrU,
    out_actions: PtrF,
    out_actions8: PtrU,
    r: PtrF,
    d: PtrF,
    t: PtrF,
    out_d: PtrF,
    out_r: PtrF,
    n_envs: Int,
    obs_size: Int,
    action_size: Int,
    buffer_size: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
    optimize_memory_usage: Int,
):
    """One sampled transition: every statement of ``_get_samples``' body loop."""
    var bi = Int(inds.unsafe_load(b))
    var ei = Int(env_inds.unsafe_load(b))
    var row = bi * n_envs + ei

    if optimize_memory_usage:
        # self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        _gather_row(
            out_next_obs,
            obs,
            out_next_obs8,
            obs8,
            b * obs_size,
            ((bi + 1) % buffer_size) * n_envs * obs_size + ei * obs_size,
            obs_size,
            obs_itemsize,
        )
    else:
        # self.next_observations[batch_inds, env_indices, :]
        _gather_row(
            out_next_obs,
            next_obs,
            out_next_obs8,
            next_obs8,
            b * obs_size,
            row * obs_size,
            obs_size,
            obs_itemsize,
        )

    # self.observations[batch_inds, env_indices, :]
    _gather_row(
        out_obs, obs, out_obs8, obs8, b * obs_size, row * obs_size, obs_size, obs_itemsize
    )
    # self.actions[batch_inds, env_indices, :]
    _gather_row(
        out_actions,
        act,
        out_actions8,
        act8,
        b * action_size,
        row * action_size,
        action_size,
        action_itemsize,
    )
    # Only use dones that are not due to timeouts
    # deactivated by default (timeouts is initialized as an array of False)
    out_d.unsafe_store(b, d.unsafe_load(row) * (1.0 - t.unsafe_load(row)))
    out_r.unsafe_store(b, r.unsafe_load(row))


# -------------------------------------------- RolloutBuffer.compute_returns_and_advantage


@always_inline
def _gae_block(
    next_non_terminal: PtrF,
    next_values: PtrF,
    rewards: PtrF,
    values: PtrF,
    src_base: Int,
    dst_base: Int,
    gamma: VEC,
    gae_lambda: VEC,
    last_gae_lam: VEC,
) -> VEC:
    """One vector of the per-env body of the GAE recurrence.

    ``dst_base`` indexes the reward/value row, ``src_base`` the
    next_non_terminal/next_values row.
    """
    var nnt = VEC(1.0) - next_non_terminal.unsafe_load[width=W](src_base)
    var nv = next_values.unsafe_load[width=W](src_base)
    var delta = (
        rewards.unsafe_load[width=W](dst_base) + gamma * nv * nnt - values.unsafe_load[width=W](dst_base)
    )
    return delta + gamma * gae_lambda * nnt * last_gae_lam


def compute_returns_and_advantage(
    rewards: Int,
    values: Int,
    episode_starts: Int,
    advantages: Int,
    returns: Int,
    last_values: Int,
    dones: Int,
    buffer_size: Int,
    n_envs: Int,
    gamma: Float32,
    gae_lambda: Float32,
):
    """The body of ``RolloutBuffer.compute_returns_and_advantage``.

    ``last_values`` and ``dones`` are one per env; the other arrays are
    ``[buffer_size, n_envs]``.  The recurrence runs backwards over steps, which
    is a sequential dependency, and vectorises over the ``n_envs`` axis.
    """
    var rew = PtrF(unsafe_from_address=rewards)
    var val = PtrF(unsafe_from_address=values)
    var starts = PtrF(unsafe_from_address=episode_starts)
    var adv = PtrF(unsafe_from_address=advantages)
    var ret = PtrF(unsafe_from_address=returns)
    var last_val = PtrF(unsafe_from_address=last_values)
    var done = PtrF(unsafe_from_address=dones)

    var gamma_v = VEC(gamma)
    var lambda_v = VEC(gae_lambda)

    for step in reversed(range(buffer_size)):
        # if step == self.buffer_size - 1:
        #     next_non_terminal = 1.0 - dones.astype(np.float32)
        #     next_values = last_values
        # else:
        #     next_non_terminal = 1.0 - self.episode_starts[step + 1]
        #     next_values = self.values[step + 1]
        var nn = starts
        var nv = val
        var src_row = (step + 1) * n_envs
        if step == buffer_size - 1:
            nn = done
            nv = last_val
            src_row = 0
        var base = step * n_envs
        # `last_gae_lam` for step is `last_gae_lam` for step + 1, which is
        # exactly the row this sweep wrote last; reading it back from
        # `advantages` keeps every lane in the right place and leaves no
        # loop-carried register across the blocks below.
        var carry_row = base + n_envs

        if n_envs >= W:
            var i = 0
            while i + W < n_envs:
                var carry = VEC(0.0)
                if step != buffer_size - 1:
                    carry = adv.unsafe_load[width=W](carry_row + i)
                var out = _gae_block(
                    nn, nv, rew, val, src_row + i, base + i, gamma_v, lambda_v, carry
                )
                adv.unsafe_store[width=W](base + i, out)
                i += W
            if i < n_envs:
                # The trailing block overlaps the previous one; each lane is
                # recomputed from identical inputs and an identical carried
                # value, so the overlapping stores write the same bits.
                var j = n_envs - W
                var carry = VEC(0.0)
                if step != buffer_size - 1:
                    carry = adv.unsafe_load[width=W](carry_row + j)
                var out = _gae_block(
                    nn, nv, rew, val, src_row + j, base + j, gamma_v, lambda_v, carry
                )
                adv.unsafe_store[width=W](base + j, out)
        else:
            # `n_envs < W`: one padded block, stored lane by lane.  The caller
            # guarantees W spare elements past the end of every array so the
            # block can read a whole vector; the lanes past `n_envs` are
            # discarded, and storing them would clobber the row this sweep has
            # already finalised.
            var carry = VEC(0.0)
            if step != buffer_size - 1:
                carry = adv.unsafe_load[width=W](carry_row)
            var out = _gae_block(
                nn, nv, rew, val, src_row, base, gamma_v, lambda_v, carry
            )
            var i = 0
            while i < n_envs:
                adv.unsafe_store(base + i, out[i])
                i += 1

    # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
    # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
    var n = 0
    while n + W <= buffer_size * n_envs:
        ret.unsafe_store[width=W](n, adv.unsafe_load[width=W](n) + val.unsafe_load[width=W](n))
        n += W
    while n < buffer_size * n_envs:
        ret.unsafe_store(n, adv.unsafe_load(n) + val.unsafe_load(n))
        n += 1


# ------------------------------------------------------------- RolloutBuffer.add


def rollout_add(
    observations: Int,
    actions: Int,
    rewards: Int,
    episode_starts: Int,
    values: Int,
    log_probs: Int,
    obs: Int,
    action: Int,
    reward: Int,
    episode_start: Int,
    value: Int,
    log_prob: Int,
    n_envs: Int,
    obs_size: Int,
    write_obs: Int,
    action_size: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
    pos: Int,
    buffer_size: Int,
) -> Int:
    """The body of ``RolloutBuffer.add``; returns the updated ``pos``."""
    var row = pos * n_envs

    # `write_obs` is 0 for the Dict subclass, which stores its observations key
    # by key through `dict_copy_rows`.
    if write_obs != 0:
        # self.observations[self.pos] = np.array(obs)
        _gather_row(
            PtrF(unsafe_from_address=observations),
            PtrF(unsafe_from_address=obs),
            PtrU(unsafe_from_address=observations),
            PtrU(unsafe_from_address=obs),
            row * obs_size,
            0,
            n_envs * obs_size,
            obs_itemsize,
        )
    # self.actions[self.pos] = np.array(action)
    _gather_row(
        PtrF(unsafe_from_address=actions),
        PtrF(unsafe_from_address=action),
        PtrU(unsafe_from_address=actions),
        PtrU(unsafe_from_address=action),
        row * action_size,
        0,
        n_envs * action_size,
        action_itemsize,
    )
    # self.rewards[self.pos] = np.array(reward)
    _gather_row(
        PtrF(unsafe_from_address=rewards),
        PtrF(unsafe_from_address=reward),
        PtrU(unsafe_from_address=rewards),
        PtrU(unsafe_from_address=reward),
        row,
        0,
        n_envs,
        4,
    )
    # self.episode_starts[self.pos] = np.array(episode_start)
    _gather_row(
        PtrF(unsafe_from_address=episode_starts),
        PtrF(unsafe_from_address=episode_start),
        PtrU(unsafe_from_address=episode_starts),
        PtrU(unsafe_from_address=episode_start),
        row,
        0,
        n_envs,
        4,
    )
    # self.values[self.pos] = value.clone().cpu().numpy().flatten()
    _gather_row(
        PtrF(unsafe_from_address=values),
        PtrF(unsafe_from_address=value),
        PtrU(unsafe_from_address=values),
        PtrU(unsafe_from_address=value),
        row,
        0,
        n_envs,
        4,
    )
    # self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
    _gather_row(
        PtrF(unsafe_from_address=log_probs),
        PtrF(unsafe_from_address=log_prob),
        PtrU(unsafe_from_address=log_probs),
        PtrU(unsafe_from_address=log_prob),
        row,
        0,
        n_envs,
        4,
    )

    return pos + 1


# --------------------------------------------------------- RolloutBuffer._get_samples


def rollout_get_samples(
    observations: Int,
    actions: Int,
    values: Int,
    log_probs: Int,
    advantages: Int,
    returns: Int,
    batch_inds: Int,
    n_batch: Int,
    dst_obs: Int,
    dst_actions: Int,
    dst_values: Int,
    dst_log_probs: Int,
    dst_advantages: Int,
    dst_returns: Int,
    obs_size: Int,
    action_size: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
) -> Int:
    """The body of ``RolloutBuffer._get_samples``; returns ``n_batch``.

    The arrays are the post-``swap_and_flatten`` ones, indexed by
    ``batch_inds``; ``actions`` is cast to float32 as upstream does.
    """
    var inds = PtrI(unsafe_from_address=batch_inds)
    var obs = PtrF(unsafe_from_address=observations)
    var obs8 = PtrU(unsafe_from_address=observations)
    var act32 = PtrF(unsafe_from_address=actions)
    var act8 = Pointer[Int8, AnyOrigin[mut=True]](unsafe_from_address=actions)
    var act16 = PtrH(unsafe_from_address=actions)
    var act64 = PtrI(unsafe_from_address=actions)
    var val = PtrF(unsafe_from_address=values)
    var log_p = PtrF(unsafe_from_address=log_probs)
    var adv = PtrF(unsafe_from_address=advantages)
    var ret = PtrF(unsafe_from_address=returns)
    var out_obs = PtrF(unsafe_from_address=dst_obs)
    var out_obs8 = PtrU(unsafe_from_address=dst_obs)
    var out_actions = PtrF(unsafe_from_address=dst_actions)
    var out_values = PtrF(unsafe_from_address=dst_values)
    var out_log_probs = PtrF(unsafe_from_address=dst_log_probs)
    var out_advantages = PtrF(unsafe_from_address=dst_advantages)
    var out_returns = PtrF(unsafe_from_address=dst_returns)

    for b in range(n_batch):
        var i = Int(inds.unsafe_load(b))
        # self.observations[batch_inds]
        _gather_row(
            out_obs, obs, out_obs8, obs8, b * obs_size, i * obs_size, obs_size, obs_itemsize
        )
        # self.actions[batch_inds].astype(np.float32, copy=False)
        _cast_row(
            out_actions.unsafe_offset(b * action_size),
            act32,
            act8,
            act16,
            act64,
            i * action_size,
            action_size,
            action_itemsize,
        )
        out_values.unsafe_store(b, val.unsafe_load(i))
        out_log_probs.unsafe_store(b, log_p.unsafe_load(i))
        out_advantages.unsafe_store(b, adv.unsafe_load(i))
        out_returns.unsafe_store(b, ret.unsafe_load(i))
    return n_batch


# --------------------------------------------------------- DictReplayBuffer.add


def dict_copy_row(dst: Int, src: Int, count: Int, itemsize: Int, pos: Int):
    """One key of the observation store of ``DictReplayBuffer.add`` /
    ``DictRolloutBuffer.add``: ``self.observations[key][self.pos] = np.array(obs[key])``.

    ``count`` is ``n_envs * prod(obs_shape[key])``, the length of one stored row,
    so the row starts at ``pos * count`` elements.  The loop over keys is upstream's Python
    ``for key in self.observations.keys()``, so the caller runs this once per
    key; that also keeps every pointer straight-line, which this toolchain
    requires.
    """
    _gather_row(
        PtrF(unsafe_from_address=dst),
        PtrF(unsafe_from_address=src),
        PtrU(unsafe_from_address=dst),
        PtrU(unsafe_from_address=src),
        pos * count,
        0,
        count,
        itemsize,
    )


# -------------------------------------------------- DictReplayBuffer._get_samples


def dict_replay_gather_key(
    observations: Int,
    dst: Int,
    count: Int,
    itemsize: Int,
    batch_inds: Int,
    env_indices: Int,
    n_batch: Int,
    buffer_size: Int,
    n_envs: Int,
) -> Int:
    """One key of ``self.observations[key][batch_inds, env_indices, :]``.

    ``count`` is ``prod(obs_shape[key])``, the length of one row.
    """
    var inds = PtrI(unsafe_from_address=batch_inds)
    var env_inds = PtrI(unsafe_from_address=env_indices)
    var obs = PtrF(unsafe_from_address=observations)
    var obs8 = PtrU(unsafe_from_address=observations)
    var out = PtrF(unsafe_from_address=dst)
    var out8 = PtrU(unsafe_from_address=dst)
    for b in range(n_batch):
        var row = Int(inds.unsafe_load(b)) * n_envs + Int(env_inds.unsafe_load(b))
        _gather_row(out, obs, out8, obs8, b * count, row * count, count, itemsize)
    return n_batch


def dict_replay_get_samples(
    actions: Int,
    rewards: Int,
    dones: Int,
    timeouts: Int,
    batch_inds: Int,
    env_indices: Int,
    n_batch: Int,
    buffer_size: Int,
    n_envs: Int,
    action_size: Int,
    action_itemsize: Int,
    dst_actions: Int,
    dst_dones: Int,
    dst_rewards: Int,
) -> Int:
    """The non-observation part of ``DictReplayBuffer._get_samples``.

    The observation gathers are :func:`dict_replay_gather_key`, one call per
    key, mirroring upstream's per-key dict comprehension.
    """
    var inds = PtrI(unsafe_from_address=batch_inds)
    var env_inds = PtrI(unsafe_from_address=env_indices)
    var act = PtrF(unsafe_from_address=actions)
    var act8 = PtrU(unsafe_from_address=actions)
    var out_actions = PtrF(unsafe_from_address=dst_actions)
    var out_actions8 = PtrU(unsafe_from_address=dst_actions)
    var r = PtrF(unsafe_from_address=rewards)
    var d = PtrF(unsafe_from_address=dones)
    var t = PtrF(unsafe_from_address=timeouts)
    var out_d = PtrF(unsafe_from_address=dst_dones)
    var out_r = PtrF(unsafe_from_address=dst_rewards)

    for b in range(n_batch):
        var row = Int(inds.unsafe_load(b)) * n_envs + Int(env_inds.unsafe_load(b))
        # self.actions[batch_inds, env_indices]
        _gather_row(
            out_actions,
            act,
            out_actions8,
            act8,
            b * action_size,
            row * action_size,
            action_size,
            action_itemsize,
        )
        # Only use dones that are not due to timeouts
        # deactivated by default (timeouts is initialized as an array of False)
        out_d.unsafe_store(b, d.unsafe_load(row) * (1.0 - t.unsafe_load(row)))
        out_r.unsafe_store(b, r.unsafe_load(row))
    return n_batch


# --------------------------------------------------- DictRolloutBuffer._get_samples


def dict_flat_gather(
    observations: Int,
    dst: Int,
    count: Int,
    itemsize: Int,
    batch_inds: Int,
    n_batch: Int,
) -> Int:
    """One key of ``self.observations[key][batch_inds]`` after
    ``swap_and_flatten``, where the arrays are indexed by a flat row number."""
    var inds = PtrI(unsafe_from_address=batch_inds)
    var obs = PtrF(unsafe_from_address=observations)
    var obs8 = PtrU(unsafe_from_address=observations)
    var out = PtrF(unsafe_from_address=dst)
    var out8 = PtrU(unsafe_from_address=dst)
    for b in range(n_batch):
        var i = Int(inds.unsafe_load(b))
        _gather_row(out, obs, out8, obs8, b * count, i * count, count, itemsize)
    return n_batch


# -------------------------------------------------------- NStepReplayBuffer._get_samples


def n_step_get_samples(
    observations: Int,
    next_observations: Int,
    actions: Int,
    rewards: Int,
    dones: Int,
    timeouts: Int,
    batch_inds: Int,
    env_indices: Int,
    n_batch: Int,
    n_steps: Int,
    buffer_size: Int,
    n_envs: Int,
    obs_size: Int,
    action_size: Int,
    obs_itemsize: Int,
    action_itemsize: Int,
    gamma: Float64,
    dst_obs: Int,
    dst_next_obs: Int,
    dst_actions: Int,
    dst_dones: Int,
    dst_rewards: Int,
    dst_discounts: Int,
    scratch: Int,
) -> Int:
    """The body of ``NStepReplayBuffer._get_samples``; returns ``n_batch``.

    ``scratch`` holds ``n_batch * n_steps`` float32 for ``rewards_seq``; the
    index array is folded into the same loop instead of being materialised,
    which is the only structural change (upstream builds it with
    ``(batch_inds[:, None] + steps) % buffer_size``).
    """
    var inds = PtrI(unsafe_from_address=batch_inds)
    var env_inds = PtrI(unsafe_from_address=env_indices)
    var obs = PtrF(unsafe_from_address=observations)
    var obs8 = PtrU(unsafe_from_address=observations)
    var next_obs = PtrF(unsafe_from_address=next_observations)
    var next_obs8 = PtrU(unsafe_from_address=next_observations)
    var act = PtrF(unsafe_from_address=actions)
    var act8 = PtrU(unsafe_from_address=actions)
    var r = PtrF(unsafe_from_address=rewards)
    var d = PtrF(unsafe_from_address=dones)
    var t = PtrF(unsafe_from_address=timeouts)
    var buf = PtrF(unsafe_from_address=scratch)
    var out_obs = PtrF(unsafe_from_address=dst_obs)
    var out_obs8 = PtrU(unsafe_from_address=dst_obs)
    var out_next_obs = PtrF(unsafe_from_address=dst_next_obs)
    var out_next_obs8 = PtrU(unsafe_from_address=dst_next_obs)
    var out_actions = PtrF(unsafe_from_address=dst_actions)
    var out_actions8 = PtrU(unsafe_from_address=dst_actions)
    var out_d = PtrF(unsafe_from_address=dst_dones)
    var out_r = PtrF(unsafe_from_address=dst_rewards)
    var out_disc = PtrF(unsafe_from_address=dst_discounts)
    var gamma_d = gamma

    for b in range(n_batch):
        var bi = Int(inds.unsafe_load(b))
        var ei = Int(env_inds.unsafe_load(b))

        # rewards_seq = self.rewards[indices, env_indices[:, None]]
        for j in range(n_steps):
            var row = ((bi + j) % buffer_size) * n_envs + ei
            buf.unsafe_store(b * n_steps + j, r.unsafe_load(row))

        # done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        # done_idx = done_or_truncated.argmax(axis=1)
        var done_idx = 0
        var found = False
        for j in range(n_steps):
            var row = ((bi + j) % buffer_size) * n_envs + ei
            if d.unsafe_load(row) != 0.0 or t.unsafe_load(row) != 0.0:
                done_idx = j
                found = True
                break
        # If no done/truncation, keep full sequence
        if not found:
            done_idx = n_steps - 1

        # mask = np.arange(self.n_steps) <= done_idx
        # target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)
        out_disc.unsafe_store(b, Float32(pow(gamma_d, Float64(done_idx + 1))))

        # discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32)
        # discounted_rewards = rewards_seq * discounts * mask
        var n_step_return = Float32(0.0)
        for j in range(done_idx + 1):
            n_step_return += buf.unsafe_load(b * n_steps + j) * Float32(
                pow(gamma_d, Float64(j))
            )
        out_r.unsafe_store(b, n_step_return)

        # last_indices = (batch_inds + done_idx) % self.buffer_size
        var last_row = ((bi + done_idx) % buffer_size) * n_envs + ei
        # next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        _gather_row(
            out_next_obs,
            next_obs,
            out_next_obs8,
            next_obs8,
            b * obs_size,
            last_row * obs_size,
            obs_size,
            obs_itemsize,
        )
        # final_dones = next_dones * (1.0 - next_timeouts)
        out_d.unsafe_store(b, d.unsafe_load(last_row) * (1.0 - t.unsafe_load(last_row)))
        # obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        _gather_row(
            out_obs, obs, out_obs8, obs8, b * obs_size, (bi * n_envs + ei) * obs_size, obs_size, obs_itemsize
        )
        # actions = self.actions[batch_inds, env_indices]
        _gather_row(
            out_actions,
            act,
            out_actions8,
            act8,
            b * action_size,
            (bi * n_envs + ei) * action_size,
            action_size,
            action_itemsize,
        )
    return n_batch
