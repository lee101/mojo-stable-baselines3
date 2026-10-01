"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address and every float scalar as a `Float64`,
because `@export` rejects pointers and implicit origins.  The wrappers below are
the only place that knowledge lives: each one is a straight `Int` -> `Int`
forward to the kernel of the same name in `ported.mojo`, and the kernels
themselves re-type the addresses internally.
"""

from std.sys import simd_width_of

from ported import (
    compute_returns_and_advantage,
    dict_copy_row,
    dict_flat_gather,
    dict_replay_gather_key,
    dict_replay_get_samples,
    n_step_get_samples,
    replay_add,
    replay_get_samples,
    rollout_add,
    rollout_get_samples,
    swap_and_flatten,
)


@export("sb3_simd_width")
def sb3_simd_width() abi("C") -> Int:
    """The float32 SIMD width the kernels assume; callers pad arrays by it."""
    return simd_width_of[DType.float32]()


@export("sb3_swap_and_flatten")
def sb3_swap_and_flatten(
    src: Int,
    dst: Int,
    buffer_size: Int,
    n_envs: Int,
    feature_size: Int,
    itemsize: Int,
) abi("C"):
    swap_and_flatten(src, dst, buffer_size, n_envs, feature_size, itemsize)


@export("sb3_replay_add")
def sb3_replay_add(args: Int) abi("C") -> Int:
    """``replay_add`` with its arguments packed into one int64 block.

    ctypes costs about 0.24 us per scalar argument on this boundary, so the
    twenty-two of them were roughly a third of a `ReplayBuffer.add` call -- the
    one entry point a training loop runs once per environment step.  The block
    is written in the order `ReplayBuffer.add` lists it; the kernel itself is
    untouched and still takes the arguments one by one.
    """
    var p = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=args)
    return replay_add(
        Int(p.unsafe_load(0)),
        Int(p.unsafe_load(1)),
        Int(p.unsafe_load(2)),
        Int(p.unsafe_load(3)),
        Int(p.unsafe_load(4)),
        Int(p.unsafe_load(5)),
        Int(p.unsafe_load(6)),
        Int(p.unsafe_load(7)),
        Int(p.unsafe_load(8)),
        Int(p.unsafe_load(9)),
        Int(p.unsafe_load(10)),
        Int(p.unsafe_load(11)),
        Int(p.unsafe_load(12)),
        Int(p.unsafe_load(13)),
        Int(p.unsafe_load(14)),
        Int(p.unsafe_load(15)),
        Int(p.unsafe_load(16)),
        Int(p.unsafe_load(17)),
        Int(p.unsafe_load(18)),
        Int(p.unsafe_load(19)),
        Int(p.unsafe_load(20)),
        Int(p.unsafe_load(21)),
    )


@export("sb3_replay_get_samples")
def sb3_replay_get_samples(
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
) abi("C") -> Int:
    return replay_get_samples(
        observations,
        next_observations,
        actions,
        rewards,
        dones,
        timeouts,
        batch_inds,
        env_indices,
        n_batch,
        dst_obs,
        dst_next_obs,
        dst_actions,
        dst_dones,
        dst_rewards,
        buffer_size,
        n_envs,
        obs_size,
        action_size,
        obs_itemsize,
        action_itemsize,
        optimize_memory_usage,
    )


@export("sb3_compute_returns_and_advantage")
def sb3_compute_returns_and_advantage(
    rewards: Int,
    values: Int,
    episode_starts: Int,
    advantages: Int,
    returns: Int,
    last_values: Int,
    dones: Int,
    buffer_size: Int,
    n_envs: Int,
    gamma: Float64,
    gae_lambda: Float64,
) abi("C"):
    compute_returns_and_advantage(
        rewards,
        values,
        episode_starts,
        advantages,
        returns,
        last_values,
        dones,
        buffer_size,
        n_envs,
        Float32(gamma),
        Float32(gae_lambda),
    )


@export("sb3_rollout_add")
def sb3_rollout_add(
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
) abi("C") -> Int:
    return rollout_add(
        observations,
        actions,
        rewards,
        episode_starts,
        values,
        log_probs,
        obs,
        action,
        reward,
        episode_start,
        value,
        log_prob,
        n_envs,
        obs_size,
        write_obs,
        action_size,
        obs_itemsize,
        action_itemsize,
        pos,
        buffer_size,
    )


@export("sb3_rollout_get_samples")
def sb3_rollout_get_samples(
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
) abi("C") -> Int:
    return rollout_get_samples(
        observations,
        actions,
        values,
        log_probs,
        advantages,
        returns,
        batch_inds,
        n_batch,
        dst_obs,
        dst_actions,
        dst_values,
        dst_log_probs,
        dst_advantages,
        dst_returns,
        obs_size,
        action_size,
        obs_itemsize,
        action_itemsize,
    )


@export("sb3_n_step_get_samples")
def sb3_n_step_get_samples(
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
) abi("C") -> Int:
    return n_step_get_samples(
        observations,
        next_observations,
        actions,
        rewards,
        dones,
        timeouts,
        batch_inds,
        env_indices,
        n_batch,
        n_steps,
        buffer_size,
        n_envs,
        obs_size,
        action_size,
        obs_itemsize,
        action_itemsize,
        gamma,
        dst_obs,
        dst_next_obs,
        dst_actions,
        dst_dones,
        dst_rewards,
        dst_discounts,
        scratch,
    )


@export("sb3_dict_copy_row")
def sb3_dict_copy_row(
    dst: Int, src: Int, count: Int, itemsize: Int, pos: Int
) abi("C"):
    dict_copy_row(dst, src, count, itemsize, pos)


@export("sb3_dict_replay_gather_key")
def sb3_dict_replay_gather_key(
    observations: Int,
    dst: Int,
    count: Int,
    itemsize: Int,
    batch_inds: Int,
    env_indices: Int,
    n_batch: Int,
    buffer_size: Int,
    n_envs: Int,
) abi("C") -> Int:
    return dict_replay_gather_key(
        observations,
        dst,
        count,
        itemsize,
        batch_inds,
        env_indices,
        n_batch,
        buffer_size,
        n_envs,
    )


@export("sb3_dict_replay_get_samples")
def sb3_dict_replay_get_samples(
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
) abi("C") -> Int:
    return dict_replay_get_samples(
        actions,
        rewards,
        dones,
        timeouts,
        batch_inds,
        env_indices,
        n_batch,
        buffer_size,
        n_envs,
        action_size,
        action_itemsize,
        dst_actions,
        dst_dones,
        dst_rewards,
    )


@export("sb3_dict_flat_gather")
def sb3_dict_flat_gather(
    observations: Int,
    dst: Int,
    count: Int,
    itemsize: Int,
    batch_inds: Int,
    n_batch: Int,
) abi("C") -> Int:
    return dict_flat_gather(observations, dst, count, itemsize, batch_inds, n_batch)
