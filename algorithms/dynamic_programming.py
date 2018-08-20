import numpy as np

from envs import MDP
from policies import Policy


def dp_state_dist(env: MDP, policy: Policy,
                  gamma: float = 1.0,
                  max_diff: float = 1e-8) -> np.ndarray:
    n_nstates = env.n_nstates
    init_state_dist, dynamics, _ = env.mdp
    trans = (dynamics[:, :, :n_nstates] * policy.probs[:, :, None]).sum(axis=1)
    state_dist = np.copy(init_state_dist)
    while True:
        new_state_dist = init_state_dist + state_dist @ trans
        diff = np.max(np.abs(new_state_dist - state_dist))
        if diff < max_diff:
            break
        state_dist = new_state_dist
    return new_state_dist / np.sum(new_state_dist)


def dp_policy_eval(env: MDP, policy: Policy,
                   gamma: float = 1.0,
                   max_diff: float = 1e-8) -> np.ndarray:
    n_nstates = env.n_nstates
    _, dynamics, rewards = env.mdp
    trans = (dynamics * policy.probs[:, :, None]).sum(axis=1)
    ntrans = trans[:, :n_nstates]
    values = mean_rewards = (trans * rewards).sum(axis=1)
    while True:
        new_values = mean_rewards + gamma * ntrans @ values
        diff = np.max(np.abs(new_values - values))
        if diff < max_diff:
            break
        values = new_values
    return new_values
