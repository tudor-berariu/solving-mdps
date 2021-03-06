from typing import List, Union
import numpy as np

from envs import MDP
from policies import Policy


def dp_state_dist(env: MDP, policy: Policy,
                  features: np.ndarray = None,
                  policy_uses_features: bool = False,
                  max_diff: float = 1e-7) -> np.ndarray:
    n_nstates = env.n_nstates
    init_state_dist, dynamics, _ = env.mdp
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics[:, :, :n_nstates] * probs[:, :, None]).sum(axis=1)
    state_dist = np.copy(init_state_dist)
    while True:
        new_state_dist = init_state_dist + env.gamma * state_dist @ trans
        diff = np.max(np.abs(new_state_dist - state_dist))
        if diff < max_diff:
            break
        state_dist = new_state_dist
    return new_state_dist / np.sum(new_state_dist)


def dp_policy_eval(env: MDP, policy: Policy,
                   features: np.ndarray = None,
                   policy_uses_features: bool = False,
                   max_diff: float = 1e-7,
                   keeptrace: bool = False
                   ) -> Union[np.ndarray, List[np.ndarray]]:
    n_nstates = env.n_nstates
    _, dynamics, rewards = env.mdp
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics * probs[:, :, None]).sum(axis=1)
    ntrans = trans[:, :n_nstates]
    values = mean_rewards = (trans * rewards).sum(axis=1)
    if keeptrace:
        trace = [values]
    while True:
        new_values = mean_rewards + env.gamma * ntrans @ values
        if keeptrace:
            trace.append(new_values)
        diff = np.max(np.abs(new_values - values))
        if diff < max_diff:
            break
        values = new_values
    return trace if keeptrace else new_values
