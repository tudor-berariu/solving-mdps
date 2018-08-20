from typing import Callable
import numpy as np

from .closed_form import cf_policy_eval, cf_state_dist
from envs import MDP
from policies import Policy


def mean_squared_value_error(env: MDP,
                             policy: Policy,
                             values: np.ndarray = None,
                             state_dist: np.ndarray = None,
                             gamma: float = 1.0,
                             ) -> Callable[[np.ndarray], float]:
    if values is None:
        values = cf_policy_eval(env, policy, gamma)
    else:
        assert values.shape == (env.n_nstates,)

    if state_dist is None:
        state_dist = cf_state_dist(env, policy, gamma)
    else:
        assert state_dist.shape == (env.n_nstates,)

    def cost_function(predictions: np.ndarray) -> float:
        diff = predictions - values
        return np.dot(diff * diff, state_dist)

    return cost_function


def bellman_error(env: MDP,
                  policy: Policy,
                  state_dist: np.ndarray = None,
                  gamma: float = 1.0,
                  ) -> Callable[[np.ndarray], float]:
    if state_dist is None:
        state_dist = cf_state_dist(env, policy, gamma)
    else:
        assert state_dist.shape == (env.n_nstates,)

    _, dynamics, rewards = env.mdp
    trans = (dynamics * policy.probs[:, :, None]).sum(axis=1)
    padding = np.zeros((env.n_states - env.n_nstates,))

    def cost_function(nvalues: np.ndarray) -> float:
        values = np.concatenate((nvalues, padding))
        td = (rewards + gamma * values[None, :] - nvalues[:, None])
        exp_td = (td * trans).sum(axis=1)
        return np.dot(exp_td * exp_td, state_dist)

    return cost_function


def td_error(env: MDP,
             policy: Policy,
             state_dist: np.ndarray = None,
             gamma: float = 1.0,
             ) -> Callable[[np.ndarray], float]:

    if state_dist is None:
        state_dist = cf_state_dist(env, policy, gamma)
    else:
        assert state_dist.shape == (env.n_nstates,)

    _, dynamics, rewards = env.mdp
    trans = (dynamics * policy.probs[:, :, None]).sum(axis=1)
    padding = np.zeros((env.n_states - env.n_nstates,))

    def cost_function(nvalues: np.ndarray) -> float:
        values = np.concatenate((nvalues, padding))
        td = (rewards + gamma * values[None, :] - nvalues[:, None])
        exp_td = (td * td * trans).sum(axis=1)
        return np.dot(exp_td, state_dist)

    return cost_function
