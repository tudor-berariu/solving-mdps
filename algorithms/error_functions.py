from typing import Callable
import numpy as np

from envs import MDP
from policies import Policy
from .closed_form import cf_policy_eval, cf_state_dist


def mean_squared_value_error(env: MDP,
                             policy: Policy,
                             features: np.ndarray = None,
                             policy_uses_features: bool = False,
                             values: np.ndarray = None,
                             state_dist: np.ndarray = None,
                             ) -> Callable[[np.ndarray], float]:
    if values is None:
        values = cf_policy_eval(env, policy, features, policy_uses_features)
    else:
        assert values.shape == (env.n_nstates,)

    if state_dist is None:
        state_dist = cf_state_dist(env, policy, features, policy_uses_features)
    else:
        assert state_dist.shape == (env.n_nstates,)

    def cost_function(predictions: np.ndarray) -> float:
        diff = predictions - values
        return np.dot(diff * diff, state_dist)

    return cost_function


def bellman_error(env: MDP,
                  policy: Policy,
                  features: np.ndarray = None,
                  policy_uses_features: bool = False,
                  state_dist: np.ndarray = None
                  ) -> Callable[[np.ndarray], float]:
    if state_dist is None:
        state_dist = cf_state_dist(env, policy)
    else:
        assert state_dist.shape == (env.n_nstates,)
    gamma = env.gamma
    _, dynamics, rewards = env.mdp
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics * probs[:, :, None]).sum(axis=1)
    padding = np.zeros((env.n_states - env.n_nstates,))

    def cost_function(nvalues: np.ndarray) -> float:
        values = np.concatenate((nvalues, padding))
        tdiff = (rewards + gamma * values[None, :] - nvalues[:, None])
        exp_td = (tdiff * trans).sum(axis=1)
        return np.dot(exp_td * exp_td, state_dist)

    return cost_function


def td_error(env: MDP,
             policy: Policy,
             features: np.ndarray = None,
             policy_uses_features: bool = False,
             state_dist: np.ndarray = None
             ) -> Callable[[np.ndarray], float]:

    if state_dist is None:
        state_dist = cf_state_dist(env, policy, features, policy_uses_features)
    else:
        assert state_dist.shape == (env.n_nstates,)
    gamma = env.gamma
    _, dynamics, rewards = env.mdp
    probs = policy.probs(features if policy_uses_features else None)

    trans = (dynamics * probs[:, :, None]).sum(axis=1)
    padding = np.zeros((env.n_states - env.n_nstates,))

    def cost_function(nvalues: np.ndarray) -> float:
        values = np.concatenate((nvalues, padding))
        tdiff = (rewards + gamma * values[None, :] - nvalues[:, None])
        exp_td = (tdiff * tdiff * trans).sum(axis=1)
        return np.dot(exp_td, state_dist)

    return cost_function
