import numpy as np
import numpy.linalg as la

from envs import MDP
from policies import Policy


def cf_state_dist(env: MDP, policy: Policy,
                  gamma: float = 1.0) -> np.ndarray:
    n_nstates = env.n_nstates
    init_state_dist, dynamics, _ = env.mdp
    trans = (dynamics[:, :, :n_nstates] * policy.probs[:, :, None]).sum(axis=1)
    visits = la.inv(np.eye(n_nstates) - gamma * trans.T) @ init_state_dist
    return visits / np.sum(visits)


def cf_policy_eval(env: MDP, policy: Policy,
                   gamma: float = 1.0) -> np.ndarray:
    n_nstates = env.n_nstates
    _, dynamics, rewards = env.mdp
    trans = (dynamics * policy.probs[:, :, None]).sum(axis=1)
    ntrans = trans[:, :n_nstates]
    inv_trans = la.inv(np.eye(n_nstates) - gamma * ntrans)
    return inv_trans @ (trans * rewards).sum(axis=1)


def cf_policy_eval_linear_approx(env: MDP,
                                 policy: Policy,
                                 features: np.ndarray,
                                 gamma: float = 1.0):
    assert features.shape[0] == env.n_nstates
    values = cf_policy_eval(env, policy, gamma)
    state_dist = cf_state_dist(env, policy, gamma)
    w_feats = features.T @ np.diag(state_dist)
    return la.inv(w_feats @ features) @ w_feats @ values


def cf_min_tderror_policy_eval(env: MDP,
                               policy: Policy,
                               features: np.ndarray,
                               gamma: float = 1.0):
    """
    values = X @ w
    A1 = gamma * gamma * X.transpose(0, 1) @ torch.diag(mu @ T) @ X
    A2 = - gamma * X[:nn].transpose(0, 1) @ diagMu @ T @ X
    A3 = - gamma * (T @ X).transpose(0, 1) @ diagMu @ X[:nn]
    A4 = X[:nn].transpose(0, 1) @ diagMu @ X[:nn]
    A = A1 + A2 + A3 + A4

    b = gamma * mu @ (T * R) @ X - diagMu @ (T * R) @ ones @ X[:nn]
    grad = 2 * (A @ w + b)
    """

    raise NotImplementedError
