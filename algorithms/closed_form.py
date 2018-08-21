import numpy as np
import numpy.linalg as la

from envs import MDP
from policies import Policy


def cf_state_dist(env: MDP,
                  policy: Policy) -> np.ndarray:
    """Closed form solution for state occupancy:
        - eta(s) = (I - gamma * T)^-1 h
        - mu(s) = eta(s)/sum(eta)
    """
    n_nstates = env.n_nstates
    init_state_dist, dynamics, _ = env.mdp
    trans = (dynamics[:, :, :n_nstates] * policy.probs[:, :, None]).sum(axis=1)
    visits = la.inv(np.eye(n_nstates) - env.gamma * trans.T) @ init_state_dist
    print(np.sum(visits))
    print(1.0 / (1.0 - env.gamma))
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
                                 gamma: float = 1.0) -> np.ndarray:
    assert features.shape[0] == env.n_nstates
    values = cf_policy_eval(env, policy, gamma)
    state_dist = cf_state_dist(env, policy, gamma)
    w_feats = features.T @ np.diag(state_dist)
    return la.inv(w_feats @ features) @ w_feats @ values


def cf_policy_eval_min_tderror(env: MDP,
                               policy: Policy,
                               features: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
    assert features.shape[0] == env.n_nstates
    _, dynamics, rewards = env.mdp
    padding = np.zeros((env.n_states - env.n_nstates, features.shape[1]))
    trans = (dynamics * policy.probs[:, :, None]).sum(axis=1)
    nfeatures = features
    features = np.concatenate((nfeatures, padding), axis=0)
    ntrans = trans[:, :env.n_nstates]
    mu = cf_state_dist(env, policy, gamma)
    diag = np.diag(mu)
    mut = diag @ ntrans
    A = (gamma ** 2) * np.diag(mu @ ntrans) - gamma * (mut + mut.T) + diag
    A = nfeatures.T @ A @ nfeatures
    tr = trans * rewards
    b = gamma * features.T @ tr.T @ mu
    b -= (nfeatures.T @ np.diag(mu) @ tr).sum(axis=1)
    return -la.inv(A) @ b


def cf_policy_eval_min_bellman(env: MDP,
                               policy: Policy,
                               features: np.ndarray,
                               gamma: float = 1.0) -> np.ndarray:
    _, dynamics, rewards = env.mdp
    trans = (dynfomamics[:, :, :env.n_nstates] * policy.probs[:, :, None]).sum(axis=1)
    xt = (gamma * trans - np.eye(env.n_nstates)) @ features
