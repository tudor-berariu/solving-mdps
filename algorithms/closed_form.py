import numpy as np
import numpy.linalg as la

from envs import MDP
from policies import Policy


def cf_state_dist(env: MDP,
                  policy: Policy,
                  features: np.ndarray = None,
                  policy_uses_features: bool = False) -> np.ndarray:
    """Closed form solution for state occupancy:
        - eta(s) = (I - gamma * T)^-1 h
        - mu(s) = eta(s)/sum(eta)
    """
    n_nstates = env.n_nstates
    init_state_dist, dynamics, _ = env.mdp
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics[:, :, :n_nstates] * probs[:, :, None]).sum(axis=1)
    visits = la.inv(np.eye(n_nstates) - env.gamma * trans.T) @ init_state_dist
    return visits / np.sum(visits)


def cf_policy_eval(env: MDP,
                   policy: Policy,
                   features: np.ndarray = None,
                   policy_uses_features: bool = False) -> np.ndarray:
    """Closed form solution for policy-evaluation
       v = (I - gamma * T)^-1 @ (T * R) @ 1
    """
    n_nstates = env.n_nstates
    _, dynamics, rewards = env.mdp
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics * probs[:, :, None]).sum(axis=1)
    ntrans = trans[:, :n_nstates]
    inv_trans = la.inv(np.eye(n_nstates) - env.gamma * ntrans)
    return inv_trans @ (trans * rewards).sum(axis=1)


def cf_minimum_msve(env: MDP,
                    policy: Policy,
                    features: np.ndarray = None,
                    policy_uses_features: bool = False) -> np.ndarray:
    values = cf_policy_eval(env, policy, features, policy_uses_features)
    state_dist = cf_state_dist(env, policy, features, policy_uses_features)
    if features is None:
        features = np.eye(env.n_nstates)
    assert features.shape[0] == env.n_nstates
    w_feats = features.T @ np.diag(state_dist)
    return la.inv(w_feats @ features) @ w_feats @ values


def cf_minimum_tderror(env: MDP,
                       policy: Policy,
                       features: np.ndarray = None,
                       policy_uses_features: bool = False) -> np.ndarray:
    mu = cf_state_dist(env, policy, features, policy_uses_features)
    if features is None:
        features = np.eye(env.n_nstates)
    assert features.shape[0] == env.n_nstates
    gamma = env.gamma
    _, dynamics, rewards = env.mdp
    padding = np.zeros((env.n_states - env.n_nstates, features.shape[1]))
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics * probs[:, :, None]).sum(axis=1)
    nfeatures = features
    features = np.concatenate((nfeatures, padding), axis=0)
    ntrans = trans[:, :env.n_nstates]
    diag = np.diag(mu)
    mut = diag @ ntrans
    A = (gamma ** 2) * np.diag(mu @ ntrans) - gamma * (mut + mut.T) + diag
    A = nfeatures.T @ A @ nfeatures
    tr = trans * rewards
    b = gamma * features.T @ tr.T @ mu
    b -= (nfeatures.T @ np.diag(mu) @ tr).sum(axis=1)
    return -la.inv(A) @ b


def cf_minimum_bellman(env: MDP,
                       policy: Policy,
                       features: np.ndarray = None,
                       policy_uses_features: bool = False) -> np.ndarray:
    mu = cf_state_dist(env, policy, features, policy_uses_features)
    if features is None:
        features = np.eye(env.n_nstates)
    assert features.shape[0] == env.n_nstates
    _, dynamics, rewards = env.mdp
    diag = np.diag(mu)
    probs = policy.probs(features if policy_uses_features else None)
    trans = (dynamics[:, :, :] * probs[:, :, None]).sum(axis=1)
    ntrans = trans[:, :env.n_nstates]
    xt = (env.gamma * ntrans - np.eye(env.n_nstates)) @ features
    A = xt.T @ diag @ xt
    b = (trans * rewards).sum(axis=1) @ diag @ xt
    return -la.inv(A) @ b
