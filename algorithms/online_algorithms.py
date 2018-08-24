from typing import Dict, List, Optional
import numpy as np

from envs import MDP
from policies import Policy
from algorithms.closed_form import cf_state_dist
from algorithms.error_functions import mean_squared_value_error
from algorithms.error_functions import bellman_error
from algorithms.error_functions import td_error
from schedulers import get_schedule


class OnlinePolicyEvaluation:

    def __init__(self,
                 env: MDP,
                 policy: Policy,
                 features: np.ndarray = None) -> None:
        self.env = env
        self.policy = policy
        if features is None:
            self.features = np.eye(env.n_nstates)
        else:
            self.features = features
        self.gamma = env.gamma

    def train(self, episodes_no: int, step: int = 1) -> Dict[str, object]:
        self.before_training()
        env = self.env
        features = self.features
        policy = self.policy
        state_dist = cf_state_dist(env, policy)
        args = [env, policy]
        kwargs = {"state_dist": state_dist}
        msve = mean_squared_value_error(*args, **kwargs)
        berr = bellman_error(*args, **kwargs)
        tderr = td_error(*args, **kwargs)

        msves, berrs, tderrs = [], [], []
        errs = []
        weights = []
        eps = []

        for ep_no in range(1, episodes_no + 1):
            state, done = env.reset(), False
            obs = state @ features
            ep_errs = []
            while not done:
                action = self.select_action(state)  # Actions cond. by state
                next_state, reward, done, _ = env.step(action)
                next_obs = None if done else (next_state @ features)
                ep_errs.extend(self.update(obs, reward, next_obs, done))
                obs, state = next_obs, next_state

            if ep_no % step == 0:
                eps.append(ep_no)
                errs.append(np.mean(ep_errs))
                values = self.predict(features)
                msves.append(msve(values))
                berrs.append(berr(values))
                tderrs.append(tderr(values))
                weights.append(np.copy(self.params))
                self.report()

        other = self.end_training()

        results = {"values": values,
                   "msve": np.array(msves),
                   "berr": np.array(berrs),
                   "tderr": np.array(tderrs),
                   "err": np.array(errs),
                   "weights": weights,
                   "episode": np.array(eps)
                   }
        results.update(other)
        return results

    @property
    def params(self) -> np.ndarray:
        if hasattr(self, 'weight'):
            return self.weight
        return None

    def select_action(self, obs: np.ndarray) -> int:
        return self.policy.sample(obs)

    def predict(self, _obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, _reward: float,
               _next_obs: np.ndarray, _done: bool) -> List[float]:
        raise NotImplementedError

    def before_training(self) -> None:
        pass

    def report(self) -> None:
        pass

    def end_training(self) -> dict:
        return {}


class GradientMonteCarlo(OnlinePolicyEvaluation):

    def __init__(self, *args, lr: dict, **kwargs):
        super(GradientMonteCarlo, self).__init__(*args, **kwargs)
        self.lr = get_schedule(**lr)

    def before_training(self):
        self.weight = np.zeros(self.features.shape[1])
        self.rewards = []
        self.observations = []
        self.lr = iter(self.lr)
        self.crt_lr = next(self.lr)
        self.lrs = []

    def update(self, obs: np.ndarray,
               reward: float,
               _next_obs: Optional[np.ndarray],
               done: bool) -> List[float]:
        self.observations.append(obs)
        self.rewards.append(reward)
        weight = self.weight
        errs = []
        if done:
            for idx, obs in enumerate(self.observations):
                ret = sum(self.rewards[idx:])
                err = (ret - weight @ obs)
                weight += self.crt_lr * err * obs
                errs.append(err * err)
            self.observations.clear()
            self.rewards.clear()
            self.crt_lr = next(self.lr)
        return errs

    def report(self):
        self.lrs.append(self.crt_lr)

    def predict(self, obs: np.ndarray):
        return obs @ self.weight

    def end_training(self):
        return {"lrs": self.lrs}


class GradientTDn(OnlinePolicyEvaluation):

    def __init__(self, *args,
                 lr: dict={"name": 1e-2},
                 n: int = 1,
                 semigradient: bool=True,
                 **kwargs):
        super(GradientTDn, self).__init__(*args, **kwargs)
        self.lr = get_schedule(**lr)
        self.n = n
        self.semigradient = semigradient

    def before_training(self):
        self.weight = np.zeros(self.features.shape[1])
        self.observations = []
        self.rewards = []
        self.gammas = self.gamma ** np.arange(self.n + 1)
        self.lrs = []
        self.lr = iter(self.lr)
        self.crt_lr = next(self.lr)

    def update(self, obs: np.ndarray,
               reward: float,
               next_obs: Optional[np.ndarray],
               done: bool) -> List[float]:
        self.observations.append(obs)
        self.rewards.append(reward)
        weight = self.weight
        gamma = self.gamma
        errs = []
        if len(self.observations) == self.n and not done:
            ret = (self.rewards + [weight @ next_obs]) @ self.gammas
            obs = self.observations.pop(0)
            self.rewards.pop(0)
            err = (ret - weight @ obs)
            if self.semigradient:
                weight += self.crt_lr * err * obs
            else:
                weight -= self.crt_lr * err * (gamma * next_obs - obs)
            errs.append(err * err)
        if done:
            ret = self.rewards @ self.gammas[:len(self.rewards)]
            while self.observations:
                obs = self.observations.pop(0)
                err = (ret - weight @ obs)
                weight += self.crt_lr * err * obs
                errs.append(err * err)
                reward = self.rewards.pop(0)
                ret = (ret - reward) / gamma

            self.observations.clear()
            self.rewards.clear()
            self.crt_lr = next(self.lr)
        return errs

    def predict(self, obs: np.ndarray):
        return obs @ self.weight

    def report(self):
        self.lrs.append(self.crt_lr)

    def end_training(self):
        return {"lrs": np.array(self.lrs)}


class LSTD(OnlinePolicyEvaluation):

    def __init__(self, *args,
                 eps: float=1e-2,
                 **kwargs):
        super(LSTD, self).__init__(*args, **kwargs)
        self.eps = eps

    def before_training(self):
        self.inv_a = np.eye(self.features.shape[1]) / self.eps
        self.b = np.zeros((self.features.shape[1]))

    def update(self, obs: np.ndarray,
               reward: float,
               next_obs: np.ndarray,
               done: bool):
        inv_a, b = self.inv_a, self.b
        if done:
            vec = obs @ inv_a
        else:
            vec = (obs - self.gamma * next_obs) @ inv_a
        inv_a -= np.outer(inv_a @ obs, vec) / (1 + vec @ obs)
        b += reward * obs

        return [.0]

    def predict(self, obs: np.ndarray):
        return obs @ (self.inv_a @ self.b)
