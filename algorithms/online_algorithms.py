from typing import Dict, Optional
import numpy as np

from envs import MDP
from policies import Policy
from algorithms.closed_form import cf_state_dist
from algorithms.error_functions import mean_squared_value_error
from algorithms.error_functions import bellman_error
from algorithms.error_functions import td_error


class OnlinePolicyEvaluation:

    def __init__(self,
                 env: MDP,
                 policy: Policy,
                 features: np.ndarray,
                 gamma: float = 1.0,
                 episodes_no: int = 100) -> None:
        self.env = env
        self.policy = policy
        self.features = features
        self.gamma = gamma
        self.episodes_no = episodes_no

    def train(self) -> Dict[str, object]:
        self.before_training()
        env = self.env
        features = self.features
        policy = self.policy
        gamma = self.gamma
        state_dist = cf_state_dist(env, policy, gamma)
        args = [env, policy]
        kwargs = {"state_dist": state_dist, "gamma": gamma}
        msve = mean_squared_value_error(*args, **kwargs)
        berr = bellman_error(*args, **kwargs)
        tderr = td_error(*args, **kwargs)

        msves, berrs, tderrs = [], [], []
        errs = []
        weights = []

        for episode in range(self.episodes_no):
            state, done = env.reset(), False
            obs = state @ features
            ep_errs = []
            while not done:
                action = self.select_action(obs)
                next_state, reward, done, _ = env.step(action)
                next_obs = None if done else (next_state @ features)
                ep_errs.extend(self.update(obs, reward, next_obs, done))
                obs = next_obs

            errs.append(np.mean(ep_errs))
            values = self.predict(features)
            msves.append(msve(values))
            berrs.append(berr(values))
            tderrs.append(tderr(values))
            weights.append(np.copy(self.params))

        return {"values": values,
                "msve": np.array(msves),
                "berr": np.array(berrs),
                "tderr": np.array(tderrs),
                "err": np.array(errs),
                "weights": weights
                }

    @property
    def params(self) -> np.ndarray:
        return None

    def select_action(self, obs: np.ndarray) -> int:
        return self.policy.sample(obs)

    def predict(self, _obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray,  _reward: float,
               _next_obs: np.ndarray, _done: bool) -> None:
        raise NotImplementedError

    def before_training(self) -> None:
        pass


class GradientMonteCarlo(OnlinePolicyEvaluation):

    def __init__(self, *args, lr: float = 1e-2, **kwargs):
        super(GradientMonteCarlo, self).__init__(*args, **kwargs)
        self.lr = lr

    def before_training(self):
        self.weight = np.zeros(self.features.shape[1])
        self.rewards = []
        self.observations = []

    def update(self, obs: np.ndarray,
               reward: float,
               _next_obs: Optional[np.ndarray],
               done: bool) -> None:
        self.observations.append(obs)
        self.rewards.append(reward)
        weight = self.weight
        errs = []
        if done:
            for idx, obs in enumerate(self.observations):
                ret = sum(self.rewards[idx:])
                err = (ret - weight @ obs)
                weight += self.lr * err * obs
                errs.append(err * err)
            self.observations.clear()
            self.rewards.clear()
        return errs

    def predict(self, obs: np.ndarray):
        return obs @ self.weight
