import numpy as np


class Policy:

    def __init__(self, nfeatures: int, nactions: int):
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.__actions = np.arange(nactions)
        self._qs = None

    def sample(self, obs: np.ndarray) -> int:
        return np.random.choice(self.__actions, p=self.probs(obs))

    def probs(self, obs: np.ndarray = None) -> np.ndarray:
        if obs is None:
            obs = np.eye(self.nfeatures)
        probs = np.exp(obs @ self._qs)
        if obs.ndim == 1:
            return probs / probs.sum()
        return probs / probs.sum(axis=1, keepdims=True)


class UniformPolicy(Policy):

    def __init__(self, nfeatures: int, nactions: int) -> None:
        super(UniformPolicy, self).__init__(nfeatures, nactions)
        self._qs = np.ones((nfeatures, nactions))


class RandomPolicy(Policy):

    def __init__(self, nfeatures: int, nactions: int) -> None:
        super(RandomPolicy, self).__init__(nfeatures, nactions)
        self._qs = np.ones((nfeatures, nactions))
