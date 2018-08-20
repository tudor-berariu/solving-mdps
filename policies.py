from typing import Union
import numpy as np
from envs import MDP


class Policy:

    def sample(self, state: Union[int, np.ndarray]) -> int:
        if isinstance(state, np.ndarray):
            state = state.argmax()
        return np.random.choice(np.arange(self._probs.shape[1]),
                                p=self._probs[state])

    @property
    def probs(self) -> np.ndarray:
        return np.copy(self._probs)


class UniformPolicy(Policy):

    def __init__(self, env: MDP) -> None:
        self._probs = np.full((env.n_nstates, env.action_space.n),
                              1. / float(env.action_space.n))
