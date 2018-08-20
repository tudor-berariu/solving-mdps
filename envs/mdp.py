from typing import Tuple
import numpy as np
from gym import Env


class MDP(Env):

    @property
    def n_states(self) -> int:
        """The number of all (terminal and non-terminal) states: |S+|"""
        return self._n_states

    @property
    def n_nstates(self) -> int:
        """The number of non-terminal states.: |S|"""
        return self._n_nstates

    @property
    def mdp(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.init_state_dist, self.dynamics, self.rewards

    @property
    def init_state_dist(self) -> np.ndarray:
        """The initial state distribution (a |S|-sized vector)"""
        assert self._init_state_dist.shape == (self.n_nstates,)
        return np.copy(self._init_state_dist)

    @property
    def dynamics(self) -> np.ndarray:
        """The state transition probabilities (a |S|x|A|x|S+| array)"""
        expected_shape = (self.n_nstates, self.action_space.n, self.n_states)
        assert self._dynamics.shape == expected_shape
        return np.copy(self._dynamics)

    @property
    def rewards(self) -> np.ndarray:
        """The reward associated with each transition (a |S|x|S+| array)"""
        assert self._rewards.shape == (self.n_nstates, self.n_states)
        return np.copy(self._rewards)
