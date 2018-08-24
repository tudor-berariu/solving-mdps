from typing import List, Tuple
import numpy as np
from gym import spaces
from gym.utils import seeding
from .mdp import MDP


class ThreeStateEnv(MDP):
    """The environment has three non-terminal states.
    """

    metadata = {'render.modes': []}

    def __init__(self) -> None:
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.zeros(3), high=np.ones(3),
                                            dtype=np.float32)
        self._n_states, self._n_nstates = 5, 3
        LEFT, RIGHT = 0, 1
        A, B, C, BAD_END, GOOD_END = range(self._n_states)

        self._init_state_dist = np.array([1., 0., 0.])
        self._dynamics = np.zeros((3, 2, 5))
        for i in range(3):
            self._dynamics[i, LEFT, i] = self._dynamics[i, RIGHT, i] = .1
        self._dynamics[A, LEFT, B] = .9
        self._dynamics[A, RIGHT, BAD_END] = .9
        self._dynamics[B, LEFT, BAD_END] = .9
        self._dynamics[B, RIGHT, A] = .4
        self._dynamics[B, RIGHT, C] = .5
        self._dynamics[C, LEFT, GOOD_END] = .3
        self._dynamics[C, LEFT, B] = .6
        self._dynamics[C, RIGHT, BAD_END] = .9

        self._rewards = np.zeros((3, 5))
        self._rewards[C, GOOD_END] = 10.
        self._rewards[[A, B, C], BAD_END] = -1.

        self._gamma = 1

        self._crt_state = None  # type: int
        self.seed()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self._crt_state < 3
        next_state_dist = self._dynamics[self._crt_state, action]
        next_state = self.np_random.choice(list(range(5)), p=next_state_dist)
        reward = self._rewards[self._crt_state, next_state]
        done = (next_state > 2)
        self._crt_state = next_state
        obs = None if done else np.eye(3)[next_state]
        return obs, reward, done, {}

    def reset(self) -> np.ndarray:
        self._crt_state = 0
        return np.array([1, 0, 0])
