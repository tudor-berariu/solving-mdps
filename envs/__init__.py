from gym.envs.registration import register, registry
from .mdp import MDP
from .three_state import ThreeStateEnv


if 'ThreeState-v0' not in [env_spec.id for env_spec in registry.all()]:
    register(id='ThreeState-v0', entry_point='envs.three_state:ThreeStateEnv')

__all__ = ["MDP", "ThreeStateEnv"]
