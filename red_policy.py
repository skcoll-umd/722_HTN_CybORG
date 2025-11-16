#Class for custom red policies
from mini_CAGE.test_agent import Meander_minimal, B_line_minimal
import numpy as np

class NoOpRed:
    """Red policy that always returns NOOP, matching MiniCAGE batch shape."""
    def __init__(self, *args, **kwargs):
        pass

    def get_action(self, observation, *args, **kwargs):
        # observation is usually shape (num_envs, obs_dim)
        if observation.ndim == 1:
            # single env row -> (1, obs_dim)
            observation = observation.reshape(1, -1)
        num_envs = observation.shape[0]
        return np.zeros((num_envs, 1), dtype=int)

#Create a policy to return meander_minimal agent
def make_meander_minimal(**kwargs):
    return Meander_minimal(**kwargs)

#Create a policy to return B_line_minimal agent
def make_b_line_minimal(**kwargs):
    return B_line_minimal(**kwargs)