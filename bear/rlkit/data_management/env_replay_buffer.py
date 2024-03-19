from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self._tdrp_index = 0

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
    def horizon_bath(self, horizon_length):
        max_index = min(self._size, self._tdrp_index+horizon_length)
        if max_index-self._tdrp_index < horizon_length/3:
            self._tdrp_index=0
            return None
        batch = dict(
            observations=self._observations[self._tdrp_index:max_index],
            actions=self._actions[self._tdrp_index:max_index],
            rewards=self._rewards[self._tdrp_index:max_index],
            terminals=self._terminals[self._tdrp_index:max_index],
            next_observations=self._next_obs[self._tdrp_index:max_index],
        )
        if max_index== self._size:
            self._tdrp_index = 0
        else:
            self._tdrp_index=max_index
        return batch
