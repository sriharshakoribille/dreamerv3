import gymnasium as gym
import numpy as np

class SelectObsWrapper(gym.Wrapper):
    """
    A wrapper for Gymnasium environments to select a subset of observation indices.
    This is useful for environments like HalfCheetah where you might want to
    separate position and velocity components of the state.
    This wrapper assumes the observation space is a single gym.spaces.Box.
    """
    def __init__(self, env, obs_indices_to_keep):
        """
        Args:
            env: The Gymnasium environment to wrap.
            obs_indices_to_keep: A list or array of indices to keep from the original observation.
        """
        super().__init__(env)
        
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("SelectObsWrapper only supports gym.spaces.Box observation spaces.")
            
        self.obs_indices_to_keep = obs_indices_to_keep
        
        original_space = self.env.observation_space
        
        self.observation_space = gym.spaces.Box(
            low=original_space.low[self.obs_indices_to_keep],
            high=original_space.high[self.obs_indices_to_keep],
            shape=(len(self.obs_indices_to_keep),),
            dtype=original_space.dtype
        )

    def _modify_obs(self, obs):
        return obs[self.obs_indices_to_keep]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._modify_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._modify_obs(obs), reward, terminated, truncated, info
