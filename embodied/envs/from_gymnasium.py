import functools

import elements
import embodied
import gymnasium as gym
import numpy as np
import cv2
from . import gymnasium_wrapper

class GymnasiumEnv(embodied.Env):
  _DEFAULT_PROP_KEY = 'proprio' # Default key for non-dict observations
  _DEFAULT_ACT_KEY = 'action'         # Default key for non-dict actions
  _DEFAULT_PARTIAL_KEY = 'pos' # Default key for partial observations for joint states
  _DEFAULT_CONTROL_COST = {'HalfCheetah-v5': 0.1, 'Hopper-v5': 1e-3}

  def __init__(self, env, repeat=1, size=(64, 64), proprio=True, image=True, seed=None, **kwargs):
    if isinstance(env, str):
      task_name = env
      partial = kwargs.pop('partial', '')
      ctrl_mult = kwargs['ctrl_mult']
      gymenv = gym.make(task_name, render_mode='rgb_array', 
                        ctrl_cost_weight=self._DEFAULT_CONTROL_COST[task_name]*ctrl_mult)

      # for modifying observation space in specific environments
      if partial == self._DEFAULT_PARTIAL_KEY:
        # For HalfCheetah and hopper, the observation is composed of positions and velocities.
        # This wrapper selects only the position components.
        obs_dim = gymenv.observation_space.shape[0]
        # For HalfCheetah-v4, obs is (17,), pos are first 8. obs_dim // 2 works.
        pos_indices = list(range(obs_dim // 2))
        gymenv = gymnasium_wrapper.SelectObsWrapper(
            gymenv, obs_indices_to_keep=pos_indices)
      self._gymenv = gymenv
    else:
      self._gymenv = env
      # User must ensure the passed env is configured for 'rgb_array' rendering
      # if they expect images. We check render_mode attribute below.

    # Ensure render_mode is 'rgb_array' if we are to render images.
    # The logic below will always attempt to render an image.
    if not hasattr(self._gymenv, 'render_mode') or self._gymenv.render_mode != 'rgb_array':
        # This is a strong assumption. If an env is passed, it might be pre-configured.
        # For now, we'll raise an error if rendering is expected and not possible.
        # Alternatively, one could try to wrap or re-initialize, but that's more intrusive.
        print(f"Warning: Gymnasium environment {env if isinstance(env, str) else env.__class__.__name__} "
              "may not be configured for 'rgb_array' rendering, which is required.")
        
    self._repeat = repeat    
    # Check if observation and action spaces are dictionaries
    self._obs_is_dict = hasattr(self._gymenv.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._gymenv.action_space, 'spaces')

    self._size = size
    self._include_proprio = proprio
    self._image_as_primary_obs = image # This flag determines the key for the rendered image

    self._done = True
    self._info = None
    
    if seed is not None:
      # Initial reset to apply the seed.
      # Ensure the environment supports seeding in reset, which is standard for Gym >= 0.26
      try:
        self._gymenv.reset(seed=seed)
      except TypeError as e:
        print(f"Warning: Environment reset does not support seed: {e}. Seeding may not be effective.")
        self._gymenv.reset()


  @property
  def env(self):
    return self._gymenv

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    spaces = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

    if self._include_proprio:
      if self._obs_is_dict:
        env_obs_spaces = self._flatten(self._gymenv.observation_space.spaces)
      else:
        env_obs_spaces = {self._DEFAULT_PROP_KEY: self._gymenv.observation_space}
      
      for key, value_space in env_obs_spaces.items():
        spaces[key] = self._convert(value_space)

    # An image is always added, key depends on self._image_as_primary_obs
    image_key = 'image' if self._image_as_primary_obs else 'log/image'
    spaces[image_key] = elements.Space(np.uint8, self._size + (3,))
    
    return spaces

  @functools.cached_property
  def act_space(self):
    if self._act_is_dict:
      spaces = self._flatten(self._gymenv.action_space.spaces)
    else:
      spaces = {self._DEFAULT_ACT_KEY: self._gymenv.action_space}
    
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = elements.Space(bool) # Add reset action
    return spaces

  def step(self, action):
    action_copy = action.copy()
    reset = action_copy.pop('reset')
    reward_final = 0.0

    if reset or self._done:
      # Pass seed=None explicitly if your reset might not expect it when not reseeding
      obs_gym, self._info = self._gymenv.reset() 
      self._done = False
      reward_final = 0.0
      is_first = True
      is_last = False
      is_terminal = False
    else:
      if self._act_is_dict:
        act_gym = self._unflatten(action_copy)
      else:
        act_gym = action_copy[self._DEFAULT_ACT_KEY]
      for _ in range(self._repeat):  
        obs_gym, reward, term, trunc, self._info = self._gymenv.step(act_gym)
        reward_final += reward
        if term or trunc:
          break
      self._done = term or trunc
      is_first = False
      is_last = self._done
      is_terminal = term # term is True if the episode ended due to task completion or failure

    # Initialize final observation dict
    obs_final = {
        'reward': np.float32(reward_final),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

    # Add proprioceptive observations from gym environment if flag is set
    if self._include_proprio:
      if self._obs_is_dict: # obs_gym is a dict
        processed_env_obs = self._flatten(obs_gym)
      else: # obs_gym is a single array
        processed_env_obs = {self._DEFAULT_PROP_KEY: obs_gym}
      
      for key, value_obs in processed_env_obs.items():
        obs_final[key] = np.asarray(value_obs)

    # Add rendered image (always, key depends on self._image_as_primary_obs)
    image_key = 'image' if self._image_as_primary_obs else 'log/image'
    obs_final[image_key] = self._render_image()
    
    return obs_final

  def _render_image(self):
    try:
      image = self._gymenv.render()
    except Exception as e:
      raise RuntimeError(
          f"Error during self._gymenv.render(): {e}. "
          "Ensure the environment is compatible with render_mode='rgb_array' "
          "and is correctly initialized."
      ) from e
      
    if image is None:
        raise RuntimeError(
            "self._gymenv.render() returned None. "
            "Ensure the environment is compatible with render_mode='rgb_array' "
            "and is correctly initialized."
        )
    # Resize image
    image = cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
    return image

  def render(self): # Public render method if needed by user
    return self._render_image()

  def close(self):
    try:
      self._gymenv.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key_str = str(key) # Ensure key is string for concatenation
      full_key = prefix + '/' + key_str if prefix else key_str
      if isinstance(value, gym.spaces.Dict): # For obs_space construction
        result.update(self._flatten(value.spaces, full_key))
      elif isinstance(value, dict): # For actual observation data
        result.update(self._flatten(value, full_key))
      else:
        result[full_key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if isinstance(space, gym.spaces.Discrete):
      return elements.Space(space.dtype if hasattr(space, 'dtype') else np.int64, (), 0, space.n)
    elif isinstance(space, gym.spaces.Box):
      low = np.nan_to_num(space.low, neginf=-np.inf, posinf=np.inf)
      high = np.nan_to_num(space.high, neginf=-np.inf, posinf=np.inf)
      return elements.Space(space.dtype, space.shape, low, high)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return elements.Space(space.dtype, space.shape, np.zeros_like(space.nvec, dtype=space.dtype), space.nvec - 1)
    elif isinstance(space, gym.spaces.MultiBinary):
        return elements.Space(space.dtype if hasattr(space, 'dtype') else np.int8, space.shape, 0, 1)
    elif isinstance(space, (gym.spaces.Tuple, gym.spaces.Sequence)):
        # Basic handling, might need more sophisticated conversion for complex tuples/sequences
        # For now, raise NotImplementedError or handle specific common cases if any
        raise NotImplementedError(f"Conversion for gym space type {type(space)} is not fully implemented.")
    else:
        if hasattr(space, 'dtype') and hasattr(space, 'shape'): # Fallback for simple spaces
            return elements.Space(space.dtype, space.shape)
        else:
            raise NotImplementedError(f"Unsupported gym space type: {type(space)}")