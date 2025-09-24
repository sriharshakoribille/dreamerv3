import functools
import embodied
import gymnasium as gym
import minigrid
import numpy as np
import cv2

class FromMiniGrid(embodied.Env):
  def __init__(self, name, repeat=1, size=(64, 64), seed=None):
    self._env = gym.make(name, render_mode='rgb_array')
    # Use MiniGrid's wrapper to get image observations directly.
    self._env = minigrid.wrappers.RGBImgPartialObsWrapper(self._env)
    self._repeat = repeat
    self._size = size
    self._done = True
    self._info = None

    if seed is not None:
      try:
        self._env.reset(seed=seed)
      except TypeError as e:
        print(f"Warning: Environment reset does not support seed: {e}. Seeding may not be effective.")
        self._env.reset()

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'direction': embodied.Space(np.int64, (), 0, 4),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    return {
        'action': embodied.Space(np.int64, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      obs, self._info = self._env.reset()
      self._done = False
      return self._obs(obs, 0.0, is_first=True)

    reward = 0.0
    for _ in range(self._repeat):
      obs, r, term, trunc, self._info = self._env.step(action['action'])
      reward += r
      if term or trunc:
        self._done = True
        break
    
    return self._obs(obs, reward, is_last=self._done, is_terminal=self._info.get('is_terminal', self._done))

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    # Resize image from observation dictionary
    image = cv2.resize(obs['image'], self._size, interpolation=cv2.INTER_AREA)
    return dict(
        image=image,
        direction=obs['direction'],
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

  def render(self):
    return self._env.render()

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass