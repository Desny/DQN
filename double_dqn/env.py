import dm_env
import gym
import numpy as np
import random


class WrappedEnv(dm_env.Environment):
    def __init__(self, game, max_noop_steps, min_noop_steps):
        self._gym_env = gym.make(game).unwrapped
        if max_noop_steps < min_noop_steps:
            raise ValueError('max_noop_steps must be greater or equal min_noop_steps')
        self._max_noop_steps = max_noop_steps
        self._min_noop_steps = min_noop_steps

    def reset(self) -> dm_env.TimeStep:
        observation = self._gym_env.reset()
        lives = np.int32(self._gym_env.ale.lives())
        timestep = dm_env.restart((observation, lives))
        return self._apply_random_noops(timestep)

    def step(self, action) -> dm_env.TimeStep:
        observation, reward, done, info = self._gym_env.step(action)
        if done:
            step_type = dm_env.StepType.LAST
            discount = 0.
        else:
            step_type = dm_env.StepType.MID
            discount = 1.
        lives = np.int32(self._gym_env.ale.lives())
        timestep = dm_env.TimeStep(
            step_type=step_type,
            observation=(observation, lives),
            reward=reward,
            discount=discount
        )
        return timestep

    def close(self):
        self._gym_env.close()

    def render(self):
        self._gym_env.render()

    @property
    def observation_spec(self):
        return self._gym_env.observation_space

    @property
    def action_spec(self):
        return self._gym_env.action_space

    @property
    def lives(self):
        return self._gym_env.ale.lives()

    def _apply_random_noops(self, initial_timestep) -> dm_env.TimeStep:
        assert initial_timestep.first()
        num_steps = random.randint(self._min_noop_steps, self._max_noop_steps + 1)
        timestep = initial_timestep
        for _ in range(num_steps):
            noop_action = self._gym_env.action_space.sample()
            timestep = self.step(noop_action)
            if timestep.last():
                raise RuntimeError('Episode ended while applying %s noop actions.' % num_steps)
        return dm_env.restart(timestep.observation)


