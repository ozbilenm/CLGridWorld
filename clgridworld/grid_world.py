from functools import lru_cache

import gym


class GridWorld(gym.Env):

    def __init__(self, observation_space, action_space, initial_state, reward_function, dynamics,
                 terminal_state_validator, visualizer):

        self.initial_state = initial_state
        self.observation_space = observation_space
        self.action_space = action_space
        self.dynamics = dynamics
        self.reward_function = reward_function
        self.terminal_state_validator = terminal_state_validator
        self.visualizer = visualizer

        # self.reward_range = reward_function.reward_range

        self.prev_state = self.initial_state
        self.curr_state = self.initial_state

    def step(self, action):
        self.prev_state = self.curr_state
        self.curr_state, reward, done, info = self._step(self.prev_state, action)
        return self.curr_state, reward, done, info

    @lru_cache(maxsize=None)
    def _step(self, prev_state, action):
        curr_state = self.dynamics.step(prev_state, action)
        reward = self.reward_function.calculate(prev_state, action, curr_state)
        done = self.terminal_state_validator.is_terminal_state(curr_state)
        info = {}
        return curr_state, reward, done, info

    def reset(self):
        self.prev_state = self.initial_state
        self.curr_state = self.initial_state
        return self.initial_state

    def render(self, mode='human'):
        self.visualizer.render(self.curr_state)

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

    def copy(self, seed=None):
        new_grid_world = GridWorld(self.observation_space, self.action_space, self.initial_state, self.reward_function,
                                   self.dynamics, self.terminal_state_validator, self.visualizer)

        new_grid_world.prev_state = self.prev_state
        new_grid_world.curr_state = self.curr_state

        new_grid_world.seed(seed)

        return new_grid_world

    def __str__(self):
        return "<GridWorld" + str({"initial_state": self.initial_state}) + ">"
