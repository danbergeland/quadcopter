import numpy as np
from base_agent import BaseAgent
import tensorflow as tf

class DDPG(BaseAgent):
    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low

        self.last_action = None
        self.last_state = None

    def step(self, state, reward, done):

        action = np.zeros_like(self.task.action_space.shape)
        self.last_action = action
        self.last_state = state
        if done:
            self.reset_episode_vars()
        return action

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
