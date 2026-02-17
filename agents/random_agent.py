"""Random agent for baseline testing."""

import numpy as np
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects random actions."""

    def __init__(self, action_space, observation_space, config=None):
        """Initialize random agent."""
        super().__init__(action_space, observation_space, config)

    def select_action(self, observation, training=True):
        """
        Select a random action.

        Args:
            observation: Current observation (unused)
            training: Training mode (unused)

        Returns:
            Random action from action space
        """
        return self.action_space.sample()

    def update(self, *args, **kwargs):
        """Random agent doesn't learn."""
        pass

    def save(self, path):
        """Random agent has no model to save."""
        pass

    def load(self, path):
        """Random agent has no model to load."""
        pass
