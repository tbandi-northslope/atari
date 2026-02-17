"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Base class for all RL agents."""

    def __init__(self, action_space, observation_space, config=None):
        """
        Initialize agent.

        Args:
            action_space: Gymnasium action space
            observation_space: Gymnasium observation space
            config: Configuration dictionary
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config or {}

    @abstractmethod
    def select_action(self, observation, training=True):
        """
        Select an action given an observation.

        Args:
            observation: Current observation from environment
            training: Whether agent is in training mode

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent's policy based on experience."""
        pass

    @abstractmethod
    def save(self, path):
        """Save agent's model."""
        pass

    @abstractmethod
    def load(self, path):
        """Load agent's model."""
        pass

    def reset(self):
        """Reset agent state if needed."""
        pass
