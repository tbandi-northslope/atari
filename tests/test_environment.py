"""Test environment creation and wrappers."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.atari_wrapper import make_atari_env
from agents.random_agent import RandomAgent


def test_environment_creation():
    """Test that environment can be created."""
    env = make_atari_env("Pong", frame_stack=4)
    assert env is not None
    env.close()


def test_environment_reset():
    """Test environment reset."""
    env = make_atari_env("Pong", frame_stack=4)
    obs, info = env.reset()
    assert obs is not None
    assert obs.shape == (4, 84, 84)
    env.close()


def test_environment_step():
    """Test environment step."""
    env = make_atari_env("Pong", frame_stack=4)
    obs, info = env.reset()

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()


def test_random_agent():
    """Test random agent can interact with environment."""
    env = make_atari_env("Pong", frame_stack=4)
    agent = RandomAgent(env.action_space, env.observation_space)

    obs, info = env.reset()

    for _ in range(100):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
