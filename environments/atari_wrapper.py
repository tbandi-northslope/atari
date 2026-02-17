"""Atari environment wrappers."""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
)


def make_atari_env(
    game_name,
    frame_skip=4,
    frame_stack=4,
    screen_size=84,
    grayscale=True,
    clip_rewards=True,
    record_video=False,
    video_folder=None,
):
    """
    Create and wrap Atari environment with standard preprocessing.

    Args:
        game_name: Name of the Atari game (e.g., 'Pong', 'Breakout')
        frame_skip: Number of frames to skip
        frame_stack: Number of frames to stack
        screen_size: Size to resize screen to
        grayscale: Whether to convert to grayscale
        clip_rewards: Whether to clip rewards to {-1, 0, 1}
        record_video: Whether to record videos
        video_folder: Folder to save videos to

    Returns:
        Wrapped Gymnasium environment
    """
    # Create base environment with frameskip disabled (AtariPreprocessing will handle it)
    if not game_name.endswith("-v4") and not game_name.endswith("-v5"):
        env_id = f"ALE/{game_name}-v5"
    else:
        env_id = game_name

    env = gym.make(env_id, render_mode="rgb_array" if record_video else None, frameskip=1)

    # Apply standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=grayscale,
        grayscale_newaxis=False,
        scale_obs=True,
    )

    # Clip rewards
    if clip_rewards:
        env = ClipRewardWrapper(env)

    # Stack frames
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)

    # Record video if requested
    if record_video and video_folder:
        env = RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda episode_id: True,
            name_prefix="gameplay",
        )

    return env


class ClipRewardWrapper(gym.RewardWrapper):
    """Wrapper to clip rewards to {-1, 0, 1}."""

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Clip reward to {-1, 0, 1}."""
        return np.sign(reward)


class EpisodeInfoWrapper(gym.Wrapper):
    """Wrapper to track episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }

        return obs, reward, terminated, truncated, info
