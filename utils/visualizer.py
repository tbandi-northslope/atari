"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_curve(metrics, save_path=None):
    """
    Plot training curves.

    Args:
        metrics: List of metric dictionaries
        save_path: Path to save plot
    """
    steps = [m["step"] for m in metrics]
    rewards = [m.get("episode_reward", 0) for m in metrics]
    losses = [m.get("loss", 0) for m in metrics if "loss" in m]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Reward curve
    axes[0].plot(steps, rewards, alpha=0.6, label="Episode Reward")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Reward")
    axes[0].legend()
    axes[0].grid(True)

    # Loss curve
    if losses:
        loss_steps = [m["step"] for m in metrics if "loss" in m]
        axes[1].plot(loss_steps, losses, alpha=0.6, label="Loss", color="orange")
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Loss")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_frame(observation, title="Observation"):
    """
    Display a single frame.

    Args:
        observation: Frame to display
        title: Title for plot
    """
    if observation.ndim == 3 and observation.shape[0] <= 4:
        # If stacked frames, show last frame
        frame = observation[-1]
    else:
        frame = observation

    plt.figure(figsize=(6, 6))
    plt.imshow(frame, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()
