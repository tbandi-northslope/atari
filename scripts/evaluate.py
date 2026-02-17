"""Evaluation script for trained agents."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import gymnasium as gym
import ale_py  # Register ALE environments

from environments.atari_wrapper import make_atari_env
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument("--game", type=str, required=True, help="Game name")
    parser.add_argument("--agent", type=str, default="random", choices=["random", "dqn"])
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    return parser.parse_args()


def evaluate(args):
    """Evaluate agent."""
    # Create environment
    env = make_atari_env(
        args.game,
        frame_stack=4,
        record_video=args.render,
        video_folder="results/videos/eval" if args.render else None,
    )

    # Create agent
    if args.agent == "random":
        agent = RandomAgent(env.action_space, env.observation_space)
    elif args.agent == "dqn":
        agent = DQNAgent(env.action_space, env.observation_space, {})
        if args.model:
            agent.load(args.model)
            print(f"Loaded model from {args.model}")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    print(f"\nEvaluating {args.agent} on {args.game}")
    print(f"Episodes: {args.episodes}")
    print()

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(args.episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    env.close()

    # Print statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    print("=" * 60)


def main():
    """Entry point."""
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
