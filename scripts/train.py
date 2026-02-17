"""Training script for Atari RL agents."""

import argparse
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
import ale_py  # Register ALE environments

from environments.atari_wrapper import make_atari_env
from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent on Atari")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--game", type=str, default="Pong", help="Game name")
    parser.add_argument("--agent", type=str, default="random", choices=["random", "dqn"])
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    return {}


def train(args, config):
    """Main training loop."""
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create directories
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "models").mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    (results_dir / "videos").mkdir(exist_ok=True)

    # Create environment
    env = make_atari_env(
        args.game,
        frame_stack=config["environment"]["frame_stack"],
        record_video=args.render,
        video_folder=str(results_dir / "videos") if args.render else None,
    )

    # Create agent
    if args.agent == "random":
        agent = RandomAgent(env.action_space, env.observation_space)
    elif args.agent == "dqn":
        agent = DQNAgent(env.action_space, env.observation_space, config["agent"])
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Create logger
    logger = Logger(results_dir / "logs", f"{args.game}_{args.agent}")

    print(f"\nTraining {args.agent} on {args.game}")
    print(f"Episodes: {args.episodes}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Training loop
    episode_rewards = []

    for episode in tqdm(range(args.episodes), desc="Training"):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Select action
            action = agent.select_action(obs, training=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update agent
            if hasattr(agent, "update"):
                agent.update((obs, action, reward, next_obs, terminated or truncated))

            obs = next_obs
            episode_reward += reward
            steps += 1

        episode_rewards.append(episode_reward)

        # Log metrics
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            metrics = {
                "episode": episode,
                "episode_reward": episode_reward,
                "avg_reward_10": avg_reward,
                "steps": steps,
            }
            if hasattr(agent, "epsilon"):
                metrics["epsilon"] = agent.epsilon

            logger.log(metrics, episode)
            logger.print_summary(metrics, episode)

        # Save model
        if episode % 100 == 0 and hasattr(agent, "save"):
            model_path = results_dir / "models" / f"{args.game}_{args.agent}_ep{episode}.pth"
            agent.save(model_path)

    env.close()

    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Final 100 avg: {np.mean(episode_rewards[-100:]):.2f}")


def main():
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)

    train(args, config)


if __name__ == "__main__":
    main()
