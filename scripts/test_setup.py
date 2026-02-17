"""Test script to verify Atari RL setup."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
from environments.atari_wrapper import make_atari_env
from agents.random_agent import RandomAgent


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import gymnasium
        import ale_py
        import torch
        import numpy
        import yaml

        print("✓ All required packages imported successfully")
        print(f"  - Gymnasium version: {gymnasium.__version__}")
        print(f"  - ALE-py version: {ale_py.__version__}")
        print(f"  - PyTorch version: {torch.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_environment(game_name="Pong", episodes=3):
    """Test creating and running an Atari environment."""
    print(f"\nTesting environment: {game_name}")

    try:
        # Create environment
        env = make_atari_env(game_name, frame_stack=4)
        print(f"✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")

        # Create random agent
        agent = RandomAgent(env.action_space, env.observation_space)

        # Run episodes
        total_rewards = []
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

            total_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

        env.close()

        print(f"✓ Environment test completed")
        print(f"  - Average reward: {np.mean(total_rewards):.2f}")
        print(f"  - Std reward: {np.std(total_rewards):.2f}")

        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_available_games():
    """List available Atari games."""
    print("\nChecking available Atari games...")

    common_games = [
        "Pong",
        "Breakout",
        "SpaceInvaders",
        "Seaquest",
        "BeamRider",
        "Enduro",
        "MsPacman",
        "Qbert",
    ]

    available = []
    for game in common_games:
        try:
            env = gym.make(f"ALE/{game}-v5")
            env.close()
            available.append(game)
        except:
            pass

    if available:
        print(f"✓ Found {len(available)} available games:")
        for game in available:
            print(f"  - {game}")
    else:
        print("✗ No games found. ROMs may not be installed.")

    return len(available) > 0


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test Atari RL setup")
    parser.add_argument("--game", default="Pong", help="Game to test")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
    args = parser.parse_args()

    print("=" * 60)
    print("Atari RL Environment Setup Test")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test available games
    results.append(("Available Games", test_available_games()))

    # Test environment
    results.append(("Environment", test_environment(args.game, args.episodes)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result[1] for result in results)
    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Your Atari RL environment is ready.")
        print("\nNext steps:")
        print("  1. Try different games: python scripts/test_setup.py --game Breakout")
        print("  2. Start training: python scripts/train.py")
        print("  3. Explore notebooks: jupyter notebook notebooks/exploration.ipynb")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
