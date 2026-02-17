"""Create training rewards graph."""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Read training logs
pong_data = []
breakout_data = []

# Read Pong logs
try:
    with open('results/logs/Pong_random.jsonl', 'r') as f:
        for line in f:
            pong_data.append(json.loads(line))
except FileNotFoundError:
    pass

# Read Breakout logs
try:
    with open('results/logs/Breakout_random.jsonl', 'r') as f:
        for line in f:
            breakout_data.append(json.loads(line))
except FileNotFoundError:
    pass

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Plot Pong
if pong_data:
    episodes = [d['episode'] for d in pong_data]
    rewards = [d['episode_reward'] for d in pong_data]
    ax1.plot(episodes, rewards, marker='o', linewidth=2, markersize=6,
             color='#667eea', label='Episode Reward')
    ax1.axhline(y=sum(rewards)/len(rewards), color='#764ba2',
                linestyle='--', linewidth=2, label=f'Average: {sum(rewards)/len(rewards):.2f}')
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Pong Training Progress', fontsize=14, fontweight='bold', color='#667eea')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_facecolor('#f8f9fa')
else:
    ax1.text(0.5, 0.5, 'No Pong data', ha='center', va='center', fontsize=14)
    ax1.set_title('Pong Training Progress', fontsize=14, fontweight='bold')

# Plot Breakout
if breakout_data:
    episodes = [d['episode'] for d in breakout_data]
    rewards = [d['episode_reward'] for d in breakout_data]
    ax2.plot(episodes, rewards, marker='o', linewidth=2, markersize=6,
             color='#f59e0b', label='Episode Reward')
    ax2.axhline(y=sum(rewards)/len(rewards), color='#dc2626',
                linestyle='--', linewidth=2, label=f'Average: {sum(rewards)/len(rewards):.2f}')
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Breakout Training Progress', fontsize=14, fontweight='bold', color='#f59e0b')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_facecolor('#f8f9fa')
else:
    ax2.text(0.5, 0.5, 'No Breakout data', ha='center', va='center', fontsize=14)
    ax2.set_title('Breakout Training Progress', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/training_progress.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Graph saved to docs/training_progress.png")
