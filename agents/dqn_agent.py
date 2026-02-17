"""DQN Agent implementation template."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from agents.base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, input_shape, n_actions):
        """
        Initialize DQN network.

        Args:
            input_shape: Shape of input observations (C, H, W)
            n_actions: Number of possible actions
        """
        super(DQNNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        """Calculate output size of convolutional layers."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Forward pass."""
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent."""

    def __init__(self, action_space, observation_space, config=None):
        """Initialize DQN agent."""
        super().__init__(action_space, observation_space, config)

        self.n_actions = action_space.n
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Networks
        input_shape = observation_space.shape
        self.policy_net = DQNNetwork(input_shape, self.n_actions).to(self.device)
        self.target_net = DQNNetwork(input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training parameters
        self.lr = config.get("learning_rate", 0.00025)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 1000000)
        self.batch_size = config.get("batch_size", 32)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = deque(maxlen=config.get("replay_buffer_size", 100000))

        self.steps = 0

    def select_action(self, observation, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return self.action_space.sample()

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_tensor)
            return q_values.argmax().item()

    def update(self, transition):
        """
        Update agent with a transition.

        Args:
            transition: Tuple of (state, action, reward, next_state, done)
        """
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(
            self.epsilon_end, self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay
        )

        self.steps += 1

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """Save agent model."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        """Load agent model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint["steps"]
        self.epsilon = checkpoint["epsilon"]
