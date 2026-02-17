# 🎮 Atari Reinforcement Learning Environment

A comprehensive setup for training RL agents on classic Atari 2600 games using Gymnasium and PyTorch.

📊 **[View Live Training Dashboard →](https://tbandi-northslope.github.io/atari/)**

## 🌟 Features

- ✅ **110+ Atari Games** - Full library of classic games
- ✅ **Modern RL Stack** - Gymnasium 1.2.3, PyTorch 2.10.0, ALE-py 0.11.2
- ✅ **Agent Implementations** - Random baseline and DQN template
- ✅ **Video Recording** - Watch your agents play in real-time
- ✅ **Comprehensive Logging** - Track training metrics and progress
- ✅ **Clean Architecture** - Modular design for easy extension

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tbandi-northslope/atari.git
cd atari

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Verify Setup

```bash
# Test the environment
KMP_DUPLICATE_LIB_OK=TRUE python scripts/test_setup.py
```

## 🎯 Usage

### Train an Agent

```bash
# Train with random agent
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --game Breakout --agent random --episodes 100

# Train with DQN
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --game Pong --agent dqn --episodes 1000
```

### Record Gameplay Videos

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/train.py --game SpaceInvaders --agent random --episodes 10 --render
```

Videos saved to `results/videos/`

## 📊 Project Structure

```
atari/
├── agents/              # RL agent implementations
├── environments/        # Environment wrappers
├── utils/              # Utility modules
├── scripts/            # Training/evaluation scripts
├── results/            # Training outputs
├── docs/               # GitHub Pages site
└── tests/              # Unit tests
```

## 🛠️ Tech Stack

- **Gymnasium** 1.2.3 - RL environment framework
- **ALE-py** 0.11.2 - Atari Learning Environment
- **PyTorch** 2.10.0 - Deep learning framework
- **Python** 3.14 - Programming language

## 📈 Training Results

View live dashboard at: **[tbandi-northslope.github.io/atari](https://tbandi-northslope.github.io/atari/)**

Recent runs:
- **Pong**: 10 episodes, avg reward -20.20
- **Breakout**: 18 episodes, avg reward 1.22, best 4.00

## 🎮 Available Games

Pong • Breakout • SpaceInvaders • Seaquest • BeamRider • Enduro • MsPacman • Qbert • Asteroids • and 100+ more!

## 📝 Configuration

Edit `config.yaml` to customize training parameters.

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/)
- [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [DQN Paper](https://www.nature.com/articles/nature14236)

---

Built with ❤️ for the RL community
