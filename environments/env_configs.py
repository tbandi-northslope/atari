"""Game-specific configurations."""

GAME_CONFIGS = {
    "Pong": {
        "env_id": "ALE/Pong-v5",
        "description": "Classic Pong game",
        "max_episode_steps": 10000,
        "action_space": 6,
    },
    "Breakout": {
        "env_id": "ALE/Breakout-v5",
        "description": "Break bricks with a ball",
        "max_episode_steps": 10000,
        "action_space": 4,
    },
    "SpaceInvaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "description": "Shoot alien invaders",
        "max_episode_steps": 10000,
        "action_space": 6,
    },
    "Seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "description": "Submarine rescue mission",
        "max_episode_steps": 10000,
        "action_space": 18,
    },
    "BeamRider": {
        "env_id": "ALE/BeamRider-v5",
        "description": "Ride the beam and shoot enemies",
        "max_episode_steps": 10000,
        "action_space": 9,
    },
    "Enduro": {
        "env_id": "ALE/Enduro-v5",
        "description": "Racing game",
        "max_episode_steps": 10000,
        "action_space": 9,
    },
    "MsPacman": {
        "env_id": "ALE/MsPacman-v5",
        "description": "Navigate maze and eat pellets",
        "max_episode_steps": 10000,
        "action_space": 9,
    },
    "Qbert": {
        "env_id": "ALE/Qbert-v5",
        "description": "Change pyramid colors",
        "max_episode_steps": 10000,
        "action_space": 6,
    },
}
