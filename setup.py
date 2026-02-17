"""Setup script for atari-rl package."""

from setuptools import setup, find_packages

setup(
    name="atari-rl",
    version="0.1.0",
    description="Atari Reinforcement Learning Environment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium[atari]>=1.0.0",
        "ale-py>=0.10.1",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
