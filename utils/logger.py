"""Logging utilities."""

import os
import json
from datetime import datetime
from pathlib import Path


class Logger:
    """Logger for training metrics."""

    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"

        self.metrics = []

    def log(self, metrics, step):
        """
        Log metrics at a given step.

        Args:
            metrics: Dictionary of metrics
            step: Current training step
        """
        log_entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}

        self.metrics.append(log_entry)

        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def print_summary(self, metrics, step):
        """Print formatted summary of metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Step {step}")
        print("-" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<30} {value:.4f}")
            else:
                print(f"{key:.<30} {value}")
        print("-" * 50)

    def load_metrics(self):
        """Load metrics from log file."""
        if not self.log_file.exists():
            return []

        metrics = []
        with open(self.log_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line))
        return metrics
