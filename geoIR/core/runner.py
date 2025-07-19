#!/usr/bin/env python3
"""Generic experiment runner with logging and result management."""

import abc
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ExperimentRunner(abc.ABC):
    """Abstract base class to standardise experiment execution."""

    def __init__(self, experiment_name: str, base_output_dir: str = "experiments/results"):
        """Initialise runner.

        Parameters
        ----------
        experiment_name : str
            Descriptive name for the experiment.
        base_output_dir : str, optional
            Root directory where results will be stored.
        """
        self.experiment_name = experiment_name
        self.base_output_dir = Path(base_output_dir)
        self.run_dir: Path = None
        self.logger: logging.Logger = None
        self.config: Dict[str, Any] = {}

    def _setup_run_environment(self):
        """Create a unique run directory and configure logging."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_output_dir / self.experiment_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging to both file and console
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers on re-entry
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.run_dir / "run.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger.addHandler(stream_handler)

        self.logger.info(f"Run results in: {self.run_dir}")

    def _save_config(self, config: Any):
        """Save experiment configuration as JSON."""
        if not self.run_dir:
            raise RuntimeError(
                "Run environment has not been initialised. Call _setup_run_environment() first."
            )

        config_path = self.run_dir / "config.json"

        config_dict = {}
        if is_dataclass(config):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            self.logger.warning("Configuration is not a dataclass or dict; saving as string.")
            config_dict = {"config": str(config)}

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}")

    @abc.abstractmethod
    def run(self):
        """Abstract method with the main experiment logic."""
        pass

    def start(self, config: Any):
        """Entry point to start the experiment."""
        # Store configuration internally
        if is_dataclass(config):
            self.config = asdict(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = {"config": str(config)}

        # Set up environment and save configuration
        self._setup_run_environment()
        self._save_config(self.config)

        try:
            # Run the main experiment logic
            self.run()
        except Exception as e:
            self.logger.error("Experiment failed with an exception.", exc_info=True)
            raise e

        self.logger.info("Experiment finished successfully.")

    def save_results(self, results: list[dict], filename: str) -> Path:
        """Save results to a JSON file."""
        if not self.run_dir:
            raise RuntimeError("Run environment has not been initialised.")

        path = self.run_dir / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {path}")
        return path

    def save_dataframe(self, df, filename: str) -> Path:
        """Save a pandas DataFrame to CSV."""
        if not self.run_dir:
            raise RuntimeError("Run environment has not been initialised.")

        path = self.run_dir / filename
        df.to_csv(path, index=False)
        self.logger.info(f"DataFrame saved to {path}")
        return path
