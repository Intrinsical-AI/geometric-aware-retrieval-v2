"""Global configuration using pydantic for type safety."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# Sub-configurations for different components
# ---------------------------------------------------------------------------

class EncoderConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    mode: Literal["dual", "mono"] = "dual"
    normalize: bool = True

class TrainerConfig(BaseModel):
    # Common params
    epochs: int = 1
    batch_size: int = 16
    lr: float = 2e-5
    verbose: bool = False

    # Geometric-specific params
    k_graph: int = 10
    geodesic: bool = False
    lambda_ricci: float = 0.0
    lambda_forman: float = 0.0
    kappa_target: float = 0.0
    ricci_backend: str = "ricci_ollivier"

    @validator('k_graph')
    def validate_k_graph(cls, v):
        """Validate k-NN graph connectivity parameter."""
        if v < 5:
            raise ValueError("k_graph must be >= 5 for meaningful graph connectivity")
        if v > 100:
            warnings.warn(f"k_graph={v} may be computationally expensive for large datasets")
        return v
    
    @validator('lambda_ricci', 'lambda_forman')
    def validate_regularization_weights(cls, v):
        """Validate regularization weights are non-negative."""
        if v < 0:
            raise ValueError("Regularization weights must be non-negative")
        if v > 10.0:
            warnings.warn(f"Large regularization weight {v} may dominate training")
        return v
    
    @validator('ricci_backend')
    def validate_ricci_backend(cls, v):
        """Validate curvature computation backend."""
        valid_backends = {"ricci_ollivier", "forman", "auto"}
        if v not in valid_backends:
            raise ValueError(f"ricci_backend must be one of {valid_backends}, got '{v}'")
        return v
    
    @validator('lr')
    def validate_learning_rate(cls, v):
        """Validate learning rate is reasonable."""
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        if v > 0.1:
            warnings.warn(f"Learning rate {v} is unusually high, consider values < 0.01")
        return v

    @property
    def is_classic_mode(self) -> bool:
        """True if no geometric regularization is applied."""
        return self.lambda_ricci == 0 and self.lambda_forman == 0
    
    @property
    def is_geometric_mode(self) -> bool:
        """True if geometric regularization is enabled."""
        return not self.is_classic_mode

# ---------------------------------------------------------------------------
# Main Experiment Configuration
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """A complete, validated configuration for a geoIR experiment."""
    # Core components
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    dataset: str
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)

    # Other settings
    device: Optional[str] = None
    dry_run: bool = False
    output_dir: Optional[str] = None
    
    class Config:
        extra = "forbid" # Forbid extra fields to catch typos in YAML

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
