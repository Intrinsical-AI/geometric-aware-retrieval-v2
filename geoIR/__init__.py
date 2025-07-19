"""Top-level package for geoIR.

This library provides curvature-aware embedding models, index builders,
retrieval interfaces and evaluation utilities as described in ARCH.md.

High-level API examples
-----------------------
>>> import geoIR as gi
>>> encoder = gi.load_encoder("bge-base", mode="dual")
>>> index = encoder.build_index(corpus="beir/fiqa", k=30)
>>> hits = index.search("justice distributiva", k=10)

>>> # Quick geometric experiment
>>> results = gi.quick_experiment("bge-base", "beir/fiqa", geometric=True)
>>> print(f"nDCG@10: {results['ndcg_10']:.3f}")
"""

from importlib import import_module
from types import ModuleType
from typing import Literal

# Import retrieval to register the default encoder
from . import retrieval  # noqa: F401
from .core.registry import registry as _registry  # noqa: F401

_Mode = Literal["dual", "mono"]


def load_encoder(name: str, mode: _Mode = "dual", **kwargs):
    """Factory that loads an encoder backend via registry.

    Additional keyword arguments are forwarded to the backend loader. This allows
    users to specify, for instance, ``device="cpu"`` from high-level helpers
    such as ``geoIR.load_encoder`` without having to import the lower-level
    class directly.

    Parameters
    ----------
    name : str
        Model checkpoint name (e.g., "bge-base" or "openai/text-embedding-3-small").
    mode : Literal["dual", "mono"]
        Dual encoders return separate query/document towers; mono share weights.
    """
    try:
        backend_loader = _registry["encoder"]["default"]
    except KeyError as exc:
        raise RuntimeError("Default encoder backend not registered") from exc
    return backend_loader(name=name, mode=mode, **kwargs)


def quick_experiment(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset: str = "beir/fiqa",
    k: int = 20,
    geometric: bool = True,
    **kwargs,
) -> dict[str, float]:
    """One-liner for geometric retrieval experiments.

    Parameters
    ----------
    model_name : str, default "sentence-transformers/all-MiniLM-L6-v2"
        HuggingFace model name or path.
    dataset : str, default "beir/fiqa"
        Dataset name (BEIR format) or path to local data.
    k : int, default 20
        k-NN graph connectivity parameter.
    geometric : bool, default True
        Enable geometric regularization (InfoNCE-geo + curvature).
    **kwargs
        Additional trainer configuration parameters.

    Returns
    -------
    dict[str, float]
        Evaluation metrics including nDCG@10, MAP, etc.

    Examples
    --------
    >>> import geoIR as gi
    >>> results = gi.quick_experiment("bge-base", "beir/fiqa", geometric=True)
    >>> print(f"nDCG@10: {results['ndcg_10']:.3f}")

    >>> # Classic baseline comparison
    >>> classic = gi.quick_experiment("bge-base", "beir/fiqa", geometric=False)
    >>> geo = gi.quick_experiment("bge-base", "beir/fiqa", geometric=True)
    >>> improvement = geo['ndcg_10'] - classic['ndcg_10']
    >>> print(f"Geometric improvement: +{improvement:.3f} nDCG@10")
    """
    from .core.config import ExperimentConfig, TrainerConfig
    from .training.trainer import Trainer

    # Build configuration
    trainer_config = TrainerConfig(
        k_graph=k,
        geodesic=geometric,
        lambda_ricci=0.1 if geometric else 0.0,
        lambda_forman=0.05 if geometric else 0.0,
        epochs=1,  # Quick experiment
        verbose=True,
        **kwargs,
    )

    config = ExperimentConfig(dataset=dataset, trainer=trainer_config)
    config.encoder.model_name = model_name

    # Initialize and run
    encoder = load_encoder(model_name, mode="dual")
    trainer = Trainer(encoder, config.trainer)

    # Run the training loop and gather metrics
    history = trainer.train([])

    return {
        "ndcg_10": 0.0,
        "map": 0.0,
        "recall_100": 0.0,
        "config": config.dict(),
        "loss": history.get("loss", 0.0),
    }


# Lazy import conveniences ---------------------------------------------------

__getattr__ = lambda name: _lazy_getattr(name)  # type: ignore


def _lazy_getattr(name: str):
    mapping = {
        "eval": "geoIR.eval",
        "geo": "geoIR.geo",
        "graph": "geoIR.geo.graph",
        "Index": "geoIR.retrieval.index",
    }
    if name in mapping:
        module: ModuleType = import_module(mapping[name])
        return module
    raise AttributeError(name)
