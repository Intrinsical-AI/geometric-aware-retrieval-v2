#!/usr/bin/env python3
"""Módulo de utilidades generales para el proyecto."""

import random
import time
from functools import wraps
from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def seed_all(seed: int = 42):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timeit(fn):
    """Decorador simple para medir tiempo de ejecución."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        print(f"{fn.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def save_tensor(path: Path, tensor: torch.Tensor):
    torch.save(tensor, path)


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path)


def encode_corpus(
    model: SentenceTransformer,
    corpus_texts: List[str],
    batch_size: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Codifica una lista de textos usando el modelo de SentenceTransformers."""
    model.to(device)
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embeddings.cpu()
