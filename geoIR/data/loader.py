"""Data loading utilities for geoIR experiments."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

def load_triplets(path: str) -> List[Tuple[str, str, str]]:
    """
    Load training triplets from a TSV file.
    Format: query\tpositive_doc\tnegative_doc
    """
    file_path = Path(path)
    if not file_path.exists():
        warnings.warn(f"Triplet file not found: {path}. Returning empty list.")
        return []

    triplets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triplets.append((parts[0], parts[1], parts[2]))
    return triplets

def load_corpus(path: str) -> List[str]:
    """
    Load a corpus from a text file, one document per line.
    """
    file_path = Path(path)
    if not file_path.exists():
        warnings.warn(f"Corpus file not found: {path}. Returning empty list.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
