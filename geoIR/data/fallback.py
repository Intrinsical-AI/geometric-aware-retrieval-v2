"""Lightweight dataset fallback when HF datasets unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import List


class MinimalDataset:
    """Lightweight dataset fallback when HF datasets unavailable.
    
    This provides a simple interface compatible with HuggingFace datasets
    for basic text loading scenarios.
    
    Attributes
    ----------
    texts : List[str]
        List of document texts.
    docs : List[str]
        Alias for texts (compatibility with existing code).
    """
    
    def __init__(self, texts: List[str]):
        """Initialize with list of texts.
        
        Parameters
        ----------
        texts : List[str]
            Document texts, one per item.
        """
        self.texts = texts
        self.docs = texts  # Compatibility alias
    
    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get item by index in HF datasets format."""
        return {"text": self.texts[idx]}
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.texts)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MinimalDataset({len(self.texts)} docs)"


def load_text_file(
    path: str, 
    max_docs: int | None = None,
    encoding: str = "utf-8"
) -> MinimalDataset:
    """Load documents from a text file (one document per line).
    
    Parameters
    ----------
    path : str
        Path to text file.
    max_docs : int, optional
        Maximum number of documents to load.
    encoding : str, default "utf-8"
        File encoding.
        
    Returns
    -------
    MinimalDataset
        Dataset with loaded documents.
        
    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    """
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Dataset path '{path}' not found")
    
    with txt_path.open("r", encoding=encoding) as fin:
        docs = [line.strip() for line in fin if line.strip()]
    
    if max_docs:
        docs = docs[:max_docs]
    
    return MinimalDataset(docs)


def create_dummy_dataset(texts: List[str]) -> MinimalDataset:
    """Create a minimal dataset from a list of texts.
    
    Parameters
    ----------
    texts : List[str]
        List of document texts.
        
    Returns
    -------
    MinimalDataset
        Dataset with the provided texts.
    """
    return MinimalDataset(texts)
