"""Data loading utilities, including synthetic data generation for tests."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int,
    num_samples: int = 100,  # Smaller for quick tests
    embedding_dim: int = 384, # common for MiniLM
    num_classes: int = 10
) -> DataLoader:
    """
    Generates a DataLoader with synthetic data for a given split.

    Args:
        dataset_name: Name of the dataset (e.g., 'fiqa', 'synthetic'). Ignored for now.
        split: 'train', 'val', or 'test'. Affects the random seed.
        batch_size: Batch size for the DataLoader.
        num_samples: Number of samples to generate.
        embedding_dim: Dimensionality of the synthetic embeddings.
        num_classes: Number of distinct classes for labels.

    Returns:
        A PyTorch DataLoader yielding batches of (embeddings, labels).
    """
    # Use different seeds for train/val/test to get different data
    seed = 42
    if split == 'val':
        seed = 43
    elif split == 'test':
        seed = 44
    
    torch.manual_seed(seed)
    
    # Generate synthetic data
    embeddings = torch.randn(num_samples, embedding_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    
    return dataloader
