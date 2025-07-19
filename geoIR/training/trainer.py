"""Unified Trainer for geoIR models.

This module provides a generic `Trainer` class that can be configured for
different training modes (classic, geometric, etc.) to streamline the
fine-tuning process.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from geoIR.core.config import TrainerConfig
from geoIR.geo.curvature import forman_ricci_weighted, ricci_ollivier
from geoIR.geo.graph import build_knn_graph, shortest_paths_dense
from geoIR.losses import forman_loss, info_nce_geo, ricci_loss
from geoIR.retrieval.encoder import Encoder


class Trainer:
    """Unified trainer for geoIR models."""

    def __init__(self, encoder: Encoder, config: TrainerConfig):
        self.encoder = encoder
        self.config = config
        self.device = encoder.device

    def train(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Run the fine-tuning loop.

        Handles both classic and geometric training modes based on the config.
        """
        if self.config.is_classic_mode:
            return self._train_classic(triplets)
        return self._train_geometric(triplets)

    def _train_classic(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Fine-tune using standard sentence-transformers loss."""
        try:
            from sentence_transformers import InputExample, losses
            from sentence_transformers.datasets import NoDuplicatesDataLoader
        except ModuleNotFoundError:
            warnings.warn("Install `sentence_transformers` for fine-tuning support.")
            return {}

        train_examples = [InputExample(texts=[q, p, n]) for q, p, n in triplets]
        train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=self.config.batch_size)
        train_loss = losses.TripletLoss(model=self.encoder.q_model)

        self.encoder.q_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            optimizer_params={"lr": self.config.lr},
            show_progress_bar=self.config.verbose,
        )
        return {"loss": train_loss.get_last_loss()}

    def _train_geometric(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, float]:
        """Run the geometric fine-tuning loop with differentiable Soft-kNN.

        This implementation uses differentiable Soft-kNN graph construction to enable
        end-to-end gradient flow, allowing the encoder to learn geometric deformations.
        The graph is constructed dynamically during training with gradient tracking.
        """
        self.encoder.q_model.train()
        optimizer = torch.optim.Adam(self.encoder.q_model.parameters(), lr=self.config.lr)

        queries, pos_docs, neg_docs = zip(*triplets)
        all_texts = sorted(list(set(queries) | set(pos_docs) | set(neg_docs)))
        text_to_idx = {text: i for i, text in enumerate(all_texts)}

        dataset = TensorDataset(torch.arange(len(triplets)))
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history = {"loss": [], "loss_nce": [], "loss_ricci": [], "loss_forman": []}

        # Import differentiable modules
        from ..geo.differentiable import geometric_loss_end_to_end
        
        for epoch in range(self.config.epochs):
            if self.config.verbose:
                print(f"--- Epoch {epoch+1}/{self.config.epochs} ---")

            # Temperature scheduling for Soft-kNN (optional)
            gamma = max(0.05, 0.2 * (0.95 ** epoch))  # Decay from 0.2 to 0.05
            if self.config.verbose:
                print(f"  - Soft-kNN temperature: {gamma:.4f}")

            # 2. Training loop over batches
            for batch_indices_tensor in loader:
                optimizer.zero_grad()

                batch_indices = batch_indices_tensor[0].tolist()
                q_texts = [queries[i] for i in batch_indices]
                p_texts = [pos_docs[i] for i in batch_indices]
                n_texts = [neg_docs[i] for i in batch_indices]

                # Encode batch texts WITH gradients
                q_vecs = self.encoder.encode(q_texts, is_query=True)
                p_vecs = self.encoder.encode(p_texts, is_query=False)
                n_vecs = self.encoder.encode(n_texts, is_query=False)

                # --- Differentiable Geometric Loss ---
                if self.config.geodesic:
                    # Use differentiable geometric pipeline
                    n_vecs_reshaped = n_vecs.unsqueeze(1)  # [batch, 1, dim] for geometric_loss_end_to_end
                    
                    total_loss, metrics = geometric_loss_end_to_end(
                        q_vecs, p_vecs, n_vecs_reshaped,
                        k_graph=self.config.k_graph,
                        gamma=gamma,  # Use scheduled temperature
                        lambda_ricci=self.config.lambda_ricci,
                        kappa_target=self.config.kappa_target,
                        heat_time=getattr(self.config, 'heat_time', 1.0),
                        heat_steps=getattr(self.config, 'heat_steps', 5)
                    )
                    
                    loss_nce = metrics['loss_info']
                    loss_r = metrics.get('loss_ricci', 0.0)
                    loss_f = 0.0  # Forman not implemented in differentiable version yet
                    
                    if self.config.verbose and batch_indices_tensor[0].item() == 0:  # Log first batch
                        print(f"    Geometric distances - pos: {metrics['mean_d_pos']:.4f}, neg: {metrics['mean_d_neg']:.4f}")
                        
                else:
                    # Fallback to standard triplet loss if not using geodesic distances
                    loss_nce = F.triplet_margin_loss(q_vecs, p_vecs, n_vecs)
                    loss_r = torch.tensor(0.0, device=self.device)
                    loss_f = torch.tensor(0.0, device=self.device)
                    
                    total_loss = loss_nce

                total_loss.backward()
                optimizer.step()

                history["loss"].append(total_loss.item())
                history["loss_nce"].append(loss_nce.item())
                history["loss_ricci"].append(loss_r.item())
                history["loss_forman"].append(loss_f.item())

            epoch_loss = np.mean(history['loss'][-len(loader):])
            if self.config.verbose:
                print(f"  - Epoch Loss: {epoch_loss:.4f}")

        self.encoder.q_model.eval()
        return {k: float(np.mean(v)) for k, v in history.items()}

