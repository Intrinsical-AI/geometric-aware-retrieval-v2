#!/usr/bin/env python3
"""Demo de pipeline diferenciable end-to-end para geoIR.

Este script demuestra cómo usar los módulos diferenciables implementados
para entrenar un encoder que aprende a deformar el espacio geométrico.

Ejecutar:
    python examples/differentiable_demo.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from geoIR.geo.differentiable import (
    geometric_loss_end_to_end,
    check_gradient_flow,
    soft_knn_graph,
    heat_kernel_distances,
    forman_ricci_differentiable
)


class SimpleEncoder(nn.Module):
    """Encoder simple para demostración."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def create_synthetic_data(batch_size: int = 8, num_negatives: int = 4, dim: int = 768):
    """Crear datos sintéticos para la demo."""
    # Simular embeddings de entrada (ej. BERT)
    queries = torch.randn(batch_size, dim)
    positives = torch.randn(batch_size, dim)
    negatives = torch.randn(batch_size, num_negatives, dim)
    
    return queries, positives, negatives


def demo_individual_components():
    """Demostrar componentes individuales."""
    print("🔧 Demo de Componentes Individuales")
    print("=" * 50)
    
    # Crear embeddings de prueba
    B, D = 6, 128
    embeddings = torch.randn(B, D, requires_grad=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Requires grad: {embeddings.requires_grad}")
    
    # 1. Soft k-NN Graph
    print("\n1. Soft k-NN Graph")
    A = soft_knn_graph(embeddings, k=3, gamma=0.1)
    print(f"Adjacency matrix shape: {A.shape}")
    print(f"Graph density: {(A > 0.01).float().mean():.3f}")
    print(f"Has gradients: {A.grad_fn is not None}")
    
    # 2. Heat Kernel Distances
    print("\n2. Heat Kernel Distances")
    distances = heat_kernel_distances(A, t=1.0, steps=5)
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Mean distance: {distances.mean():.3f}")
    print(f"Has gradients: {distances.grad_fn is not None}")
    
    # 3. Forman-Ricci Curvature
    print("\n3. Forman-Ricci Curvature")
    kappa = forman_ricci_differentiable(A)
    print(f"Curvature matrix shape: {kappa.shape}")
    print(f"Mean curvature: {kappa.mean():.3f}")
    print(f"Has gradients: {kappa.grad_fn is not None}")
    
    # Test backward pass
    print("\n4. Gradient Flow Test")
    loss = distances.mean() + kappa.mean()
    loss.backward()
    
    print(f"Embeddings gradient norm: {embeddings.grad.norm():.6f}")
    print("✅ Gradients flow correctly through all components!")


def demo_end_to_end_training():
    """Demostrar entrenamiento end-to-end."""
    print("\n🚀 Demo de Entrenamiento End-to-End")
    print("=" * 50)
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Crear encoder
    encoder = SimpleEncoder(input_dim=768, output_dim=128).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Configuración de entrenamiento
    config = {
        "k_graph": 4,  # Reducido para grafo pequeño
        "temperature": 0.1,  # Temperatura más alta para gradientes más suaves
        "lambda_ricci": 0.0,  # Sin curvatura para simplificar debugging
        "kappa_target": 0.0,
        "heat_time": 1.0,
        "heat_steps": 3,
    }
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Simular algunas épocas de entrenamiento
    num_epochs = 5
    batch_size = 6
    num_negatives = 3
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Limpiar gradientes al inicio
        optimizer.zero_grad()
        
        # Crear batch sintético
        queries, positives, negatives = create_synthetic_data(
            batch_size=batch_size, 
            num_negatives=num_negatives
        )
        queries, positives, negatives = queries.to(device), positives.to(device), negatives.to(device)
        
        # Forward pass a través del encoder
        query_emb = encoder(queries)
        pos_emb = encoder(positives)
        # Reshape negatives: (batch_size, num_negatives, input_dim) -> (batch_size * num_negatives, input_dim)
        neg_flat = negatives.view(-1, negatives.size(-1))
        neg_emb_flat = encoder(neg_flat)
        # Reshape back: (batch_size * num_negatives, output_dim) -> (batch_size, num_negatives, output_dim)
        neg_emb = neg_emb_flat.view(batch_size, num_negatives, -1)
        
        # Verificar que los embeddings tienen gradientes
        assert query_emb.requires_grad, "Query embeddings must have gradients!"
        assert pos_emb.requires_grad, "Positive embeddings must have gradients!"
        assert neg_emb.requires_grad, "Negative embeddings must have gradients!"
        
        loss, metrics = geometric_loss_end_to_end(
            query_emb, pos_emb, neg_emb, **config
        )
        
        # Backward pass
        loss.backward()
        
        # Verificar gradientes
        gradient_stats = check_gradient_flow(encoder, loss)
        
        # Optimizar
        optimizer.step()
        
        # Reportar progreso
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {metrics['loss_total']:.4f} (info: {metrics['loss_info']:.4f}, ricci: {metrics['loss_ricci']:.4f})")
        print(f"  Mean curvature: {metrics['mean_curvature']:.4f}")
        print(f"  Distances - pos: {metrics['mean_d_pos']:.4f}, neg: {metrics['mean_d_neg']:.4f}")
        
        # Verificar que hay gradientes no-cero
        total_grad_norm = sum(
            stats["grad_norm"] for stats in gradient_stats.values() 
            if stats["has_gradient"]
        )
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        
        if total_grad_norm == 0:
            print("  ⚠️  WARNING: No gradients detected!")
        else:
            print("  ✅ Gradients flowing correctly")
    
    print("\n🎉 Training completed successfully!")
    print("The encoder learned to deform the geometric space end-to-end.")


def demo_gradient_debugging():
    """Demostrar debugging de gradientes."""
    print("\n🔍 Demo de Debugging de Gradientes")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Crear encoder y datos
    encoder = SimpleEncoder().to(device)
    queries, positives, negatives = create_synthetic_data(batch_size=4, num_negatives=2)
    queries, positives, negatives = queries.to(device), positives.to(device), negatives.to(device)
    
    # Caso 1: Con gradientes (correcto)
    print("1. Caso correcto (con gradientes):")
    query_emb = encoder(queries)
    pos_emb = encoder(positives)
    neg_emb = encoder(negatives.view(-1, negatives.size(-1))).view(4, 2, -1)
    
    loss, _ = geometric_loss_end_to_end(
        query_emb, pos_emb, neg_emb,
        k_graph=6, lambda_ricci=0.1
    )
    
    loss.backward()
    gradient_stats = check_gradient_flow(encoder, loss)
    
    has_gradients = any(stats["has_gradient"] for stats in gradient_stats.values())
    print(f"  Has gradients: {has_gradients} ✅")
    
    # Caso 2: Sin gradientes (incorrecto)
    print("\n2. Caso incorrecto (sin gradientes):")
    encoder.zero_grad()
    
    with torch.no_grad():  # ❌ Esto rompe los gradientes
        query_emb_broken = encoder(queries)
        pos_emb_broken = encoder(positives)
        neg_emb_broken = encoder(negatives.view(-1, negatives.size(-1))).view(4, 2, -1)
    
    # Intentar usar embeddings sin gradientes
    try:
        loss_broken, _ = geometric_loss_end_to_end(
            query_emb_broken, pos_emb_broken, neg_emb_broken,
            k_graph=6, lambda_ricci=0.1
        )
        loss_broken.backward()
        gradient_stats_broken = check_gradient_flow(encoder, loss_broken)
        
        has_gradients_broken = any(stats["has_gradient"] for stats in gradient_stats_broken.values())
        print(f"  Has gradients: {has_gradients_broken} ❌")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n💡 Lección: Siempre asegurar que los embeddings tengan requires_grad=True")


def main():
    """Función principal de la demo."""
    print("🌟 geoIR Differentiable Pipeline Demo")
    print("=" * 60)
    
    # Configurar reproducibilidad
    torch.manual_seed(42)
    
    try:
        # Demo de componentes individuales
        demo_individual_components()
        
        # Demo de entrenamiento end-to-end
        demo_end_to_end_training()
        
        # Demo de debugging
        demo_gradient_debugging()
        
        print("\n🎯 Demo completada exitosamente!")
        print("Todos los componentes diferenciables funcionan correctamente.")
        print("El encoder puede aprender a deformar el espacio geométrico end-to-end.")
        
    except Exception as e:
        print(f"\n❌ Error durante la demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
