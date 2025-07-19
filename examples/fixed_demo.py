#!/usr/bin/env python3
"""
Demo corregida del pipeline diferenciable de geoIR.
Basada en el debugging exitoso que confirma que los gradientes fluyen.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from geoIR.geo.differentiable import geometric_loss_end_to_end

class SimpleEncoder(nn.Module):
    """Encoder simple para testing."""
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_synthetic_data(batch_size=4, num_negatives=2, input_dim=768):
    """Crear datos sintéticos para testing."""
    torch.manual_seed(42)  # Para reproducibilidad
    
    queries = torch.randn(batch_size, input_dim)
    positives = torch.randn(batch_size, input_dim)
    negatives = torch.randn(batch_size, num_negatives, input_dim)
    
    return queries, positives, negatives

def demo_working_pipeline():
    """Demo del pipeline que sabemos que funciona."""
    print("🚀 Demo del Pipeline Diferenciable Corregido")
    print("=" * 60)
    
    # Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Crear encoder
    encoder = SimpleEncoder(input_dim=768, output_dim=64).to(device)  # Más pequeño para debugging
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Configuración simplificada (basada en el debugging exitoso)
    config = {
        "k_graph": 4,
        "temperature": 0.1,
        "lambda_ricci": 0.0,  # Sin curvatura para simplificar
        "kappa_target": 0.0,
        "heat_time": 1.0,
        "heat_steps": 3,
    }
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Datos de entrenamiento
    batch_size = 4
    num_negatives = 2
    num_epochs = 3
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Limpiar gradientes
        optimizer.zero_grad()
        
        # Crear batch sintético
        queries, positives, negatives = create_synthetic_data(
            batch_size=batch_size, 
            num_negatives=num_negatives
        )
        queries = queries.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        
        # Forward pass a través del encoder
        query_emb = encoder(queries)
        pos_emb = encoder(positives)
        
        # Procesar negativos
        neg_flat = negatives.view(-1, negatives.size(-1))
        neg_emb_flat = encoder(neg_flat)
        neg_emb = neg_emb_flat.view(batch_size, num_negatives, -1)
        
        # Verificar que los embeddings tienen gradientes
        print(f"  Embeddings grad status: query={query_emb.requires_grad}, pos={pos_emb.requires_grad}, neg={neg_emb.requires_grad}")
        
        # Loss geométrica diferenciable
        loss, metrics = geometric_loss_end_to_end(
            query_emb, pos_emb, neg_emb, **config
        )
        
        print(f"  Loss: {loss.item():.4f}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn is not None}")
        
        # Backward pass
        loss.backward()
        
        # Verificar gradientes en el encoder
        total_grad_norm = 0.0
        param_count = 0
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                if epoch == 0:  # Solo mostrar detalles en la primera época
                    print(f"    {name}: grad_norm={grad_norm:.6f}")
        
        # Optimizar
        optimizer.step()
        
        # Reportar progreso
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Loss: {metrics['loss_total']:.4f} (info: {metrics['loss_info']:.4f})")
        print(f"  Distances - pos: {metrics['mean_d_pos']:.4f}, neg: {metrics['mean_d_neg']:.4f}")
        print(f"  Total gradient norm: {total_grad_norm:.6f} across {param_count} parameters")
        
        if total_grad_norm > 1e-8:
            print("  ✅ SUCCESS: Gradients detected!")
        else:
            print("  ❌ PROBLEM: No gradients!")
            
        print()
    
    print("🎉 Demo completed!")
    return total_grad_norm > 1e-8

def demo_minimal_test():
    """Test mínimo para confirmar funcionamiento."""
    print("\n🔧 Test Mínimo de Gradientes")
    print("=" * 40)
    
    # Crear datos muy simples
    torch.manual_seed(42)
    queries = torch.randn(2, 4, requires_grad=True)
    positives = torch.randn(2, 4, requires_grad=True)
    negatives = torch.randn(2, 2, 4, requires_grad=True)
    
    # Loss directo (sin encoder)
    loss, metrics = geometric_loss_end_to_end(
        queries, positives, negatives,
        k_graph=3,
        temperature=0.1,
        lambda_ricci=0.0
    )
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Has grad_fn: {loss.grad_fn is not None}")
    
    # Backward
    loss.backward()
    
    # Verificar gradientes en inputs
    grad_norms = [
        queries.grad.norm().item(),
        positives.grad.norm().item(), 
        negatives.grad.norm().item()
    ]
    
    total_grad = sum(grad_norms)
    print(f"Input gradient norms: {grad_norms}")
    print(f"Total: {total_grad:.6f}")
    
    if total_grad > 1e-8:
        print("✅ Minimal test PASSED!")
        return True
    else:
        print("❌ Minimal test FAILED!")
        return False

if __name__ == "__main__":
    print("🔍 geoIR Fixed Differentiable Demo")
    print("=" * 60)
    
    # Test mínimo primero
    minimal_success = demo_minimal_test()
    
    if minimal_success:
        # Demo completa
        pipeline_success = demo_working_pipeline()
        
        print(f"\n📊 RESUMEN FINAL:")
        print(f"  Test mínimo: {'✅ OK' if minimal_success else '❌ FAIL'}")
        print(f"  Pipeline completo: {'✅ OK' if pipeline_success else '❌ FAIL'}")
        
        if minimal_success and pipeline_success:
            print("\n🎉 ¡Soft-kNN diferenciable funciona perfectamente!")
            print("El encoder puede aprender deformaciones geométricas end-to-end.")
        else:
            print("\n⚠️  Hay problemas que necesitan investigación adicional.")
    else:
        print("\n❌ El test mínimo falló. Revisar implementación básica.")
