#!/usr/bin/env python3
"""
Simple test to verify gradient flow in the Soft-kNN pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from geoIR.geo.differentiable import geometric_loss_end_to_end

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def test_gradient_flow():
    print("🔧 Testing Gradient Flow in Soft-kNN Pipeline")
    print("=" * 50)
    
    # Create simple encoder
    encoder = SimpleEncoder()
    optimizer = optim.SGD(encoder.parameters(), lr=0.1)
    
    # Create diverse input data (important for non-degenerate loss)
    torch.manual_seed(42)
    queries = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], requires_grad=True)
    positives = torch.tensor([[1.1, 0.1, 0.0, 0.0], [0.1, 1.1, 0.0, 0.0]], requires_grad=True)
    negatives = torch.tensor([[[0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]], requires_grad=True)
    
    print(f"Input shapes: queries={queries.shape}, positives={positives.shape}, negatives={negatives.shape}")
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # Forward through encoder
        q_emb = encoder(queries)
        p_emb = encoder(positives)
        n_emb = encoder(negatives.squeeze(1)).unsqueeze(1)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Embeddings require_grad: {q_emb.requires_grad}, {p_emb.requires_grad}, {n_emb.requires_grad}")
        
        # Compute loss
        loss, metrics = geometric_loss_end_to_end(
            q_emb, p_emb, n_emb,
            k_graph=2,
            temperature=0.1,
            lambda_ricci=0.0
        )
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Pos distances: {metrics['mean_d_pos']:.4f}")
        print(f"  Neg distances: {metrics['mean_d_neg']:.4f}")
        
        # Backward
        loss.backward()
        
        # Check gradients
        for name, param in encoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name} grad norm: {grad_norm:.6f}")
            else:
                print(f"  {name} grad: None")
        
        # Update
        optimizer.step()
        
        # Check if loss is changing
        if epoch > 0 and abs(loss.item() - prev_loss) < 1e-6:
            print("  ⚠️  Loss not changing - possible gradient issue")
        else:
            print("  ✅ Loss changing - gradients working")
        
        prev_loss = loss.item()

def test_direct_connection():
    """Test direct connection between encoder and loss."""
    print("\n🔧 Testing Direct Encoder-Loss Connection")
    print("=" * 50)
    
    encoder = SimpleEncoder()
    optimizer = optim.SGD(encoder.parameters(), lr=0.1)
    
    # Simple data
    x = torch.randn(2, 4)
    target = torch.tensor([0, 1])
    
    for epoch in range(3):
        optimizer.zero_grad()
        
        # Forward
        output = encoder(x)
        loss = nn.CrossEntropyLoss()(output, target)
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        
        # Backward
        loss.backward()
        
        # Check gradients
        total_grad = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
        print(f"  Total grad norm: {total_grad:.6f}")
        
        # Update
        optimizer.step()

if __name__ == "__main__":
    # Test direct encoder connection first
    test_direct_connection()
    
    # Test geometric pipeline
    test_gradient_flow()
