import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoRALinear(nn.Module):
    """
    DoRA: Weight-Decomposed Low-Rank Adaptation Layer.
    Implementation based on Liu et al. (2024).
    """
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.05, weight=None, bias=None):
        super().__init__()
        
        # 1. Base Weight (Frozen)
        # In practice, this references the pre-trained weight directly
        if weight is None:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = weight # Shared reference, frozen
        
        if bias is not None:
            self.bias = bias
        else:
            self.bias = None

        # 2. LoRA Branch Matrices (A and B)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # 3. DoRA Core: Magnitude Vector 'm' (Trainable)
        # Initialize m to match the column norm of the original weight
        # This ensures that initially, W' == W0
        with torch.no_grad():
            self.m = nn.Parameter(self.weight.norm(p=2, dim=1, keepdim=True))
        
        # Initialize LoRA parameters
        # A: Random Gaussian, B: Zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Forward pass implementing: W' = (m + dm) * (V0 + BA) / ||V0 + BA||
        """
        # Step 1: Calculate LoRA Update
        # lora_update shape: [out_features, in_features]
        lora_update = self.lora_B @ self.lora_A * self.scaling
        
        # Step 2: Combine to get Direction Matrix V
        # Note: self.weight is frozen W0
        weight_V = self.weight + lora_update
        
        # Step 3: Normalize V to isolate Direction
        # norm_V shape: [out_features, 1]
        norm_V = weight_V.norm(p=2, dim=1, keepdim=True)
        direction = weight_V / (norm_V + 1e-6)
        
        # Step 4: Apply Trainable Magnitude 'm'
        new_weight = self.m * direction
        
        # Step 5: Linear Transformation
        return F.linear(x, new_weight, self.bias)

# Verification Block
if __name__ == "__main__":
    # Test dimensions
    batch_size, seq_len, hidden_dim = 2, 128, 768
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Instantiate DoRA Layer
    dora_layer = DoRALinear(hidden_dim, hidden_dim, rank=16)
    
    # Forward pass
    output = dora_layer(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")
    print("DoRA Layer forward pass successful.")
