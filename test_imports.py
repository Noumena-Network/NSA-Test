#!/usr/bin/env python3
"""Test all module imports and basic functionality."""

import torch

print("Testing NSA module imports...")

# Test basic imports
from nsa import (
    NSAConfig,
    NSAAttention,
    _nsa_sliding_window_fwd_kernel,
    _nsa_compression_fwd_kernel,
    _nsa_sparse_selection_fwd_kernel,
)

print("✅ All imports successful!")

# Test basic config creation
print("\nTesting NSAConfig...")
config = NSAConfig(
    d_model=512,
    head_dim_qk=64,
    head_dim_v=64,
    n_heads=8,
    n_kv_groups=2,
)
print(f"  Created config: d_model={config.d_model}, n_heads={config.n_heads}")
print("✅ NSAConfig works!")

# Test basic module creation
print("\nTesting NSAAttention module...")
module = NSAAttention(config).cuda()
print(f"  Created module with {sum(p.numel() for p in module.parameters())} parameters")
print("✅ NSAAttention module works!")

# Test a small forward pass
print("\nTesting small forward pass...")
B, T, D = 1, 32, 512
x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
with torch.no_grad():
    output, info = module(x, output_attentions=True)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
assert output.shape == x.shape, "Output shape mismatch!"
print("✅ Forward pass works!")

# Test kernel access
print("\nTesting kernel access...")
grid = lambda META: (1, 1)
print("  Sliding window kernel: ", _nsa_sliding_window_fwd_kernel)
print("  Compression kernel: ", _nsa_compression_fwd_kernel)
print("  Selection kernel: ", _nsa_sparse_selection_fwd_kernel)
print("✅ All kernels accessible!")

print("\n" + "=" * 60)
print("✅ ALL IMPORT TESTS PASSED!")
print("=" * 60)
