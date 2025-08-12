# Native Sparse Attention (NSA)

Production-grade implementation of Native Sparse Attention from the paper:
**"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"** (arXiv:2502.11089v1)

## Features

- **Production implementation**: <0.001 relative error target for gradients
- **Three attention branches**:
  - Sliding window attention
  - Compression attention
  - Sparse selection attention
- **Optimized Triton kernels** with FlashAttention-style blocked computation
- **Grouped Query Attention (GQA)** support
- **Full forward and backward pass** implementations

## Installation

```bash
pip install -e .
```

## Requirements

- PyTorch >= 2.0
- Triton >= 2.0
- CUDA-capable GPU

## Quick Start

```python
import torch
import triton
import math
from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
)

# Configure dimensions
B, H, G, T = 2, 8, 2, 512  # Batch, Heads, Groups, Sequence length
dk = dv = 64  # Head dimensions
window_size = 256

# Create tensors
q = torch.randn(B, H, T, dk, device='cuda', dtype=torch.float32)
k = torch.randn(B, G, dk, T, device='cuda', dtype=torch.float32)  # Note: K shape
v = torch.randn(B, G, T, dv, device='cuda', dtype=torch.float32)

# Prepare outputs
out = torch.zeros(B, H, T, dv, device='cuda', dtype=torch.float32)
L = torch.zeros(B, H, T, device='cuda', dtype=torch.float32)
M = torch.zeros(B, H, T, device='cuda', dtype=torch.float32)

# Launch kernel
grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
scale = 1.0 / math.sqrt(dk)

_nsa_sliding_window_fwd_kernel[grid](
    q, k, v, out,
    scale, L, M,
    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # Note stride swap!
    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    L.stride(0), L.stride(1), L.stride(2),
    M.stride(0), M.stride(1), M.stride(2),
    Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
    WINDOW_SIZE=window_size,
    HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
    BLOCK_M=32, BLOCK_N=32,
)
```

## Critical Implementation Notes

### Gate Mechanism (Eq. 5)
By default, gating follows the paper: per-branch gates via MLP + sigmoid, applied per head and timestep.
Set `NSAConfig.gate_mode` to `"sigmoid"` (default, paper-faithful) or `"softmax"` (normalized).

### Selection (Eq. 8→9→10→11)
Block importance scores derive from compression attention (Eq. 8), mapped via the triangular rule (Eq. 9),
aggregated across heads in each GQA group (Eq. 10), then top-n blocks are selected per query (Eq. 11).
By default, selection includes 1 initial + 2 local blocks (paper training config). To disable this and use pure top‑n,
set `NSAConfig.include_fixed_in_selection=False`.

### K Tensor Layout
The K tensor uses shape `[B, G, dk, T]` but the kernel expects strides as if it were `[B, G, T, dk]`. 
**Always swap the last two stride values when calling kernels:**

```python
# Correct stride passing for K tensor with shape [B, G, dk, T]:
k.stride(0), k.stride(1), k.stride(3), k.stride(2)  # Swapped!
```

### Gradient Tensors
For backward pass, dK must be allocated as `[B, G, T, dk]` to match kernel expectations:

```python
dk = torch.zeros(B, G, T, dk, device='cuda', dtype=torch.float32)
```

## Running Tests

```bash
# Test accuracy (should achieve 0% error)
python tests/test_accuracy.py

# Test all branches
python tests/test_branches.py
```

## Benchmark Results

On NVIDIA A100:
- Sliding window: ~219K tokens/sec
- Compression: ~3K tokens/sec  
- Sliding/compression gradients: <0.001 relative error achieved

## Paper Configuration

From the paper (Table 1):
- Compression: l=32, d=16
- Selection: l'=64, n=16 (with 3 fixed blocks)
- Sliding window: size=512
- GQA: 4 heads per group

## Citation

```bibtex
@article{nsa2025,
  title={Native Sparse Attention: Achieving End-to-End Sparsity for Language Models},
  author={...},
  journal={arXiv preprint arXiv:2502.11089v1},
  year={2025}
}
```

## License

arXiv.org perpetual non-exclusive license

## Contributors

Production implementation by XJDR.
