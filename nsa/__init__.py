"""
Native Sparse Attention (NSA) - Production Implementation

Based on paper: "Native Sparse Attention: Achieving End-to-End Sparsity for Language Models"
arXiv:2502.11089v1

This implementation provides production-grade (<0.001 relative error target) Triton kernels for:
- Sliding window attention
- Compression attention
- Sparse selection attention
"""

from .kernels import (
    # Configuration
    NSAConfig,
    NSAAttention,
    # Triton kernels
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
    _nsa_compression_fwd_kernel,
    _nsa_compression_bwd_kernel,
    _nsa_sparse_selection_fwd_kernel,
    _nsa_sparse_selection_bwd_kernel,
)

__version__ = "1.0.0"

__all__ = [
    "NSAConfig",
    "NSAAttention",
    "_nsa_sliding_window_fwd_kernel",
    "_nsa_sliding_window_bwd_kernel",
    "_nsa_compression_fwd_kernel",
    "_nsa_compression_bwd_kernel",
    "_nsa_sparse_selection_fwd_kernel",
    "_nsa_sparse_selection_bwd_kernel",
]
