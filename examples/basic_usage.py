#!/usr/bin/env python3
"""
Basic usage example for NSA kernels.
Shows the CORRECT way to handle K tensor strides.
"""

import torch
import triton
import math
from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
)

def sliding_window_attention(q, k, v, window_size=256):
    """
    Example of correct NSA sliding window usage.
    
    CRITICAL: K must be shape [B, G, dk, T] and we swap strides when calling kernel!
    
    Args:
        q: [B, H, T, dk] query tensor  
        k: [B, G, dk, T] key tensor - NOTE THE SHAPE!
        v: [B, G, T, dv] value tensor
        window_size: sliding window size
        
    Returns:
        output: [B, H, T, dv]
    """
    B, H, T, dk = q.shape
    _, G, _, _ = k.shape
    dv = v.shape[-1]
    
    # Validate inputs
    assert k.shape == (B, G, dk, T), f"K must be [B, G, dk, T], got {k.shape}"
    assert v.shape == (B, G, T, dv), f"V must be [B, G, T, dv], got {v.shape}"
    assert H % G == 0, f"H={H} must be divisible by G={G}"
    
    # Prepare outputs
    out = torch.zeros(B, H, T, dv, device=q.device, dtype=q.dtype)
    L = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=q.device, dtype=torch.float32)
    
    # Scale factor
    scale = 1.0 / math.sqrt(dk)
    
    # Launch forward kernel
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    
    _nsa_sliding_window_fwd_kernel[grid](
        q, k, v, out,
        scale, L, M,
        # Q strides - normal
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides - CRITICAL: swap stride(3) and stride(2)!
        # Kernel expects: stride_kn (time), stride_kk (dk)
        # But K shape is [B, G, dk, T] so stride(3)=time, stride(2)=dk
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # SWAPPED!
        # V strides - normal
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Output strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        # Kernel parameters
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    return out, L, M


def sliding_window_backward(q, k, v, out, grad_out, L, M, window_size=256):
    """
    Backward pass for sliding window attention.
    
    CRITICAL: dK must be allocated as [B, G, T, dk] for kernel!
    """
    B, H, T, dk_dim = q.shape
    _, G, _, _ = k.shape
    dv_dim = v.shape[-1]
    
    scale = 1.0 / math.sqrt(dk_dim)
    
    # Allocate gradients
    dq = torch.zeros_like(q)
    # CRITICAL: dK must be [B, G, T, dk] for atomic operations in kernel
    dk = torch.zeros(B, G, T, dk_dim, device=k.device, dtype=k.dtype)
    dv = torch.zeros_like(v)
    
    # Launch backward kernel
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    
    _nsa_sliding_window_bwd_kernel[grid](
        q, k, v, out, grad_out,
        scale, L, M,
        # Same stride pattern as forward
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # SWAPPED!
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        dq, dk, dv,
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk_dim, HEAD_DIM_V=dv_dim,
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Transpose dK back to match K's shape [B, G, dk, T]
    dk = dk.transpose(-2, -1)
    
    return dq, dk, dv


if __name__ == "__main__":
    # Example usage
    B, H, G, T = 2, 8, 2, 512
    dk = dv = 64
    
    # Create tensors with CORRECT shapes
    q = torch.randn(B, H, T, dk, device='cuda')
    k = torch.randn(B, G, dk, T, device='cuda')  # Note: [B, G, dk, T]
    v = torch.randn(B, G, T, dv, device='cuda')
    
    # Forward pass
    out, L, M = sliding_window_attention(q, k, v, window_size=256)
    print(f"Forward output shape: {out.shape}")
    
    # Backward pass
    grad_out = torch.randn_like(out)
    dq, dk, dv = sliding_window_backward(q, k, v, out, grad_out, L, M, window_size=256)
    print(f"dQ shape: {dq.shape}")
    print(f"dK shape: {dk.shape}")  # Should match K shape
    print(f"dV shape: {dv.shape}")
    
    print("\n✅ Example completed successfully!")
    print("Remember: Always check K tensor shape and stride order!")