#!/usr/bin/env python3
"""
Test NSA kernels at larger scale to ensure they work in production settings.
"""

import torch
import triton
import math
import time

from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
)


def test_large_scale():
    """Test with larger, more realistic configurations."""
    print("\n" + "=" * 80)
    print("LARGE SCALE PRODUCTION TEST")
    print("=" * 80)

    # Large scale configuration
    B = 4  # Batch size
    H = 32  # Number of heads (like GPT-3)
    G = 8  # 4 heads per group
    T = 2048  # Sequence length
    dk = 128  # Head dimension
    dv = 128
    window_size = 256
    device = "cuda"
    dtype = torch.float32  # Use fp32 to maximize numeric headroom

    print("\nLarge-scale configuration:")
    print(f"  B={B}, H={H}, G={G}, T={T}")
    print(f"  dk={dk}, dv={dv}, window={window_size}")
    print(f"  dtype={dtype}")
    print(f"  Total FLOPs: ~{2 * B * H * T * T * dk / 1e9:.2f} GFLOPs (full attention)")
    print(f"  Memory for Q/K/V: ~{3 * B * H * T * dk * 2 / 1e9:.2f} GB")

    torch.manual_seed(42)

    # Create tensors
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype) * 0.02
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype) * 0.02
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype) * 0.02

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    scale = 1.0 / math.sqrt(dk)

    print("\n[1] Testing forward pass...")

    out_kernel = torch.zeros(B, H, T, dv, device=device, dtype=dtype)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    # Warmup
    for _ in range(3):
        _nsa_sliding_window_fwd_kernel[grid](
            q,
            k,
            v,
            out_kernel,
            scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(3),
            k.stride(2),  # Fixed strides
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out_kernel.stride(0),
            out_kernel.stride(1),
            out_kernel.stride(2),
            out_kernel.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            Z=B,
            H=H,
            N_KV_GROUPS=G,
            N_CTX=T,
            WINDOW_SIZE=window_size,
            HEAD_DIM_QK=dk,
            HEAD_DIM_V=dv,
            BLOCK_M=32,
            BLOCK_N=32,
        )

    torch.cuda.synchronize()

    # Time forward pass
    start = time.time()
    _nsa_sliding_window_fwd_kernel[grid](
        q,
        k,
        v,
        out_kernel,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(3),
        k.stride(2),  # Fixed strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out_kernel.stride(0),
        out_kernel.stride(1),
        out_kernel.stride(2),
        out_kernel.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        Z=B,
        H=H,
        N_KV_GROUPS=G,
        N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )
    torch.cuda.synchronize()
    forward_time = time.time() - start

    print(f"  Forward pass completed in {forward_time * 1000:.2f} ms")
    print(f"  Output norm: {out_kernel.float().norm():.4f}")

    print("\n[2] Testing backward pass...")

    grad_out = torch.randn_like(out_kernel) * 0.01
    dq_kernel = torch.zeros_like(q)
    dk_kernel = torch.zeros(B, G, T, dk, device=device, dtype=dtype)
    dv_kernel = torch.zeros_like(v)

    # Warmup
    for _ in range(3):
        _nsa_sliding_window_bwd_kernel[grid](
            q,
            k,
            v,
            out_kernel,
            grad_out,
            scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(3),
            k.stride(2),  # Fixed strides
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out_kernel.stride(0),
            out_kernel.stride(1),
            out_kernel.stride(2),
            out_kernel.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            dq_kernel,
            dk_kernel,
            dv_kernel,
            dq_kernel.stride(0),
            dq_kernel.stride(1),
            dq_kernel.stride(2),
            dq_kernel.stride(3),
            dk_kernel.stride(0),
            dk_kernel.stride(1),
            dk_kernel.stride(2),
            dk_kernel.stride(3),
            dv_kernel.stride(0),
            dv_kernel.stride(1),
            dv_kernel.stride(2),
            dv_kernel.stride(3),
            Z=B,
            H=H,
            N_KV_GROUPS=G,
            N_CTX=T,
            WINDOW_SIZE=window_size,
            HEAD_DIM_QK=dk,
            HEAD_DIM_V=dv,
            BLOCK_M=32,
            BLOCK_N=32,
        )

    torch.cuda.synchronize()

    # Time backward pass
    start = time.time()
    _nsa_sliding_window_bwd_kernel[grid](
        q,
        k,
        v,
        out_kernel,
        grad_out,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(3),
        k.stride(2),  # Fixed strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out_kernel.stride(0),
        out_kernel.stride(1),
        out_kernel.stride(2),
        out_kernel.stride(3),
        L.stride(0),
        L.stride(1),
        L.stride(2),
        M.stride(0),
        M.stride(1),
        M.stride(2),
        dq_kernel,
        dk_kernel,
        dv_kernel,
        dq_kernel.stride(0),
        dq_kernel.stride(1),
        dq_kernel.stride(2),
        dq_kernel.stride(3),
        dk_kernel.stride(0),
        dk_kernel.stride(1),
        dk_kernel.stride(2),
        dk_kernel.stride(3),
        dv_kernel.stride(0),
        dv_kernel.stride(1),
        dv_kernel.stride(2),
        dv_kernel.stride(3),
        Z=B,
        H=H,
        N_KV_GROUPS=G,
        N_CTX=T,
        WINDOW_SIZE=window_size,
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )
    torch.cuda.synchronize()
    backward_time = time.time() - start

    print(f"  Backward pass completed in {backward_time * 1000:.2f} ms")
    print(f"  dQ norm: {dq_kernel.float().norm():.4f}")
    print(f"  dK norm: {dk_kernel.float().norm():.4f}")
    print(f"  dV norm: {dv_kernel.float().norm():.4f}")

    print("\n[3] Performance summary:")
    total_time = forward_time + backward_time
    print(f"  Total time: {total_time * 1000:.2f} ms")
    print(f"  Throughput: {B * T / total_time:.0f} tokens/sec")

    # Estimate FLOPS (rough approximation for sliding window)
    flops_per_token = 2 * H * window_size * dk  # Per token attention computation
    total_flops = B * T * flops_per_token * 3  # Forward + backward ~3x forward
    tflops = total_flops / total_time / 1e12
    print(f"  Estimated performance: {tflops:.2f} TFLOPS")

    print("\n" + "=" * 80)
    print("✅ LARGE SCALE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    test_large_scale()
