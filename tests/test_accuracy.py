#!/usr/bin/env python3
"""
Test production accuracy requirements for NSA kernels.
Must achieve <0.001 relative error for production implementation.
"""

import torch
import torch.nn.functional as F
import triton
import math

from nsa import (
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
)


def test_production_accuracy():
    """Test that we meet production accuracy requirements."""
    print("\n" + "=" * 80)
    print("PRODUCTION ACCURACY TEST")
    print("Requirement: <0.001 relative error (production target)")
    print("=" * 80)

    # Test configuration matching production use
    B = 2
    H = 8  # More heads to test GQA thoroughly
    G = 2  # 4 heads per group
    T = 64
    dk = dv = 32
    window_size = 16
    device = "cuda"
    dtype = torch.float32

    print("\nConfiguration:")
    print(f"  B={B}, H={H}, G={G}, T={T}")
    print(f"  Heads per group: {H // G}")
    print(f"  dk={dk}, dv={dv}, window={window_size}")

    # Create test tensors with realistic values
    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype) * 0.02
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype) * 0.02
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype) * 0.02

    print("\nTensor shapes and strides:")
    print(f"  Q: {q.shape}, strides: {q.stride()}")
    print(f"  K: {k.shape}, strides: {k.stride()}")
    print(f"  V: {v.shape}, strides: {v.stride()}")

    # Enable gradient computation
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    scale = 1.0 / math.sqrt(dk)

    print("\n[1] PyTorch Reference (Ground Truth):")

    # Compute reference with PyTorch
    # Expand K/V for all heads
    heads_per_group = H // G
    k_expanded = (
        k.unsqueeze(2).expand(B, G, heads_per_group, dk, T).reshape(B, H, dk, T)
    )
    v_expanded = (
        v.unsqueeze(2).expand(B, G, heads_per_group, T, dv).reshape(B, H, T, dv)
    )

    # Compute attention for each position
    outputs = []
    for b in range(B):
        for h in range(H):
            out_h = []
            for t in range(T):
                # Window bounds
                start = max(0, t - window_size + 1)
                end = t + 1

                # Get Q, K, V for this position
                q_t = q[b : b + 1, h : h + 1, t : t + 1, :]  # [1, 1, 1, dk]
                k_window = k_expanded[
                    b : b + 1, h : h + 1, :, start:end
                ]  # [1, 1, dk, window]
                v_window = v_expanded[
                    b : b + 1, h : h + 1, start:end, :
                ]  # [1, 1, window, dv]

                # Compute attention
                scores = torch.matmul(q_t, k_window) * scale  # [1, 1, 1, window]
                attn = F.softmax(scores, dim=-1)
                out_t = torch.matmul(attn, v_window)  # [1, 1, 1, dv]

                out_h.append(out_t.squeeze())
            outputs.append(torch.stack(out_h))

    out_ref = torch.stack([torch.stack(outputs[b * H : (b + 1) * H]) for b in range(B)])

    # Compute gradients
    grad_out = torch.randn_like(out_ref) * 0.01
    out_ref.backward(grad_out)

    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    print(f"  Forward output norm: {out_ref.norm():.6f}")
    print(f"  dQ norm: {dq_ref.norm():.6f}")
    print(f"  dK norm: {dk_ref.norm():.6f}")
    print(f"  dV norm: {dv_ref.norm():.6f}")

    # Reset for kernel test
    q.grad = None
    k.grad = None
    v.grad = None

    print("\n[2] NSA Kernel Implementation:")

    # Forward kernel
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    # CRITICAL FIX: Swap stride_kn and stride_kk to match kernel expectations
    # Kernel expects: stride_kn = time axis, stride_kk = dk axis
    # But K has shape [B, G, dk, T] so:
    # - k.stride(2) is dk axis -> should be stride_kk
    # - k.stride(3) is T axis -> should be stride_kn
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
        k.stride(2),  # SWAPPED!
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

    # Backward kernel - CRITICAL: dK must be [B, G, T, dk] not [B, G, dk, T]
    dq_kernel = torch.zeros_like(q)
    # dk_kernel shape must be [B, G, T, dk] for kernel's atomic indexing
    dk_kernel = torch.zeros(B, G, T, dk, device=device, dtype=dtype)
    dv_kernel = torch.zeros_like(v)

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
        k.stride(2),  # SWAPPED!
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

    # Transpose dK back to [B, G, dk, T] for comparison with reference
    dk_kernel_transposed = dk_kernel.transpose(-2, -1)  # [B, G, dk, T]

    print(f"  Forward output norm: {out_kernel.norm():.6f}")
    print(f"  dQ norm: {dq_kernel.norm():.6f}")
    print(f"  dK norm: {dk_kernel_transposed.norm():.6f}")
    print(f"  dV norm: {dv_kernel.norm():.6f}")

    print("\n[3] PRODUCTION ACCURACY CHECK:")

    # Compute relative errors (use transposed dK for comparison)
    forward_err = (out_kernel - out_ref).norm() / out_ref.norm()
    dq_err = (dq_kernel - dq_ref).norm() / dq_ref.norm()
    dk_err = (dk_kernel_transposed - dk_ref).norm() / dk_ref.norm()
    dv_err = (dv_kernel - dv_ref).norm() / dv_ref.norm()

    print(f"  Forward relative error: {forward_err:.6f}")
    print(f"  dQ relative error: {dq_err:.6f}")
    print(f"  dK relative error: {dk_err:.6f}")
    print(f"  dV relative error: {dv_err:.6f}")

    # Production requirement: all errors < 0.001
    threshold = 0.001

    print(f"\n  Production Threshold: {threshold}")

    passed = True
    if forward_err >= threshold:
        print(f"  ❌ Forward FAILED: {forward_err:.6f} >= {threshold}")
        passed = False
    else:
        print(f"  ✅ Forward PASSED: {forward_err:.6f} < {threshold}")

    if dq_err >= threshold:
        print(f"  ❌ dQ FAILED: {dq_err:.6f} >= {threshold}")
        passed = False
    else:
        print(f"  ✅ dQ PASSED: {dq_err:.6f} < {threshold}")

    if dk_err >= threshold:
        print(f"  ❌ dK FAILED: {dk_err:.6f} >= {threshold}")
        passed = False
    else:
        print(f"  ✅ dK PASSED: {dk_err:.6f} < {threshold}")

    if dv_err >= threshold:
        print(f"  ❌ dV FAILED: {dv_err:.6f} >= {threshold}")
        passed = False
    else:
        print(f"  ✅ dV PASSED: {dv_err:.6f} < {threshold}")

    print("\n" + "=" * 80)
    if passed:
        print("✅ PRODUCTION READY: All accuracy requirements met!")
        print("Implementation meets production accuracy standards.")
    else:
        print("❌ NOT PRODUCTION READY: Accuracy requirements not met.")
        print("Further optimization needed for deployment.")

    print("=" * 80)


if __name__ == "__main__":
    test_production_accuracy()
