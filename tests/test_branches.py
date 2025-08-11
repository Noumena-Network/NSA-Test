#!/usr/bin/env python3
"""
Comprehensive test for all three NSA branches: compression, selection, sliding window.
Tests forward and backward passes with production accuracy requirements.
"""

import torch
import torch.nn.functional as F
import triton
import math

from nsa import (
    _nsa_compression_fwd_kernel,
    _nsa_compression_bwd_kernel,
    _nsa_sliding_window_fwd_kernel,
    _nsa_sliding_window_bwd_kernel,
)


def test_sliding_window_branch():
    """Test sliding window attention branch."""
    print("\n" + "=" * 80)
    print("SLIDING WINDOW BRANCH TEST")
    print("=" * 80)

    B, H, G = 2, 8, 2
    T = 64
    dk = dv = 32
    window_size = 16
    device = "cuda"
    dtype = torch.float32

    print(f"\nConfiguration: B={B}, H={H}, G={G}, T={T}, window={window_size}")

    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype) * 0.02
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype) * 0.02
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype) * 0.02

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    scale = 1.0 / math.sqrt(dk)

    # Reference
    heads_per_group = H // G
    k_expanded = (
        k.unsqueeze(2).expand(B, G, heads_per_group, dk, T).reshape(B, H, dk, T)
    )
    v_expanded = (
        v.unsqueeze(2).expand(B, G, heads_per_group, T, dv).reshape(B, H, T, dv)
    )

    outputs = []
    for b in range(B):
        for h in range(H):
            out_h = []
            for t in range(T):
                start = max(0, t - window_size + 1)
                end = t + 1

                q_t = q[b : b + 1, h : h + 1, t : t + 1, :]
                k_window = k_expanded[b : b + 1, h : h + 1, :, start:end]
                v_window = v_expanded[b : b + 1, h : h + 1, start:end, :]

                scores = torch.matmul(q_t, k_window) * scale
                attn = F.softmax(scores, dim=-1)
                out_t = torch.matmul(attn, v_window)

                out_h.append(out_t.squeeze())
            outputs.append(torch.stack(out_h))

    out_ref = torch.stack([torch.stack(outputs[b * H : (b + 1) * H]) for b in range(B)])

    grad_out = torch.randn_like(out_ref) * 0.01
    out_ref.backward(grad_out)

    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    q.grad = None
    k.grad = None
    v.grad = None

    # Kernel
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

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
        k.stride(2),  # FIXED strides
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

    dq_kernel = torch.zeros_like(q)
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
        k.stride(2),  # FIXED strides
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

    dk_kernel_transposed = dk_kernel.transpose(-2, -1)

    # Check errors
    forward_err = (out_kernel - out_ref).norm() / out_ref.norm()
    dq_err = (dq_kernel - dq_ref).norm() / dq_ref.norm()
    dk_err = (dk_kernel_transposed - dk_ref).norm() / dk_ref.norm()
    dv_err = (dv_kernel - dv_ref).norm() / dv_ref.norm()

    print("\nResults:")
    print(f"  Forward error: {forward_err:.6f}")
    print(f"  dQ error: {dq_err:.6f}")
    print(f"  dK error: {dk_err:.6f}")
    print(f"  dV error: {dv_err:.6f}")

    threshold = 0.001
    passed = (
        forward_err < threshold
        and dq_err < threshold
        and dk_err < threshold
        and dv_err < threshold
    )

    if passed:
        print(f"  ✅ PASSED (all errors < {threshold})")
    else:
        print(f"  ❌ FAILED (some errors >= {threshold})")

    return passed


def test_compression_branch():
    """Test compression attention branch."""
    print("\n" + "=" * 80)
    print("COMPRESSION BRANCH TEST")
    print("=" * 80)

    B, H, G = 2, 8, 2
    T = 64
    dk = dv = 32
    N_BLOCKS = 8  # Compress to 8 blocks
    BLOCK_SIZE = T // N_BLOCKS  # 8 positions per block
    device = "cuda"
    dtype = torch.float32

    print(f"\nConfiguration: B={B}, H={H}, G={G}, T={T}, N_BLOCKS={N_BLOCKS}")

    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype) * 0.02

    # Compressed K and V
    k_compressed = torch.randn(B, G, dk, N_BLOCKS, device=device, dtype=dtype) * 0.02
    v_compressed = torch.randn(B, G, N_BLOCKS, dv, device=device, dtype=dtype) * 0.02

    # Block ends for causality
    block_ends = torch.arange(
        BLOCK_SIZE - 1, T, BLOCK_SIZE, device=device, dtype=torch.int32
    )

    q.requires_grad_(True)
    k_compressed.requires_grad_(True)
    v_compressed.requires_grad_(True)

    scale = 1.0 / math.sqrt(dk)

    # Reference computation
    heads_per_group = H // G
    k_exp = (
        k_compressed.unsqueeze(2)
        .expand(B, G, heads_per_group, dk, N_BLOCKS)
        .reshape(B, H, dk, N_BLOCKS)
    )
    v_exp = (
        v_compressed.unsqueeze(2)
        .expand(B, G, heads_per_group, N_BLOCKS, dv)
        .reshape(B, H, N_BLOCKS, dv)
    )

    outputs = []
    for b in range(B):
        for h in range(H):
            out_h = []
            for t in range(T):
                q_t = q[b : b + 1, h : h + 1, t : t + 1, :]

                # Causal mask based on block ends
                mask = block_ends <= t
                valid_blocks = mask.sum().item()

                if valid_blocks > 0:
                    k_valid = k_exp[b : b + 1, h : h + 1, :, :valid_blocks]
                    v_valid = v_exp[b : b + 1, h : h + 1, :valid_blocks, :]

                    scores = torch.matmul(q_t, k_valid) * scale
                    attn = F.softmax(scores, dim=-1)
                    out_t = torch.matmul(attn, v_valid)
                else:
                    out_t = torch.zeros(1, 1, 1, dv, device=device, dtype=dtype)

                out_h.append(out_t.squeeze())
            outputs.append(torch.stack(out_h))

    out_ref = torch.stack([torch.stack(outputs[b * H : (b + 1) * H]) for b in range(B)])

    grad_out = torch.randn_like(out_ref) * 0.01
    out_ref.backward(grad_out)

    dq_ref = q.grad.clone()
    dk_ref = k_compressed.grad.clone()
    dv_ref = v_compressed.grad.clone()

    q.grad = None
    k_compressed.grad = None
    v_compressed.grad = None

    # Kernel
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    _nsa_compression_fwd_kernel[grid](
        q,
        k_compressed,
        v_compressed,
        block_ends,
        out_kernel,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_compressed.stride(0),
        k_compressed.stride(1),
        k_compressed.stride(3),
        k_compressed.stride(2),  # FIXED
        v_compressed.stride(0),
        v_compressed.stride(1),
        v_compressed.stride(2),
        v_compressed.stride(3),
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
        N_CTX_Q=T,
        N_BLOCKS=N_BLOCKS,
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )

    dq_kernel = torch.zeros_like(q)
    dk_kernel = torch.zeros(B, G, N_BLOCKS, dk, device=device, dtype=dtype)
    dv_kernel = torch.zeros_like(v_compressed)

    _nsa_compression_bwd_kernel[grid](
        q,
        k_compressed,
        v_compressed,
        block_ends,
        out_kernel,
        grad_out,
        scale,
        L,
        M,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_compressed.stride(0),
        k_compressed.stride(1),
        k_compressed.stride(3),
        k_compressed.stride(2),  # FIXED
        v_compressed.stride(0),
        v_compressed.stride(1),
        v_compressed.stride(2),
        v_compressed.stride(3),
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
        N_CTX_Q=T,
        N_BLOCKS=N_BLOCKS,
        HEAD_DIM_QK=dk,
        HEAD_DIM_V=dv,
        BLOCK_M=32,
        BLOCK_N=32,
    )

    dk_kernel_transposed = dk_kernel.transpose(-2, -1)

    # Check errors
    forward_err = (out_kernel - out_ref).norm() / out_ref.norm()
    dq_err = (dq_kernel - dq_ref).norm() / dq_ref.norm()
    dk_err = (dk_kernel_transposed - dk_ref).norm() / dk_ref.norm()
    dv_err = (dv_kernel - dv_ref).norm() / dv_ref.norm()

    print("\nResults:")
    print(f"  Forward error: {forward_err:.6f}")
    print(f"  dQ error: {dq_err:.6f}")
    print(f"  dK error: {dk_err:.6f}")
    print(f"  dV error: {dv_err:.6f}")

    threshold = 0.001
    passed = (
        forward_err < threshold
        and dq_err < threshold
        and dk_err < threshold
        and dv_err < threshold
    )

    if passed:
        print(f"  ✅ PASSED (all errors < {threshold})")
    else:
        print(f"  ❌ FAILED (some errors >= {threshold})")

    return passed


def test_selection_branch():
    """Test sparse selection attention branch (forward + backward parity)."""
    print("\n" + "=" * 80)
    print("SPARSE SELECTION BRANCH TEST")
    print("=" * 80)

    from nsa import _nsa_sparse_selection_fwd_kernel, _nsa_sparse_selection_bwd_kernel

    B, H, G = 2, 8, 2
    T = 64
    dk = dv = 32
    l_prime = 32  # Selection block size (must be >= BLOCK_N=32)
    n = 4  # Blocks per query
    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(dk)

    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype, requires_grad=True)

    # Build per-group block indices [B, G, T, n]
    n_blocks_kv = (T + l_prime - 1) // l_prime
    block_indices = torch.empty(B, G, T, n, device=device, dtype=torch.int32)
    for b in range(B):
        for g in range(G):
            for t in range(T):
                local = t // l_prime
                cand = [0, max(0, local - 1), local, min(local + 1, n_blocks_kv - 1)]
                # Deduplicate and pad
                uniq = []
                seen = set()
                for idx in cand:
                    if idx not in seen:
                        seen.add(idx)
                        uniq.append(idx)
                while len(uniq) < n:
                    uniq.append(-1)  # Pad with -1 sentinel instead of duplicating
                block_indices[b, g, t] = torch.tensor(
                    uniq[:n], device=device, dtype=torch.int32
                )

    # Dense masked reference (GQA expanded)
    heads_per_group = H // G
    k_exp = k.unsqueeze(2).expand(B, G, heads_per_group, dk, T).reshape(B, H, dk, T)
    v_exp = v.unsqueeze(2).expand(B, G, heads_per_group, T, dv).reshape(B, H, T, dv)

    mask = torch.zeros(B, H, T, T, device=device, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            g = h // heads_per_group
            for t in range(T):
                for nb in range(n):
                    blk = int(block_indices[b, g, t, nb].item())
                    if blk >= 0:  # Skip -1 sentinel values
                        start = blk * l_prime
                        end = min(start + l_prime, T)
                        mask[b, h, t, start:end] = True
    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
    mask = mask & causal

    scores = torch.matmul(q * scale, k_exp)  # [B,H,T,T]
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(attn, v_exp)  # [B,H,T,dv]

    grad_out = torch.randn_like(out_ref) * 0.01
    out_ref.backward(grad_out)
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    # Reset grads for kernel path
    q.grad = None
    k.grad = None
    v.grad = None

    # Kernel forward
    out_kernel = torch.zeros_like(out_ref)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

    _nsa_sparse_selection_fwd_kernel[grid](
        q, k, v, block_indices, out_kernel, scale, L, M,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # SWAPPED
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_kernel.stride(0), out_kernel.stride(1), out_kernel.stride(2), out_kernel.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Z=B, H=H, N_KV_GROUPS=G, N_CTX_Q=T, N_CTX_KV=T,
        N_BLOCKS=n, BLOCK_SIZE_SELECTION=l_prime,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32, IS_CAUSAL=True,
    )

    # Kernel backward
    dq_kernel = torch.zeros_like(q)
    dk_kernel = torch.zeros(B, G, T, dk, device=device, dtype=dtype)  # [B,G,T,dk]
    dv_kernel = torch.zeros_like(v)

    _nsa_sparse_selection_bwd_kernel[grid](
        # Forward tensors
        q, k, v, block_indices, out_kernel, grad_out,
        scale, L, M,
        # Strides - Q/K/V/Out
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),  # SWAPPED
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_kernel.stride(0), out_kernel.stride(1), out_kernel.stride(2), out_kernel.stride(3),
        # Strides - Block indices
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        # Strides - L/M
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        # Gradients (outputs)
        dq_kernel, dk_kernel, dv_kernel,
        dq_kernel.stride(0), dq_kernel.stride(1), dq_kernel.stride(2), dq_kernel.stride(3),
        dk_kernel.stride(0), dk_kernel.stride(1), dk_kernel.stride(2), dk_kernel.stride(3),
        dv_kernel.stride(0), dv_kernel.stride(1), dv_kernel.stride(2), dv_kernel.stride(3),
        # Shapes/params
        Z=B, H=H, N_KV_GROUPS=G, N_CTX_Q=T, N_CTX_KV=T,
        N_BLOCKS=n, BLOCK_SIZE_SELECTION=l_prime,
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        BLOCK_M=32, BLOCK_N=32, IS_CAUSAL=True,
    )

    dk_kernel_T = dk_kernel.transpose(-2, -1)  # [B,G,dk,T] to compare

    # Errors
    fwd_err = (out_kernel - out_ref).norm() / (out_ref.norm() + 1e-8)
    dq_err = (dq_kernel - dq_ref).norm() / (dq_ref.norm() + 1e-8)
    dk_err = (dk_kernel_T - dk_ref).norm() / (dk_ref.norm() + 1e-8)
    dv_err = (dv_kernel - dv_ref).norm() / (dv_ref.norm() + 1e-8)

    print("\nResults:")
    print(f"  Forward error: {fwd_err:.6f}")
    print(f"  dQ error:      {dq_err:.6f}")
    print(f"  dK error:      {dk_err:.6f}")
    print(f"  dV error:      {dv_err:.6f}")
    
    thr = 1e-3
    assert fwd_err < thr and dq_err < thr and dk_err < thr and dv_err < thr, (
        "Selection branch parity failed"
    )
    print(f"  ✅ PASSED (all errors < {thr})")
    return True


def main():
    """Run all branch tests."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE NSA BRANCH TESTING")
    print("Testing all three NSA branches with production accuracy requirements")
    print("=" * 80)

    results = {}

    # Test each branch
    results["sliding_window"] = test_sliding_window_branch()
    results["compression"] = test_compression_branch()
    results["selection"] = test_selection_branch()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = True
    for branch, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {branch:20s}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL BRANCHES PASSED!")
        print("NSA implementation meets production requirements.")
    else:
        print("❌ SOME BRANCHES FAILED")
        print("Further debugging needed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
