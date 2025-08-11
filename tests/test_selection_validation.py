#!/usr/bin/env python3
"""
Production-grade validation tests for the selection branch.
Tests forward pass and all gradients against masked dense reference.
"""

import torch
import torch.nn.functional as F
import triton
import math

from nsa import _nsa_sparse_selection_fwd_kernel, _nsa_sparse_selection_bwd_kernel


def create_selection_reference(
    q: torch.Tensor,  # [B, H, T, dk]
    k: torch.Tensor,  # [B, G, dk, T]
    v: torch.Tensor,  # [B, G, T, dv]
    indices: torch.Tensor,  # [B, G, T, n]
    scale: float,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Reference implementation using masked dense attention.
    """
    B, H, T, dk = q.shape
    B, G, dk, T_kv = k.shape
    dv = v.shape[-1]
    n = indices.shape[-1]
    
    heads_per_group = H // G
    
    # Output tensor
    output = torch.zeros(B, H, T, dv, device=q.device, dtype=q.dtype)
    
    # Process each batch and head
    for b in range(B):
        for h in range(H):
            g = h // heads_per_group
            
            # Get Q for this head
            q_h = q[b, h]  # [T, dk]
            
            # Get K, V for this group
            k_g = k[b, g]  # [dk, T]
            v_g = v[b, g]  # [T, dv]
            
            # Compute full attention scores
            scores = torch.matmul(q_h, k_g) * scale  # [T, T]
            
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(T, T_kv, device=q.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.bool(), float('-inf'))
            
            # Apply selection mask
            for t in range(T):
                # Get selected blocks for this query
                selected_blocks = indices[b, g, t]  # [n]
                
                # Create mask for selected positions
                mask = torch.ones(T_kv, device=q.device, dtype=torch.bool)
                has_valid_block = False
                for block_idx in selected_blocks:
                    if block_idx >= 0:  # Valid block index
                        start = block_idx * block_size
                        end = min(start + block_size, T_kv)
                        if start < T_kv:
                            mask[start:end] = False
                            has_valid_block = True
                
                # Apply selection mask (set non-selected to -inf)
                # But ensure at least one position is not masked to avoid NaN
                if not has_valid_block and t < T_kv:
                    # If no valid blocks, keep position t unmasked (self-attention)
                    mask[min(t, T_kv-1)] = False
                
                scores[t].masked_fill_(mask, float('-inf'))
            
            # Compute attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply attention to values
            output[b, h] = torch.matmul(attn_weights, v_g)
    
    return output


def test_selection_forward():
    """Test selection forward pass against reference."""
    print("\n" + "="*80)
    print("SELECTION FORWARD TEST")
    print("="*80)
    
    # Test configuration
    B, H, G, T = 2, 8, 2, 256
    dk, dv = 64, 64
    block_size = 64
    n_blocks = 4
    
    device = 'cuda'
    dtype = torch.float32
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(B, H, T, dk, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype, requires_grad=True)
    
    # Create block indices (select different blocks for each query)
    indices = torch.full((B, G, T, n_blocks), -1, device=device, dtype=torch.int32)
    for b in range(B):
        for g in range(G):
            for t in range(T):
                # For causal attention, we can only attend to blocks up to current position
                max_valid_block = t // block_size  # Block containing position t
                if max_valid_block >= 0:
                    # Number of blocks we can select (capped by n_blocks)
                    n_valid = min(n_blocks, max_valid_block + 1)
                    selected = torch.randperm(max_valid_block + 1, device=device)[:n_valid]
                    indices[b, g, t, :n_valid] = selected
                elif t >= 0:
                    # For very early positions, at least include block 0 (partial)
                    indices[b, g, t, 0] = 0
    
    scale = 1.0 / math.sqrt(dk)
    
    # Run reference implementation
    ref_output = create_selection_reference(q, k, v, indices, scale, block_size)
    
    # Run kernel
    output = torch.zeros_like(ref_output)
    L = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    M = torch.zeros(B, H, T, device=device, dtype=torch.float32)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    
    _nsa_sparse_selection_fwd_kernel[grid](
        q, k, v, indices, output,
        scale, L, M,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides (swapped for kernel)
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Indices strides
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        # L/M strides
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        # Shape parameters
        Z=B, H=H, N_KV_GROUPS=G,
        N_CTX_Q=T, N_CTX_KV=T,
        N_BLOCKS=n_blocks,
        BLOCK_SIZE_SELECTION=block_size,
        IS_CAUSAL=True,
        # Head dimensions
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        # Block sizes
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Compare outputs
    diff = (output - ref_output).abs()
    rel_error = diff / (ref_output.abs() + 1e-5)
    
    max_diff = diff.max().item()
    max_rel_error = rel_error.max().item()
    
    print(f"\n[RESULTS] Forward Pass:")
    print(f"  Max absolute error: {max_diff:.6f}")
    print(f"  Max relative error: {max_rel_error:.6f}")
    
    assert max_rel_error < 0.001, f"Forward relative error {max_rel_error} exceeds threshold"
    print("✅ Selection forward pass matches reference!")
    
    return q, k, v, indices, output, ref_output, scale, L, M


def test_selection_backward():
    """Test selection backward pass gradients."""
    print("\n" + "="*80)
    print("SELECTION BACKWARD TEST")
    print("="*80)
    
    # Run forward first to get tensors
    q, k, v, indices, output, ref_output, scale, L, M = test_selection_forward()
    
    B, H, T, dv = output.shape
    G = k.shape[1]
    dk = q.shape[-1]
    block_size = 64
    n_blocks = indices.shape[-1]
    
    device = q.device
    dtype = q.dtype
    
    # Create gradient for output
    do = torch.randn_like(output)
    
    # Compute reference gradients
    ref_output.backward(do)
    ref_dq = q.grad.clone()
    ref_dk = k.grad.clone()
    ref_dv = v.grad.clone()
    
    # Clear gradients
    q.grad = None
    k.grad = None
    v.grad = None
    
    # Allocate gradient tensors for kernel
    dq = torch.zeros_like(q)
    dk_kernel = torch.zeros(B, G, T, dk, device=device, dtype=dtype)  # Note: [B,G,T,dk] for kernel
    dv = torch.zeros_like(v)
    
    grid = lambda META: (triton.cdiv(T, META['BLOCK_M']), B * H)
    
    _nsa_sparse_selection_bwd_kernel[grid](
        q, k, v, indices,
        output,    # Out (from forward)
        do,        # dOut
        scale, L, M,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides (swapped)
        k.stride(0), k.stride(1), k.stride(3), k.stride(2),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Indices strides
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        # L/M strides
        L.stride(0), L.stride(1), L.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        # Gradient outputs
        dq, dk_kernel, dv,
        # DQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # DK strides (kernel expects [B,G,T,dk])
        dk_kernel.stride(0), dk_kernel.stride(1), dk_kernel.stride(2), dk_kernel.stride(3),
        # DV strides
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        # Shape parameters
        Z=B, H=H, N_KV_GROUPS=G,
        N_CTX_Q=T, N_CTX_KV=T,
        N_BLOCKS=n_blocks,
        BLOCK_SIZE_SELECTION=block_size,
        IS_CAUSAL=True,
        # Head dimensions
        HEAD_DIM_QK=dk, HEAD_DIM_V=dv,
        # Block sizes
        BLOCK_M=32, BLOCK_N=32,
    )
    
    # Transpose dk back to match reference shape
    dk = dk_kernel.transpose(-1, -2)  # [B,G,T,dk] -> [B,G,dk,T]
    
    # Compare gradients
    print("\n[RESULTS] Backward Pass:")
    
    # dQ gradient
    diff_q = (dq - ref_dq).abs()
    rel_error_q = diff_q / (ref_dq.abs() + 1e-5)
    print(f"  dQ max relative error: {rel_error_q.max().item():.6f}")
    
    # dK gradient
    diff_k = (dk - ref_dk).abs()
    rel_error_k = diff_k / (ref_dk.abs() + 1e-5)
    print(f"  dK max relative error: {rel_error_k.max().item():.6f}")
    
    # dV gradient
    diff_v = (dv - ref_dv).abs()
    rel_error_v = diff_v / (ref_dv.abs() + 1e-5)
    print(f"  dV max relative error: {rel_error_v.max().item():.6f}")
    
    assert rel_error_q.max().item() < 0.001, "dQ gradient error exceeds threshold"
    assert rel_error_k.max().item() < 0.001, "dK gradient error exceeds threshold"
    assert rel_error_v.max().item() < 0.001, "dV gradient error exceeds threshold"
    
    print("✅ All selection gradients match reference!")


if __name__ == "__main__":
    test_selection_backward()
    print("\n" + "="*80)
    print("✅ ALL SELECTION VALIDATION TESTS PASSED!")
    print("="*80)