"""
PyTorch reference implementations for NSA.
These are extracted from our working test files and verified to match Triton kernels.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


def sliding_window_attention(
    q: torch.Tensor,  # [B, H, T, dk]
    k: torch.Tensor,  # [B, G, dk, T] - NOTE THE SHAPE!
    v: torch.Tensor,  # [B, G, T, dv]
    window_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation for sliding window attention.
    This exact implementation is used in test_accuracy.py and achieves 0% error.

    Args:
        q: [B, H, T, dk] query tensor
        k: [B, G, dk, T] key tensor - CRITICAL: dk before T!
        v: [B, G, T, dv] value tensor
        window_size: size of sliding window
        scale: scaling factor (default: 1/sqrt(dk))

    Returns:
        output: [B, H, T, dv]
    """
    B, H, T, dk = q.shape
    _, G, _, _ = k.shape
    dv = v.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    # Expand K/V for all heads (GQA)
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
    return out_ref


def compression_attention(
    q: torch.Tensor,  # [B, H, T, dk]
    k_compressed: torch.Tensor,  # [B, G, dk, N_BLOCKS]
    v_compressed: torch.Tensor,  # [B, G, N_BLOCKS, dv]
    block_ends: torch.Tensor,  # [N_BLOCKS]
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation for compression attention.
    This exact implementation is used in test_branches.py and achieves 0% error.

    Args:
        q: [B, H, T, dk] query tensor
        k_compressed: [B, G, dk, N_BLOCKS] compressed keys
        v_compressed: [B, G, N_BLOCKS, dv] compressed values
        block_ends: [N_BLOCKS] tensor of block end positions
        scale: scaling factor

    Returns:
        output: [B, H, T, dv]
    """
    B, H, T, dk = q.shape
    _, G, _, N_BLOCKS = k_compressed.shape
    dv = v_compressed.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    # Expand K/V for all heads (GQA)
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
                    out_t = torch.zeros(1, 1, 1, dv, device=q.device, dtype=q.dtype)

                out_h.append(out_t.squeeze())
            outputs.append(torch.stack(out_h))

    out_ref = torch.stack([torch.stack(outputs[b * H : (b + 1) * H]) for b in range(B)])
    return out_ref


def dense_attention_reference(
    q: torch.Tensor,  # [B, H, T, dk]
    k: torch.Tensor,  # [B, G, dk, T] for our kernels, or [B, H, T, dk] for standard
    v: torch.Tensor,  # [B, G, T, dv] for our kernels, or [B, H, T, dv] for standard
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dense attention reference from test_backward.py.
    Handles both GQA (G != H) and MHA (G == H) cases.

    Returns:
        output: [B, H, T, dv]
        probs: [B, H, T, T] attention weights
    """
    B, H, T, dk = q.shape

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    # Handle different K/V shapes
    if k.shape[1] != H:  # GQA case: k is [B, G, dk, T]
        G = k.shape[1]
        dv = v.shape[-1]
        heads_per_group = H // G

        # Expand K/V for all heads
        k_expanded = (
            k.unsqueeze(2).expand(B, G, heads_per_group, dk, T).reshape(B, H, dk, T)
        )
        v_expanded = (
            v.unsqueeze(2).expand(B, G, heads_per_group, T, dv).reshape(B, H, T, dv)
        )

        # Transpose K to [B, H, T, dk]
        k_for_matmul = k_expanded.transpose(-2, -1)
    else:  # MHA case: k is [B, H, T, dk]
        k_for_matmul = k
        v_expanded = v

    # Q @ K^T
    scores = torch.matmul(q * scale, k_for_matmul.transpose(-1, -2))  # [B, H, T, T]

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax
    probs = F.softmax(scores, dim=-1)

    # Attention output
    out = torch.matmul(probs, v_expanded)  # [B, H, T, dv]

    return out, probs


def selection_attention_reference(
    q: torch.Tensor,  # [B, H, T, dk]
    k: torch.Tensor,  # [B, G, dk, T]
    v: torch.Tensor,  # [B, G, T, dv]
    selected_indices: torch.Tensor,  # [B, G, T, n]
    l_prime: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation for sparse selection attention (paper Eq. 11).
    - Expands K/V to heads (GQA) and constructs a per-(b,h,t) KV mask from the
      selected block indices of its group, unioned across blocks, with causal masking.
    """
    B, H, T, dk = q.shape
    G = k.shape[1]
    T_kv = k.shape[-1]
    dv = v.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(dk)

    # Expand K/V to heads
    heads_per_group = H // G
    k_expanded = k.unsqueeze(2).expand(B, G, heads_per_group, dk, T_kv).reshape(B, H, dk, T_kv)
    v_expanded = v.unsqueeze(2).expand(B, G, heads_per_group, T_kv, dv).reshape(B, H, T_kv, dv)

    # Build selection mask [B, H, T, T_kv]
    mask = torch.zeros(B, H, T, T_kv, device=q.device, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            g = h // (H // G)
            for t in range(T):
                blocks = selected_indices[b, g, t]
                for idx in blocks.tolist():
                    if idx < 0:
                        continue
                    start = idx * l_prime
                    end = min(start + l_prime, T_kv)
                    if start < end:
                        mask[b, h, t, start:end] = True
                # causal
                mask[b, h, t, t + 1 :] = False

    # Compute scores and apply mask
    # q @ K -> [B, H, T, T_kv]
    scores = torch.matmul(q * scale, k_expanded)  # K is [B,H,dk,T_kv]
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_expanded)  # [B,H,T,dv]
    return out


def create_block_ends(
    T: int, block_size: int, stride: int, device="cuda"
) -> torch.Tensor:
    """
    Create block end positions for compression attention.

    Args:
        T: sequence length
        block_size: size of each block (l in paper)
        stride: stride between blocks (d in paper)
        device: device to create tensor on

    Returns:
        block_ends: [N_BLOCKS] tensor
    """
    block_ends = []
    for start in range(0, T, stride):
        end = min(start + block_size, T) - 1
        block_ends.append(end)
        if end >= T - 1:
            break
    return torch.tensor(block_ends, device=device, dtype=torch.int32)


def compress_kv_simple(
    k: torch.Tensor,  # [B, G, dk, T]
    v: torch.Tensor,  # [B, G, T, dv]
    block_size: int,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple K/V compression using average pooling.

    Returns:
        k_compressed: [B, G, dk, N_BLOCKS]
        v_compressed: [B, G, N_BLOCKS, dv]
        block_ends: [N_BLOCKS]
    """
    B, G, dk, T = k.shape
    dv = v.shape[-1]

    k_blocks = []
    v_blocks = []
    block_ends = []

    for start in range(0, T, stride):
        end = min(start + block_size, T)
        block_ends.append(end - 1)

        # Average pool the block
        k_block = k[:, :, :, start:end].mean(dim=-1)  # [B, G, dk]
        v_block = v[:, :, start:end, :].mean(dim=2)  # [B, G, dv]

        k_blocks.append(k_block)
        v_blocks.append(v_block)

        if end >= T:
            break

    k_compressed = torch.stack(k_blocks, dim=-1)  # [B, G, dk, N_BLOCKS]
    v_compressed = torch.stack(v_blocks, dim=2)  # [B, G, N_BLOCKS, dv]
    
    # Build [n_blocks] shaped block_ends tensor (same for all batches/groups)
    block_ends = torch.tensor(block_ends, device=k.device, dtype=torch.int32)  # [n_blocks]

    return k_compressed, v_compressed, block_ends


# Utility functions for testing


def compare_outputs(
    output_kernel: torch.Tensor,
    output_ref: torch.Tensor,
    tolerance: float = 1e-3,
    name: str = "Output",
) -> Tuple[bool, float]:
    """
    Compare kernel output with reference implementation.

    Returns:
        passed: True if within tolerance
        rel_error: relative error
    """
    diff = (output_kernel - output_ref).abs()
    rel_error = diff.max() / (output_ref.abs().max() + 1e-8)

    passed = rel_error < tolerance

    print(f"{name} relative error: {rel_error:.6f}")
    if not passed:
        print(f"  Max absolute error: {diff.max():.6f}")
        max_idx = diff.argmax()
        print(f"  Kernel value at max error: {output_kernel.flatten()[max_idx]:.6f}")
        print(f"  Reference value at max error: {output_ref.flatten()[max_idx]:.6f}")

    return passed, rel_error


def generate_test_tensors(
    B: int = 2,
    H: int = 8,
    G: int = 2,
    T: int = 64,
    dk: int = 32,
    dv: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test tensors with correct shapes for NSA.

    CRITICAL: Returns K with shape [B, G, dk, T]!

    Returns:
        q: [B, H, T, dk]
        k: [B, G, dk, T]  # Note: dk before T!
        v: [B, G, T, dv]
    """
    torch.manual_seed(seed)

    q = torch.randn(B, H, T, dk, device=device, dtype=dtype) * 0.02
    k = torch.randn(B, G, dk, T, device=device, dtype=dtype) * 0.02  # dk before T!
    v = torch.randn(B, G, T, dv, device=device, dtype=dtype) * 0.02

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return q, k, v
