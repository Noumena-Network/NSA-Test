"""
Native Sparse Attention (NSA) - Production-Grade Triton Kernels V7

FINAL PRODUCTION VERSION addressing all review feedback:
- Per-timestep selection derivation with proper [B,H,T,n_comp_blocks] shape
- Correct GQA aggregation and per-query top-k selection
- Always-include blocks (1 initial + 2 local) with deduplication
- Complete forward path wiring
- Proper d_model → heads/groups projection
- Full compress_tokens implementation with MLP pooling

Reference: arXiv:2502.11089v1 "Native Sparse Attention"
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Minimal autograd wrappers for Triton kernels
# Following FlashAttention's pattern - thin wrappers that just connect kernels to autograd
class CompressionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k_compressed, v_compressed, block_ends, scale, config):
        B, H, T, dk = q.shape
        dv = v_compressed.shape[-1]
        device = q.device
        dtype = q.dtype

        # Allocate output
        o = torch.empty(B, H, T, dv, device=device, dtype=dtype)
        L = torch.empty(B, H, T, device=device, dtype=torch.float32)
        M = torch.empty(B, H, T, device=device, dtype=torch.float32)

        # Call forward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_compression_fwd_kernel[grid](
            q,
            k_compressed,
            v_compressed,
            block_ends,
            o,
            scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_compressed.stride(0),
            k_compressed.stride(1),
            k_compressed.stride(2),
            k_compressed.stride(3),
            v_compressed.stride(0),
            v_compressed.stride(1),
            v_compressed.stride(2),
            v_compressed.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            Z=B,
            H=H,
            N_KV_GROUPS=k_compressed.shape[1],
            N_CTX_Q=T,
            N_BLOCKS=k_compressed.shape[-1],
            HEAD_DIM_QK=triton.next_power_of_2(dk),
            HEAD_DIM_V=triton.next_power_of_2(dv),
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
        )

        ctx.save_for_backward(q, k_compressed, v_compressed, block_ends, o, L, M)
        ctx.scale = scale
        ctx.config = config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k_compressed, v_compressed, block_ends, o, L, M = ctx.saved_tensors
        B, H, T, dk = q.shape
        G = k_compressed.shape[1]
        dv = v_compressed.shape[-1]

        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros(
            B, G, k_compressed.shape[-1], dk, device=q.device, dtype=q.dtype
        )
        dv = torch.zeros_like(v_compressed)

        # Call backward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_compression_bwd_kernel[grid](
            q,
            k_compressed,
            v_compressed,
            block_ends,
            o,
            do,
            ctx.scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_compressed.stride(0),
            k_compressed.stride(1),
            k_compressed.stride(2),
            k_compressed.stride(3),
            v_compressed.stride(0),
            v_compressed.stride(1),
            v_compressed.stride(2),
            v_compressed.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            dq,
            dk,
            dv,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            Z=B,
            H=H,
            N_KV_GROUPS=G,
            N_CTX_Q=T,
            N_BLOCKS=k_compressed.shape[-1],
            HEAD_DIM_QK=triton.next_power_of_2(q.shape[-1]),
            HEAD_DIM_V=triton.next_power_of_2(v_compressed.shape[-1]),
            BLOCK_M=ctx.config.block_m,
            BLOCK_N=ctx.config.block_n,
        )

        dk = dk.transpose(-1, -2)  # [B,G,n_blocks,dk] -> [B,G,dk,n_blocks]
        return dq, dk, dv, None, None, None


class SelectionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, scale, config):
        B, H, T, dk = q.shape
        dv = v.shape[-1]
        device = q.device
        dtype = q.dtype

        # Allocate output
        o = torch.empty(B, H, T, dv, device=device, dtype=dtype)
        L = torch.empty(B, H, T, device=device, dtype=torch.float32)
        M = torch.empty(B, H, T, device=device, dtype=torch.float32)

        # Call forward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_sparse_selection_fwd_kernel[grid](
            q,
            k,
            v,
            indices.to(torch.int32),
            o,
            scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            indices.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            Z=B,
            H=H,
            N_KV_GROUPS=k.shape[1],
            N_CTX_Q=T,
            N_CTX_KV=k.shape[-1],
            N_BLOCKS=indices.shape[-1],
            BLOCK_SIZE_SELECTION=config.l_prime,
            IS_CAUSAL=True,
            HEAD_DIM_QK=triton.next_power_of_2(dk),
            HEAD_DIM_V=triton.next_power_of_2(dv),
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
        )

        ctx.save_for_backward(q, k, v, indices, o, L, M)
        ctx.scale = scale
        ctx.config = config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, indices, o, L, M = ctx.saved_tensors
        B, H, T, dk = q.shape
        G = k.shape[1]
        T_kv = k.shape[-1]
        dv = v.shape[-1]

        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros(B, G, T_kv, dk, device=q.device, dtype=q.dtype)
        dv = torch.zeros_like(v)

        # Call backward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_sparse_selection_bwd_kernel[grid](
            q,
            k,
            v,
            indices,
            o,
            do,
            ctx.scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            indices.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            dq,
            dk,
            dv,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            Z=B,
            H=H,
            N_KV_GROUPS=G,
            N_CTX_Q=T,
            N_CTX_KV=T_kv,
            N_BLOCKS=indices.shape[-1],
            BLOCK_SIZE_SELECTION=ctx.config.l_prime,
            IS_CAUSAL=True,
            HEAD_DIM_QK=triton.next_power_of_2(q.shape[-1]),
            HEAD_DIM_V=triton.next_power_of_2(v.shape[-1]),
            BLOCK_M=ctx.config.block_m,
            BLOCK_N=ctx.config.block_n,
        )

        dk = dk.transpose(-1, -2)  # [B,G,T,dk] -> [B,G,dk,T]
        return dq, dk, dv, None, None, None


class SlidingWindowAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, window_size, config):
        B, H, T, dk = q.shape
        dv = v.shape[-1]
        device = q.device
        dtype = q.dtype

        # Allocate output
        o = torch.empty(B, H, T, dv, device=device, dtype=dtype)
        L = torch.empty(B, H, T, device=device, dtype=torch.float32)
        M = torch.empty(B, H, T, device=device, dtype=torch.float32)

        # Call forward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_sliding_window_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            Z=B,
            H=H,
            N_KV_GROUPS=k.shape[1],
            N_CTX=T,
            WINDOW_SIZE=window_size,
            HEAD_DIM_QK=triton.next_power_of_2(dk),
            HEAD_DIM_V=triton.next_power_of_2(dv),
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
        )

        ctx.save_for_backward(q, k, v, o, L, M)
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.config = config
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, M = ctx.saved_tensors
        B, H, T, dk = q.shape
        G = k.shape[1]
        dv = v.shape[-1]

        # Allocate gradients
        dq = torch.zeros_like(q)
        dk = torch.zeros(B, G, T, dk, device=q.device, dtype=q.dtype)
        dv = torch.zeros_like(v)

        # Call backward kernel
        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _nsa_sliding_window_bwd_kernel[grid](
            q,
            k,
            v,
            o,
            do,
            ctx.scale,
            L,
            M,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            L.stride(0),
            L.stride(1),
            L.stride(2),
            M.stride(0),
            M.stride(1),
            M.stride(2),
            dq,
            dk,
            dv,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            Z=B,
            H=H,
            N_KV_GROUPS=G,
            N_CTX=T,
            WINDOW_SIZE=ctx.window_size,
            HEAD_DIM_QK=triton.next_power_of_2(q.shape[-1]),
            HEAD_DIM_V=triton.next_power_of_2(v.shape[-1]),
            BLOCK_M=ctx.config.block_m,
            BLOCK_N=ctx.config.block_n,
        )

        dk = dk.transpose(-1, -2)  # [B,G,T,dk] -> [B,G,dk,T]
        return dq, dk, dv, None, None, None


@dataclass
class NSAConfig:
    """Production NSA configuration following paper Section 4.1."""

    # Model dimensions
    d_model: int = 2560  # Model dimension (input/output)
    head_dim_qk: int = 192  # d_k from paper (will be padded for Triton)
    head_dim_v: int = 128  # d_v from paper

    # Compression parameters (Section 3.3.1)
    l: int = 32  # Compression block size
    d: int = 16  # Compression stride (d < l for overlap)

    # Selection parameters (Section 3.3.2)
    l_prime: int = 64  # Selection block size
    n: int = 16  # Total blocks to select per query
    n_dynamic: int = 13  # Dynamic blocks (n - n_fixed)
    n_fixed: int = 3  # Fixed blocks: 1 initial + 2 local

    # Sliding window (Section 3.3.3)
    w: int = 512  # Window size

    # GQA/MQA parameters
    n_heads: int = 64  # Total query heads
    n_kv_groups: int = 4  # Number of KV groups

    # Training configuration
    dropout_p: float = 0.0
    use_bias: bool = False

    # Kernel optimization (must be powers of 2)
    # B200 has 232KB shared memory, can handle 64x64 blocks
    # Production configuration for B200 GPUs
    block_m: int = 32  # Query tile size (reduced for B200 shared memory limits)
    block_n: int = 32  # KV tile size (reduced for B200 shared memory limits)

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_heads % self.n_kv_groups == 0
        assert self.l % self.d == 0
        assert self.l_prime % self.d == 0
        assert self.n_fixed + self.n_dynamic == self.n
        assert self.w > 0

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_groups

    @property
    def head_dim_qk_padded(self) -> int:
        """Get padded QK dimension for Triton kernels (next power of 2)."""
        dim = self.head_dim_qk
        power = 1
        while power < dim:
            power *= 2
        return power

    @property
    def head_dim_v_padded(self) -> int:
        """Get padded V dimension for Triton kernels (next power of 2)."""
        dim = self.head_dim_v
        power = 1
        while power < dim:
            power *= 2
        return power


@triton.jit
def _nsa_sparse_selection_fwd_kernel(
    # Tensors
    Q,
    K,
    V,
    Block_indices,  # [B, G, T, n] - per-group block indices
    Out,
    sm_scale,
    L,
    M,  # Logsumexp statistics
    # Strides - Q
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    # Strides - K (with group dimension)
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    # Strides - V (with group dimension)
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    # Strides - Output
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    # Strides - Block indices (with group dimension)
    stride_bz,
    stride_bg,
    stride_bm,
    stride_bn,
    # Strides - Logsumexp
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    # Shape parameters
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_KV: tl.constexpr,
    # NSA parameters
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE_SELECTION: tl.constexpr,
    # Model dimensions
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    # Kernel tiling
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # Flags
    IS_CAUSAL: tl.constexpr,
):
    """Production sparse selection kernel with all fixes applied."""
    # Program IDs
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    # Offsets
    q_offset = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_offset = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_offset = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_offset = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    # Create block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # Load queries
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = q * sm_scale

    # Initialize accumulators
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Start at 0, not 1e-10
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    # Process selected blocks
    for block_idx in range(N_BLOCKS):
        # Load per-group per-query block indices
        block_idx_ptrs = (
            Block_indices
            + pid_z * stride_bz
            + kv_group * stride_bg  # Index by KV group
            + offs_m[:, None] * stride_bm
            + block_idx * stride_bn
        )

        valid_m = offs_m < N_CTX_Q
        # Load block indices for this query position
        # Since we're loading with offs_m[:, None], shape is [BLOCK_M, 1]
        # We need to flatten to [BLOCK_M]
        selected_blocks = tl.load(block_idx_ptrs, mask=valid_m[:, None], other=-1)
        # Reshape from [BLOCK_M, 1] to [BLOCK_M]
        selected_blocks = tl.reshape(selected_blocks, [BLOCK_M])

        # Find unique blocks (skip -1 sentinel values)
        blocks_valid = selected_blocks >= 0
        has_valid = tl.sum(blocks_valid.to(tl.int32)) > 0

        # Only process if we have valid blocks
        if has_valid:
            min_block = tl.min(tl.where(blocks_valid, selected_blocks, 999999))
            max_block = tl.max(tl.where(blocks_valid, selected_blocks, -1))

            # Process each unique block
            for unique_block in range(min_block, max_block + 1):
                # Early skip
                queries_use_block = selected_blocks == unique_block
                n_queries_using = tl.sum(queries_use_block.to(tl.int32))

                # Only process if queries use this block
                if n_queries_using > 0:
                    # Block boundaries - ensure int32 for block pointers
                    block_start = tl.full((), unique_block * BLOCK_SIZE_SELECTION, tl.int32)
                    block_end = tl.minimum(
                        block_start + tl.full((), BLOCK_SIZE_SELECTION, tl.int32),
                        tl.full((), N_CTX_KV, tl.int32)
                    )

                    # Per-row masks
                    block_start_row = tl.where(queries_use_block, block_start, N_CTX_KV)
                    block_end_row = tl.where(queries_use_block, block_end, 0)

                    # Process in tiles
                    for kv_tile_offset in range(0, BLOCK_SIZE_SELECTION, BLOCK_N):
                        tile_start = block_start + tl.full((), kv_tile_offset, tl.int32)

                        # Only process if tile is within block
                        if tile_start < block_end:
                            # KV pointers
                            K_block_ptr = tl.make_block_ptr(
                                base=K + k_offset,
                                shape=(HEAD_DIM_QK, N_CTX_KV),
                                strides=(stride_kk, stride_kn),
                                offsets=(0, tile_start),
                                block_shape=(HEAD_DIM_QK, BLOCK_N),
                                order=(0, 1),
                            )

                            V_block_ptr = tl.make_block_ptr(
                                base=V + v_offset,
                                shape=(N_CTX_KV, HEAD_DIM_V),
                                strides=(stride_vn, stride_vk),
                                offsets=(tile_start, 0),
                                block_shape=(BLOCK_N, HEAD_DIM_V),
                                order=(1, 0),
                            )

                            # Load KV
                            k = tl.load(K_block_ptr, boundary_check=(0, 1))
                            v = tl.load(V_block_ptr, boundary_check=(0, 1))

                            # Compute QK^T
                            qk = tl.dot(q, k, allow_tf32=False)

                            # Apply masks
                            offs_n = tile_start + tl.arange(0, BLOCK_N)

                            # Fixed: Use explicit tensor mask for non-causal
                            if IS_CAUSAL:
                                causal_mask = offs_n[None, :] <= offs_m[:, None]
                            else:
                                causal_mask = (
                                    offs_m[:, None] >= -1
                                )  # Always true, but tensor type

                            boundary_mask = (
                                offs_n[None, :] >= block_start_row[:, None]
                            ) & (offs_n[None, :] < block_end_row[:, None])

                            valid_mask = (
                                causal_mask
                                & boundary_mask
                                & (offs_m[:, None] < N_CTX_Q)
                            )
                            qk = tl.where(valid_mask, qk, float("-inf"))

                            # Online softmax with NaN protection
                            # Check if each row has any valid values
                            valid_rows = tl.sum(valid_mask.to(tl.int32), axis=1) > 0
                            
                            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                            # Masked shift to avoid -inf - (-inf) = NaN
                            qk_shift = tl.where(valid_mask, qk - m_ij[:, None], float("-inf"))
                            p = tl.math.exp(qk_shift)
                            l_ij = tl.sum(p, axis=1)

                            # Conditional alpha to avoid exp(NaN)
                            alpha = tl.where(valid_rows, tl.math.exp(m_i - m_ij), 1.0)
                            acc = acc * alpha[:, None]
                            acc = tl.dot(p, v, acc, allow_tf32=False)

                            # Conditional updates to avoid NaN propagation
                            l_i = tl.where(valid_rows, l_i * alpha + l_ij, l_i)
                            m_i = tl.where(valid_rows, m_ij, m_i)

    # Normalize and store with safe division
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

    # Store statistics
    if L is not None:
        l_offset = pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh
        l_ptrs = L + l_offset + offs_m * stride_lm
        log_l = tl.where(l_i > 0, tl.math.log(l_i), float("-inf"))
        tl.store(l_ptrs, log_l, mask=offs_m < N_CTX_Q)

    if M is not None:
        m_offset = pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh
        m_ptrs = M + m_offset + offs_m * stride_mm
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)


@triton.jit
def _nsa_sparse_selection_bwd_kernel(
    # Forward tensors
    Q,
    K,
    V,
    Block_indices,  # [B, G, T, n]
    Out,  # [B, H, T, dv]
    dOut,  # [B, H, T, dv]
    sm_scale,
    L,
    M,  # saved logsumexp stats
    # Strides - Q/K/V/Out
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    # Strides - Block indices
    stride_bz,
    stride_bg,
    stride_bm,
    stride_bn,
    # Strides - L/M
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    # Gradients (outputs)
    dQ,
    dK,
    dV,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkg,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvg,
    stride_dvn,
    stride_dvk,
    # Shapes/params
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_KV: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_SIZE_SELECTION: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Backward for sparse selection: computes dQ, dK, dV.
    Reconstructs probabilities with saved M/L; applies identical masks;
    uses atomics to accumulate into group-shared dK/dV.
    """
    # Program IDs
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    # Base offsets
    q_off = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_off = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_off = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_off = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    dq_off = pid_z.to(tl.int64) * stride_dqz + pid_h.to(tl.int64) * stride_dqh
    dk_off = pid_z.to(tl.int64) * stride_dkz + kv_group.to(tl.int64) * stride_dkg
    dv_off = pid_z.to(tl.int64) * stride_dvz + kv_group.to(tl.int64) * stride_dvg

    # Block pointers
    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dQ_ptr = tl.make_block_ptr(
        base=dQ + dq_off,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_dqm, stride_dqk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dOut_ptr = tl.make_block_ptr(
        base=dOut + o_off,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # Load tiles
    # Load Q unscaled and create scaled version for logits
    q_unscaled = tl.load(Q_ptr, boundary_check=(0, 1))
    q_scaled = q_unscaled * sm_scale  # Pre-scale Q for logits and dK computation
    do = tl.load(dOut_ptr, boundary_check=(0, 1))

    # Load output O and compute D = O^T @ dO globally
    Out_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    o = tl.load(Out_ptr, boundary_check=(0, 1))
    # Compute O^T @ dO for each query (this is sum(P * dP) globally)
    D = tl.sum(o * do, axis=1)  # [BLOCK_M]

    # Initialize
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_row = tl.load(
        M
        + (pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh)
        + offs_m * stride_mm,
        mask=offs_m < N_CTX_Q,
        other=-float("inf"),
    )
    l_row = tl.load(
        L
        + (pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh)
        + offs_m * stride_lm,
        mask=offs_m < N_CTX_Q,
        other=0.0,
    )

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM_QK], dtype=tl.float32)

    # Iterate over n selected blocks (same as forward)
    for block_idx in range(N_BLOCKS):
        # Load per-row selected block indices for this block position
        bidx_ptrs = (
            Block_indices
            + pid_z * stride_bz
            + kv_group * stride_bg
            + offs_m[:, None] * stride_bm
            + block_idx * stride_bn
        )
        valid_m = offs_m < N_CTX_Q
        sel_blocks = tl.load(bidx_ptrs, mask=valid_m[:, None], other=-1)
        sel_blocks = tl.sum(
            sel_blocks, axis=1
        )  # Flatten to 1D by summing (since it's a single column)

        # Range of unique blocks (skip -1 sentinel values)
        blocks_valid = sel_blocks >= 0
        has_valid = tl.sum(blocks_valid.to(tl.int32)) > 0
        if has_valid:  # Only process if we have valid blocks
            min_blk = tl.min(tl.where(blocks_valid, sel_blocks, 999999))
            max_blk = tl.max(tl.where(blocks_valid, sel_blocks, -1))
            for unique_block in range(min_blk, max_blk + 1):
                # Query mask for this block
                q_use = sel_blocks == unique_block
                if tl.sum(q_use.to(tl.int32)) > 0:  # Replace continue with conditional
                    blk_start = tl.full((), unique_block * BLOCK_SIZE_SELECTION, tl.int32)
                    blk_end = tl.minimum(
                        blk_start + tl.full((), BLOCK_SIZE_SELECTION, tl.int32),
                        tl.full((), N_CTX_KV, tl.int32)
                    )

                    # Row-wise boundaries
                    blk_start_row = tl.where(q_use, blk_start, N_CTX_KV)
                    blk_end_row = tl.where(q_use, blk_end, 0)

                    # Iterate KV tiles within the block
                    for kv_off in range(0, BLOCK_SIZE_SELECTION, BLOCK_N):
                        tile_start = blk_start + tl.full((), kv_off, tl.int32)
                        if tile_start < blk_end:  # Replace break with conditional
                            K_ptr = tl.make_block_ptr(
                                base=K + k_off,
                                shape=(HEAD_DIM_QK, N_CTX_KV),
                                strides=(stride_kk, stride_kn),
                                offsets=(0, tile_start),
                                block_shape=(HEAD_DIM_QK, BLOCK_N),
                                order=(0, 1),
                            )
                            V_ptr = tl.make_block_ptr(
                                base=V + v_off,
                                shape=(N_CTX_KV, HEAD_DIM_V),
                                strides=(stride_vn, stride_vk),
                                offsets=(tile_start, 0),
                                block_shape=(BLOCK_N, HEAD_DIM_V),
                                order=(1, 0),
                            )

                            k_tile = tl.load(K_ptr, boundary_check=(0, 1))
                            v_tile = tl.load(V_ptr, boundary_check=(0, 1))

                            # Recompute logits
                            # Compute logits s = (Q * sm_scale) @ K using pre-scaled Q
                            s = tl.dot(q_scaled, k_tile, allow_tf32=False)

                            offs_n = tile_start + tl.arange(0, BLOCK_N)
                            if IS_CAUSAL:
                                causal = offs_n[None, :] <= offs_m[:, None]
                            else:
                                causal = offs_m[:, None] >= -1
                            boundary = (offs_n[None, :] >= blk_start_row[:, None]) & (
                                offs_n[None, :] < blk_end_row[:, None]
                            )
                            valid = (offs_m[:, None] < N_CTX_Q) & (
                                offs_n[None, :] < N_CTX_KV
                            )
                            mask = causal & boundary & valid
                            s = tl.where(mask, s, -float("inf"))

                            # Reconstruct probabilities using saved M/L
                            # p = exp(s - M) / exp(L) = exp(s - M - L)
                            # Masked shift to avoid -inf - (-inf) = NaN
                            s_shift = tl.where(mask, s - m_row[:, None], float("-inf"))
                            p = tl.math.exp(s_shift)
                            # Normalize by exp(L)
                            p = p / tl.math.exp(l_row)[:, None]
                            p = tl.where(mask, p, 0.0)

                            # dp = dO V^T (per row)
                            dp = tl.dot(do, tl.trans(v_tile), allow_tf32=False)
                            dp = tl.where(mask, dp, 0.0)

                            # dV += p^T @ dO  (atomics on group-shared grads)
                            # shape: (BLOCK_N, dv)
                            dV_tile = tl.dot(tl.trans(p), do, allow_tf32=False)
                            # Atomic add into dV at tile positions
                            offs_v = tl.arange(0, HEAD_DIM_V)
                            offs_n = tile_start + tl.arange(0, BLOCK_N)
                            mask_n = offs_n < N_CTX_KV
                            # Accumulate to dV
                            dv_ptrs = (
                                dV
                                + dv_off
                                + offs_n[:, None] * stride_dvn
                                + offs_v[None, :] * stride_dvk
                            )
                            tl.atomic_add(
                                dv_ptrs,
                                dV_tile,
                                mask=mask_n[:, None] & (offs_v[None, :] < HEAD_DIM_V),
                            )

                            # Softmax backward: dS = P * (dP - D)
                            # D is the global row sum computed earlier: D = rowsum(O * dO) = rowsum(P * dP)
                            dS = p * (dp - D[:, None])

                            # dQ += dS @ K, where K needed here is [BLOCK_N, HEAD_DIM_QK]
                            # k_tile is loaded as [HEAD_DIM_QK, BLOCK_N], so transpose it
                            dq_acc += tl.dot(dS, tl.trans(k_tile), allow_tf32=False)

                            # dK += dS^T @ Q  (atomics on group-shared grads)
                            # Compute dK using pre-scaled Q (no additional scaling needed)
                            dK_tile = tl.dot(tl.trans(dS), q_scaled, allow_tf32=False)
                            offs_k = tl.arange(0, HEAD_DIM_QK)
                            offs_n = tile_start + tl.arange(0, BLOCK_N)
                            mask_n = offs_n < N_CTX_KV
                            # Accumulate to dK
                            dk_ptrs = (
                                dK
                                + dk_off
                                + offs_n[:, None] * stride_dkn
                                + offs_k[None, :] * stride_dkk
                            )
                            tl.atomic_add(
                                dk_ptrs,
                                dK_tile,
                                mask=mask_n[:, None] & (offs_k[None, :] < HEAD_DIM_QK),
                            )

    # Write dQ (multiply by sm_scale for chain rule)
    dq_acc = dq_acc * sm_scale
    tl.store(dQ_ptr, dq_acc, boundary_check=(0, 1))


@triton.jit
def _nsa_sliding_window_fwd_kernel(
    Q,
    K,
    V,
    Out,
    sm_scale,
    L,
    M,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sliding window kernel with consistent float('-inf') masking."""
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    q_offset = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_offset = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_offset = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_offset = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = q * sm_scale

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Start at 0, not 1e-10
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    window_start = tl.maximum(0, offs_m - WINDOW_SIZE + 1)
    window_end = tl.minimum(offs_m + 1, N_CTX)

    # Cannot use runtime values in range(), must iterate full sequence
    for start_n in range(0, N_CTX, BLOCK_N):
        K_block_ptr = tl.make_block_ptr(
            base=K + k_offset,
            shape=(HEAD_DIM_QK, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(HEAD_DIM_QK, BLOCK_N),
            order=(0, 1),
        )

        V_block_ptr = tl.make_block_ptr(
            base=V + v_offset,
            shape=(N_CTX, HEAD_DIM_V),
            strides=(stride_vn, stride_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM_V),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        qk = tl.dot(q, k, allow_tf32=False)

        offs_n = start_n + tl.arange(0, BLOCK_N)

        window_mask = (offs_n[None, :] >= window_start[:, None]) & (
            offs_n[None, :] < window_end[:, None]
        )
        valid_mask = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)

        mask = window_mask & valid_mask
        qk = tl.where(mask, qk, float("-inf"))

        # Online softmax with NaN protection
        valid_rows = tl.sum(mask.to(tl.int32), axis=1) > 0
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # Masked shift to avoid -inf - (-inf) = NaN
        qk_shift = tl.where(mask, qk - m_ij[:, None], float("-inf"))
        p = tl.math.exp(qk_shift)
        l_ij = tl.sum(p, axis=1)

        # Conditional alpha to avoid exp(NaN)
        alpha = tl.where(valid_rows, tl.math.exp(m_i - m_ij), 1.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, allow_tf32=False)

        # Conditional updates
        l_i = tl.where(valid_rows, l_i * alpha + l_ij, l_i)
        m_i = tl.where(valid_rows, m_ij, m_i)

    # Safe division: avoid NaN when l_i is 0
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

    if L is not None:
        l_offset = pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh
        l_ptrs = L + l_offset + offs_m * stride_lm
        # Safe log: avoid NaN when l_i is 0
        log_l = tl.where(l_i > 0, tl.math.log(l_i), float("-inf"))
        tl.store(l_ptrs, log_l, mask=offs_m < N_CTX)

    if M is not None:
        m_offset = pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh
        m_ptrs = M + m_offset + offs_m * stride_mm
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)


@triton.jit
def _nsa_sliding_window_bwd_kernel(
    # Forward tensors
    Q,
    K,
    V,
    Out,
    dOut,
    sm_scale,
    L,
    M,
    # Strides
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    # Grads
    dQ,
    dK,
    dV,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkg,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvg,
    stride_dvn,
    stride_dvk,
    # Shape
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward for sliding window attention."""
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    q_off = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_off = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_off = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_off = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    dq_off = pid_z.to(tl.int64) * stride_dqz + pid_h.to(tl.int64) * stride_dqh
    dk_off = pid_z.to(tl.int64) * stride_dkz + kv_group.to(tl.int64) * stride_dkg
    dv_off = pid_z.to(tl.int64) * stride_dvz + kv_group.to(tl.int64) * stride_dvg

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dQ_ptr = tl.make_block_ptr(
        base=dQ + dq_off,
        shape=(N_CTX, HEAD_DIM_QK),
        strides=(stride_dqm, stride_dqk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dOut_ptr = tl.make_block_ptr(
        base=dOut + o_off,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # Load Q unscaled and create scaled version for logits
    q_unscaled = tl.load(Q_ptr, boundary_check=(0, 1))
    q_scaled = q_unscaled * sm_scale  # Pre-scale Q for logits and dK computation
    do = tl.load(dOut_ptr, boundary_check=(0, 1))

    # Load output O and compute D = O^T @ dO globally
    Out_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    o = tl.load(Out_ptr, boundary_check=(0, 1))
    # Compute O^T @ dO for each query (this is sum(P * dP) globally)
    D = tl.sum(o * do, axis=1)  # [BLOCK_M]

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_row = tl.load(
        M
        + (pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh)
        + offs_m * stride_mm,
        mask=offs_m < N_CTX,
        other=-float("inf"),
    )
    l_row = tl.load(
        L
        + (pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh)
        + offs_m * stride_lm,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM_QK], dtype=tl.float32)

    # Per-row windows
    win_lo = tl.maximum(0, offs_m - WINDOW_SIZE + 1)
    win_hi = tl.minimum(offs_m + 1, N_CTX)

    # Cannot use runtime values in range(), must iterate full sequence
    # and rely on masking for correctness
    for start_n in range(0, N_CTX, BLOCK_N):
        K_ptr = tl.make_block_ptr(
            base=K + k_off,
            shape=(HEAD_DIM_QK, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, start_n),
            block_shape=(HEAD_DIM_QK, BLOCK_N),
            order=(0, 1),
        )
        V_ptr = tl.make_block_ptr(
            base=V + v_off,
            shape=(N_CTX, HEAD_DIM_V),
            strides=(stride_vn, stride_vk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM_V),
            order=(1, 0),
        )

        k_tile = tl.load(K_ptr, boundary_check=(0, 1))
        v_tile = tl.load(V_ptr, boundary_check=(0, 1))

        # Compute logits s = (Q * sm_scale) @ K using pre-scaled Q
        s = tl.dot(q_scaled, k_tile, allow_tf32=False)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        win_mask = (offs_n[None, :] >= win_lo[:, None]) & (
            offs_n[None, :] < win_hi[:, None]
        )
        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        mask = win_mask & valid
        s = tl.where(mask, s, -float("inf"))

        # Reconstruct p with masked shift to avoid NaN
        s_shift = tl.where(mask, s - m_row[:, None], float("-inf"))
        p = tl.math.exp(s_shift)
        p = p / tl.math.exp(l_row)[:, None]
        p = tl.where(mask, p, 0.0)

        # dp = dO V^T
        dp = tl.dot(do, tl.trans(v_tile), allow_tf32=False)
        dp = tl.where(mask, dp, 0.0)

        # dV += p^T @ dO
        dV_tile = tl.dot(tl.trans(p), do, allow_tf32=False)
        # Write dV_tile using direct memory access
        offs_v = tl.arange(0, HEAD_DIM_V)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        # Iterate over positions and accumulate
        dv_ptrs = (
            dV + dv_off + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk
        )
        tl.atomic_add(
            dv_ptrs, dV_tile, mask=mask_n[:, None] & (offs_v[None, :] < HEAD_DIM_V)
        )

        # Softmax backward: dS = P * (dP - D)
        # D is the global row sum computed earlier: D = rowsum(O * dO) = rowsum(P * dP)
        dS = p * (dp - D[:, None])
        # dQ = dS @ K where K should be [BLOCK_N, HEAD_DIM_QK]
        # But k_tile is [HEAD_DIM_QK, BLOCK_N], so transpose it
        dq_acc += tl.dot(dS, tl.trans(k_tile), allow_tf32=False)

        # Compute dK using pre-scaled Q (no additional scaling needed)
        dK_tile = tl.dot(tl.trans(dS), q_scaled, allow_tf32=False)
        # Write dK_tile using direct memory access
        offs_k = tl.arange(0, HEAD_DIM_QK)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        # Accumulate to dK
        dk_ptrs = (
            dK + dk_off + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
        )
        tl.atomic_add(
            dk_ptrs, dK_tile, mask=mask_n[:, None] & (offs_k[None, :] < HEAD_DIM_QK)
        )

    # Apply sm_scale factor for chain rule (dQ_unscaled = dQ_scaled * sm_scale)
    dq_acc = dq_acc * sm_scale
    tl.store(dQ_ptr, dq_acc, boundary_check=(0, 1))


@triton.jit
def _nsa_compression_fwd_kernel(
    Q,
    K_compressed,
    V_compressed,
    Block_ends,
    Out,
    sm_scale,
    L,
    M,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compression kernel with block-end causality."""
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    q_offset = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_offset = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_offset = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_offset = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    q = q * sm_scale

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Start at 0, not 1e-10
    acc = tl.zeros([BLOCK_M, HEAD_DIM_V], dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        base=K_compressed + k_offset,
        shape=(HEAD_DIM_QK, N_BLOCKS),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_compressed + v_offset,
        shape=(N_BLOCKS, HEAD_DIM_V),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=(1, 0),
    )

    for block_start in range(0, N_BLOCKS, BLOCK_N):
        block_ends_ptrs = Block_ends + block_start + tl.arange(0, BLOCK_N)
        block_end_positions = tl.load(
            block_ends_ptrs,
            mask=(block_start + tl.arange(0, BLOCK_N)) < N_BLOCKS,
            other=999999,
        )

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        qk = tl.dot(q, k, allow_tf32=False)

        # Block-end causality: block_ends[i] = i*d + (l-1)
        block_causal_mask = block_end_positions[None, :] <= offs_m[:, None]
        valid_block_mask = (block_start + tl.arange(0, BLOCK_N))[None, :] < N_BLOCKS
        valid_query_mask = offs_m[:, None] < N_CTX_Q

        mask = block_causal_mask & valid_block_mask & valid_query_mask
        qk = tl.where(mask, qk, float("-inf"))

        # Online softmax with NaN protection
        valid_rows = tl.sum(mask.to(tl.int32), axis=1) > 0
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # Masked shift to avoid -inf - (-inf) = NaN
        qk_shift = tl.where(mask, qk - m_ij[:, None], float("-inf"))
        p = tl.math.exp(qk_shift)
        l_ij = tl.sum(p, axis=1)

        # Conditional alpha to avoid exp(NaN)
        alpha = tl.where(valid_rows, tl.math.exp(m_i - m_ij), 1.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc, allow_tf32=False)

        # Conditional updates
        l_i = tl.where(valid_rows, l_i * alpha + l_ij, l_i)
        m_i = tl.where(valid_rows, m_ij, m_i)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Safe division: if l_i is 0, output is 0 (no valid attention)
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

    if L is not None:
        l_offset = pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh
        l_ptrs = L + l_offset + offs_m * stride_lm
        # Safe log: if l_i is 0, store -inf
        log_l = tl.where(l_i > 0, tl.math.log(l_i), -float("inf"))
        tl.store(l_ptrs, log_l, mask=offs_m < N_CTX_Q)

    if M is not None:
        m_offset = pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh
        m_ptrs = M + m_offset + offs_m * stride_mm
        tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)


@triton.jit
def _nsa_compression_bwd_kernel(
    # Forward tensors
    Q,
    K_blocks,
    V_blocks,
    Block_ends,
    Out,
    dOut,
    sm_scale,
    L,
    M,
    # Strides
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kg,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vg,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    # Grads
    dQ,
    dK,
    dV,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dkz,
    stride_dkg,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvg,
    stride_dvn,
    stride_dvk,
    # Shapes
    Z: tl.constexpr,
    H: tl.constexpr,
    N_KV_GROUPS: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    HEAD_DIM_QK: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward for compression attention (block-end causal)."""
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    pid_z = pid_hz // H
    pid_h = pid_hz % H
    kv_group = pid_h // (H // N_KV_GROUPS)

    q_off = pid_z.to(tl.int64) * stride_qz + pid_h.to(tl.int64) * stride_qh
    k_off = pid_z.to(tl.int64) * stride_kz + kv_group.to(tl.int64) * stride_kg
    v_off = pid_z.to(tl.int64) * stride_vz + kv_group.to(tl.int64) * stride_vg
    o_off = pid_z.to(tl.int64) * stride_oz + pid_h.to(tl.int64) * stride_oh

    dq_off = pid_z.to(tl.int64) * stride_dqz + pid_h.to(tl.int64) * stride_dqh
    dk_off = pid_z.to(tl.int64) * stride_dkz + kv_group.to(tl.int64) * stride_dkg
    dv_off = pid_z.to(tl.int64) * stride_dvz + kv_group.to(tl.int64) * stride_dvg

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dQ_ptr = tl.make_block_ptr(
        base=dQ + dq_off,
        shape=(N_CTX_Q, HEAD_DIM_QK),
        strides=(stride_dqm, stride_dqk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_QK),
        order=(1, 0),
    )
    dOut_ptr = tl.make_block_ptr(
        base=dOut + o_off,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )

    # Load Q unscaled and create scaled version for logits
    q_unscaled = tl.load(Q_ptr, boundary_check=(0, 1))
    q_scaled = q_unscaled * sm_scale  # Pre-scale Q for logits and dK computation
    do = tl.load(dOut_ptr, boundary_check=(0, 1))

    # Load output O and compute D = O^T @ dO globally
    Out_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM_V),
        strides=(stride_om, stride_ok),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM_V),
        order=(1, 0),
    )
    o = tl.load(Out_ptr, boundary_check=(0, 1))
    # Compute O^T @ dO for each query (this is sum(P * dP) globally)
    D = tl.sum(o * do, axis=1)  # [BLOCK_M]

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_row = tl.load(
        M
        + (pid_z.to(tl.int64) * stride_mz + pid_h.to(tl.int64) * stride_mh)
        + offs_m * stride_mm,
        mask=offs_m < N_CTX_Q,
        other=-float("inf"),
    )
    l_row = tl.load(
        L
        + (pid_z.to(tl.int64) * stride_lz + pid_h.to(tl.int64) * stride_lh)
        + offs_m * stride_lm,
        mask=offs_m < N_CTX_Q,
        other=0.0,
    )

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM_QK], dtype=tl.float32)

    K_ptr = tl.make_block_ptr(
        base=K_blocks + k_off,
        shape=(HEAD_DIM_QK, N_BLOCKS),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM_QK, BLOCK_N),
        order=(0, 1),
    )
    V_ptr = tl.make_block_ptr(
        base=V_blocks + v_off,
        shape=(N_BLOCKS, HEAD_DIM_V),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM_V),
        order=(1, 0),
    )

    for blk_start in range(0, N_BLOCKS, BLOCK_N):
        blk_ends_ptrs = Block_ends + blk_start + tl.arange(0, BLOCK_N)
        blk_end_pos = tl.load(
            blk_ends_ptrs,
            mask=(blk_start + tl.arange(0, BLOCK_N)) < N_BLOCKS,
            other=999999,
        )

        k_tile = tl.load(K_ptr, boundary_check=(0, 1))
        v_tile = tl.load(V_ptr, boundary_check=(0, 1))

        # Compute logits s = (Q * sm_scale) @ K using pre-scaled Q
        s = tl.dot(q_scaled, k_tile, allow_tf32=False)
        blk_causal = blk_end_pos[None, :] <= offs_m[:, None]
        valid_blk = (blk_start + tl.arange(0, BLOCK_N))[None, :] < N_BLOCKS
        valid_row = offs_m[:, None] < N_CTX_Q
        mask = blk_causal & valid_blk & valid_row
        s = tl.where(mask, s, -float("inf"))

        # Reconstruct p with masked shift to avoid NaN
        s_shift = tl.where(mask, s - m_row[:, None], float("-inf"))
        p = tl.math.exp(s_shift)
        p = p / tl.math.exp(l_row)[:, None]
        p = tl.where(mask, p, 0.0)

        # dp = dO V^T
        dp = tl.dot(do, tl.trans(v_tile), allow_tf32=False)
        dp = tl.where(mask, dp, 0.0)

        # dV += p^T @ dO
        dV_tile = tl.dot(tl.trans(p), do, allow_tf32=False)
        offs_v = tl.arange(0, HEAD_DIM_V)
        offs_n = blk_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_BLOCKS
        # Accumulate to dV
        dv_ptrs = (
            dV + dv_off + offs_n[:, None] * stride_dvn + offs_v[None, :] * stride_dvk
        )
        tl.atomic_add(
            dv_ptrs, dV_tile, mask=mask_n[:, None] & (offs_v[None, :] < HEAD_DIM_V)
        )

        # Softmax backward: dS = P * (dP - D)
        # D is the global row sum computed earlier: D = rowsum(O * dO) = rowsum(P * dP)
        dS = p * (dp - D[:, None])
        # dQ = dS @ K where K should be [BLOCK_N, HEAD_DIM_QK]
        # But k_tile is [HEAD_DIM_QK, BLOCK_N], so transpose it
        dq_acc += tl.dot(dS, tl.trans(k_tile), allow_tf32=False)

        # Compute dK using pre-scaled Q (no additional scaling needed)
        dK_tile = tl.dot(tl.trans(dS), q_scaled, allow_tf32=False)
        offs_k = tl.arange(0, HEAD_DIM_QK)
        offs_n = blk_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_BLOCKS
        # Accumulate to dK
        dk_ptrs = (
            dK + dk_off + offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
        )
        tl.atomic_add(
            dk_ptrs, dK_tile, mask=mask_n[:, None] & (offs_k[None, :] < HEAD_DIM_QK)
        )

        K_ptr = tl.advance(K_ptr, (0, BLOCK_N))
        V_ptr = tl.advance(V_ptr, (BLOCK_N, 0))

    # Apply sm_scale factor for chain rule (dQ_unscaled = dQ_scaled * sm_scale)
    dq_acc = dq_acc * sm_scale
    tl.store(dQ_ptr, dq_acc, boundary_check=(0, 1))


def derive_selection_from_compression_per_timestep(
    compression_scores: torch.Tensor,  # [B, H, T, n_comp_blocks]
    config: NSAConfig,
) -> torch.Tensor:
    """
    Per-timestep selection derivation following exact Eq. 9 from paper.

    Each query position t has its own compression attention scores p_cmp(t),
    from which we derive selection scores p_slc(t) via Eq. 9 triangular mapping.

    Args:
        compression_scores: [B, H, T, n_comp_blocks] - attention scores per timestep
        config: NSA configuration

    Returns:
        selection_scores: [B, H, T, n_sel_blocks] - importance scores per timestep
    """
    B, H, T, n_comp_blocks = compression_scores.shape

    # Validate alignment
    assert config.l % config.d == 0
    assert config.l_prime % config.d == 0

    a = config.l // config.d  # Compression blocks per l
    s = config.l_prime // config.d  # Compression blocks per l'
    n_sel_blocks = (n_comp_blocks + s - 1) // s

    device = compression_scores.device
    dtype = compression_scores.dtype
    selection_scores = torch.zeros(B, H, T, n_sel_blocks, device=device, dtype=dtype)

    # Apply Eq. 9 independently for each timestep
    # Rectangular double sum: m=0..s-1, n=0..a-1
    for j in range(n_sel_blocks):
        for m in range(s):
            for n in range(a):  # Full range, no min()
                comp_idx = s * j + m + n
                if comp_idx < n_comp_blocks:
                    selection_scores[:, :, :, j] += compression_scores[
                        :, :, :, comp_idx
                    ]

    return selection_scores


def select_top_k_blocks_per_query_with_gqa(
    selection_scores: torch.Tensor,  # [B, H, T, n_blocks]
    config: NSAConfig,
    always_include_blocks: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Per-query top-k selection with GQA aggregation (Eq. 10 from paper).

    1. Aggregate scores across heads in each group (Eq. 10)
    2. Select top-k blocks per query position
    3. Add always-include blocks (1 initial + 2 local)
    4. Deduplicate to exactly n blocks per query

    Args:
        selection_scores: [B, H, T, n_blocks] - per-timestep importance scores
        config: NSA configuration
        always_include_blocks: Fixed blocks to always include

    Returns:
        indices: [B, G, T, n] - selected block indices per group
    """
    B, H, T, n_blocks = selection_scores.shape

    # Step 1: GQA aggregation (Eq. 10)
    # Reshape to groups and aggregate across heads in each group
    scores_grouped = selection_scores.view(
        B, config.n_kv_groups, config.heads_per_group, T, n_blocks
    )
    aggregated_scores = scores_grouped.sum(dim=2)  # [B, G, T, n_blocks]

    # Step 2: Select top-k per query position
    n_dynamic = config.n_dynamic  # n - n_fixed
    # Ensure we don't select more blocks than available
    k_actual = min(n_dynamic, n_blocks)

    # Use stable sort for complete determinism instead of topk
    # Sort in descending order and take top k
    _, sorted_indices = torch.sort(aggregated_scores, dim=-1, descending=True, stable=True)
    top_indices = sorted_indices[..., :k_actual]  # [B, G, T, k_actual]

    # Pad if we selected fewer blocks than needed
    if k_actual < n_dynamic:
        padding = torch.zeros(
            B,
            config.n_kv_groups,
            T,
            n_dynamic - k_actual,
            device=aggregated_scores.device,
            dtype=torch.long,
        )
        top_indices = torch.cat([top_indices, padding], dim=-1)

    # Keep per-group (don't expand to heads)
    # top_indices is [B, G, T, n_dynamic]

    # Step 3: Add always-include blocks (VECTORIZED)
    device = selection_scores.device
    G = config.n_kv_groups

    # Initial block (block 0) for all positions
    initial_blocks = torch.zeros(B, G, T, 1, device=device, dtype=torch.long)

    # Local blocks (2 most recent blocks for each query) - VECTORIZED
    positions = torch.arange(T, device=device)
    local_block_1 = torch.maximum(
        torch.zeros_like(positions), (positions - config.l_prime) // config.l_prime
    )
    local_block_2 = positions // config.l_prime

    # Stack and expand to [B, G, T, 2]
    local_blocks = torch.stack([local_block_1, local_block_2], dim=-1)
    local_blocks = local_blocks.unsqueeze(0).unsqueeze(0).expand(B, G, T, 2)

    # Combine always-include blocks
    always_include = torch.cat(
        [initial_blocks, local_blocks], dim=-1
    )  # [B, G, T, n_fixed]

    # Step 4: Combine and deduplicate (STABLE UNIQUE)
    all_indices = torch.cat(
        [always_include, top_indices], dim=-1
    )  # [B, G, T, n_fixed + n_dynamic]

    # Deduplicate per group per query (keep first n unique, preserving order)
    final_indices = torch.zeros(B, G, T, config.n, device=device, dtype=torch.long)

    for b in range(B):
        for g in range(G):
            for t in range(T):
                # Take this group's indices
                indices_t = all_indices[b, g, t]

                # Stable unique: preserve order of first occurrence
                seen = set()
                unique_indices = []
                for idx in indices_t.tolist():
                    if idx not in seen and len(unique_indices) < config.n:
                        seen.add(idx)
                        unique_indices.append(idx)

                # Convert back to tensor and pad with -1 if needed
                unique_indices = torch.tensor(
                    unique_indices, device=device, dtype=torch.long
                )
                if len(unique_indices) < config.n:
                    padding = torch.full(
                        (config.n - len(unique_indices),),
                        -1,
                        device=device,
                        dtype=torch.long,
                    )
                    unique_indices = torch.cat([unique_indices, padding])

                final_indices[b, g, t] = unique_indices

    return final_indices


class NSAAttention(torch.nn.Module):
    """
    Complete NSA implementation.

    Features:
    - Per-timestep selection derivation
    - Proper d_model → heads/groups projection
    - Complete compress_tokens with MLP pooling
    - Full forward path implementation
    """

    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config

        # Project from d_model to heads/groups
        # Query projection: d_model → H * dk
        self.q_proj = torch.nn.Linear(
            config.d_model, config.n_heads * config.head_dim_qk, bias=config.use_bias
        )

        # Three independent K/V projections for each branch
        # Compression: d_model → G * dk/dv
        self.k_compress = torch.nn.Linear(
            config.d_model,
            config.n_kv_groups * config.head_dim_qk,
            bias=config.use_bias,
        )
        self.v_compress = torch.nn.Linear(
            config.d_model, config.n_kv_groups * config.head_dim_v, bias=config.use_bias
        )

        # Selection: d_model → G * dk/dv
        self.k_select = torch.nn.Linear(
            config.d_model,
            config.n_kv_groups * config.head_dim_qk,
            bias=config.use_bias,
        )
        self.v_select = torch.nn.Linear(
            config.d_model, config.n_kv_groups * config.head_dim_v, bias=config.use_bias
        )

        # Sliding: d_model → G * dk/dv
        self.k_sliding = torch.nn.Linear(
            config.d_model,
            config.n_kv_groups * config.head_dim_qk,
            bias=config.use_bias,
        )
        self.v_sliding = torch.nn.Linear(
            config.d_model, config.n_kv_groups * config.head_dim_v, bias=config.use_bias
        )

        # Compression MLP with position encoding
        self.compress_mlp = torch.nn.Sequential(
            torch.nn.Linear(config.head_dim_qk + config.l, config.head_dim_qk * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.head_dim_qk * 2, config.head_dim_qk),
        )

        # Gate network for branch mixing
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(config.d_model // 2, 3 * config.n_heads),
        )

        # Output projection: H * dv → d_model
        self.o_proj = torch.nn.Linear(
            config.n_heads * config.head_dim_v, config.d_model, bias=config.use_bias
        )

        self.dropout = torch.nn.Dropout(config.dropout_p)
        self.scale = 1.0 / math.sqrt(config.head_dim_qk)

    def compress_tokens(
        self,
        keys: torch.Tensor,  # [B, G, T, dk]
        values: torch.Tensor,  # [B, G, T, dv]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compression with MLP pooling and block_ends computation.

        Returns:
            compressed_keys: [B, G, dk, n_blocks_padded] - padded to power of 2
            compressed_values: [B, G, n_blocks_padded, dv] - padded to power of 2
            block_ends: [n_blocks_padded] with values i*d + (l-1) for real blocks, 999999 for padding
            n_blocks_actual: int - actual number of blocks (before padding)
        """
        B, G, T, dk = keys.shape
        dv = values.shape[-1]

        # Calculate number of blocks
        n_blocks_actual = (T - self.config.l) // self.config.d + 1

        # Pad to next power of 2 for Triton
        n_blocks_padded = 1
        while n_blocks_padded < n_blocks_actual:
            n_blocks_padded *= 2

        device = keys.device
        dtype = keys.dtype

        compressed_keys = []
        compressed_values = []
        block_ends = []

        for i in range(n_blocks_actual):
            start = i * self.config.d
            end = min(start + self.config.l, T)

            # Extract block
            k_block = keys[:, :, start:end, :]  # [B, G, block_len, dk]
            v_block = values[:, :, start:end, :]  # [B, G, block_len, dv]

            block_len = end - start

            # Create position encoding
            pos = torch.arange(block_len, device=device, dtype=dtype)
            pos = pos / block_len  # Normalize to [0, 1]

            # Expand position encoding
            pos_expanded = pos.view(1, 1, block_len, 1).expand(
                B, G, block_len, self.config.l
            )

            # Concatenate keys with position encoding
            k_with_pos = torch.cat(
                [k_block, pos_expanded[..., : self.config.l]], dim=-1
            )

            # Apply MLP and pool
            k_compressed = self.compress_mlp(k_with_pos)  # [B, G, block_len, dk]
            k_compressed = k_compressed.mean(dim=2)  # [B, G, dk]

            # Simple average pooling for values
            v_compressed = v_block.mean(dim=2)  # [B, G, dv]

            compressed_keys.append(k_compressed)
            compressed_values.append(v_compressed)

            # Block end position: i*d + (l-1)
            block_ends.append(i * self.config.d + self.config.l - 1)

        # Pad to power of 2
        for i in range(n_blocks_actual, n_blocks_padded):
            # Add dummy blocks (will be masked out by block_ends = 999999)
            compressed_keys.append(torch.zeros(B, G, dk, device=device, dtype=dtype))
            compressed_values.append(torch.zeros(B, G, dv, device=device, dtype=dtype))
            block_ends.append(999999)  # Sentinel value for padding

        # Stack compressed blocks
        compressed_keys = torch.stack(
            compressed_keys, dim=-1
        )  # [B, G, dk, n_blocks_padded]
        compressed_values = torch.stack(
            compressed_values, dim=2
        )  # [B, G, n_blocks_padded, dv]

        # Build [n_blocks_padded] with sentinel for padding
        # Block ends are the same for all batches/groups (just based on position)
        block_ends_tensor = torch.full(
            (n_blocks_padded,), 999999, device=device, dtype=torch.int32
        )
        for i in range(n_blocks_actual):
            block_ends_tensor[i] = i * self.config.d + self.config.l - 1

        return compressed_keys, compressed_values, block_ends_tensor, n_blocks_actual

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, d_model]
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Complete forward path.

        Flow: compression → p_cmp(t) → derive selection → selection + sliding → gate → output
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Deterministic eval: optionally disable TF32 and enforce deterministic
        # algorithms for the entire forward to ensure reproducibility across
        # repeated calls with identical inputs.
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
        prev_prec = None
        prev_det = torch.are_deterministic_algorithms_enabled()
        if hasattr(torch, "get_float32_matmul_precision"):
            try:
                prev_prec = torch.get_float32_matmul_precision()
            except Exception:
                prev_prec = None
        if not self.training:
            try:
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
                # Enforce deterministic algorithms during eval forward
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

        # Project queries
        q = self.q_proj(hidden_states)  # [B, T, H*dk]
        q = q.view(B, T, self.config.n_heads, self.config.head_dim_qk).transpose(
            1, 2
        )  # [B, H, T, dk]

        # Project K/V for three branches
        # Compression
        k_comp = (
            self.k_compress(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_qk)
            .transpose(1, 2)
        )
        v_comp = (
            self.v_compress(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_v)
            .transpose(1, 2)
        )

        # Selection
        k_sel = (
            self.k_select(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_qk)
            .transpose(1, 2)
        )
        v_sel = (
            self.v_select(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_v)
            .transpose(1, 2)
        )

        # Sliding
        k_slide = (
            self.k_sliding(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_qk)
            .transpose(1, 2)
        )
        v_slide = (
            self.v_sliding(hidden_states)
            .view(B, T, self.config.n_kv_groups, self.config.head_dim_v)
            .transpose(1, 2)
        )

        # 1. COMPRESSION BRANCH
        k_compressed, v_compressed, block_ends, n_blocks_actual = self.compress_tokens(
            k_comp, v_comp
        )
        n_blocks = k_compressed.shape[-1]

        # Compute compression attention scores (Eq. 8 from paper)
        # Q @ K_compressed^T to get attention weights
        H = self.config.n_heads
        # Reshape q for compression attention: [B, H, T, dk]
        q_comp = q.view(B, H, T, self.config.head_dim_qk)

        # Compute Q @ K_compressed^T for each head-group pair
        # We need to expand k_compressed for all heads in each group
        G = self.config.n_kv_groups
        heads_per_group = H // G

        # k_compressed: [B, G, dk, n_blocks] -> [B, H, dk, n_blocks]
        k_comp_expanded = k_compressed.unsqueeze(2).repeat(1, 1, heads_per_group, 1, 1)
        k_comp_expanded = k_comp_expanded.view(B, H, self.config.head_dim_qk, n_blocks)

        # Compute attention scores: [B, H, T, dk] @ [B, H, dk, n_blocks] -> [B, H, T, n_blocks]
        compression_scores = torch.matmul(q_comp, k_comp_expanded) * self.scale

        # Apply causal mask based on block_ends using broadcasting
        # block_ends: [n_blocks] - same for all batches/groups
        H, G = self.config.n_heads, self.config.n_kv_groups
        t_arange = torch.arange(T, device=device)  # [T]
        # Broadcast block_ends to compare with time positions
        visible = (
            block_ends[None, None, None, :] <= t_arange[None, None, :, None]
        )  # [1,1,T,n_blocks]
        # Expand to full batch/head dimensions
        visible = visible.expand(B, H, T, -1)  # [B,H,T,n_blocks]

        # Enforce at least one visible block per row (fallback to block 0)
        any_visible = visible.any(dim=-1, keepdim=True)  # [B,H,T,1]
        fallback = ~any_visible  # rows with no visible blocks
        # Mask tensor with only the first block set True
        first_only = torch.zeros_like(visible)
        first_only[..., 0] = True
        visible = visible | (first_only & fallback)

        # Apply mask and softmax
        compression_scores = compression_scores.masked_fill(~visible, float("-inf"))
        compression_scores = torch.softmax(compression_scores, dim=-1)

        # 2. DERIVE SELECTION from compression scores
        selection_scores = derive_selection_from_compression_per_timestep(
            compression_scores, self.config
        )

        # 3. SELECT TOP-K BLOCKS per query with GQA
        selected_indices = select_top_k_blocks_per_query_with_gqa(
            selection_scores, self.config
        )  # [B, G, T, n]

        # 4. CALL KERNELS
        # Prepare for kernel calls
        G = self.config.n_kv_groups

        # Reshape tensors for kernels
        # Q: [B, H, T, dk] -> need to flatten batch-head for kernels
        q_kernel = q.contiguous()

        # K/V for selection/sliding: [B, G, T, dk/dv] -> [B, G, dk/dv, T] for kernels
        k_sel_kernel = k_sel.transpose(-1, -2).contiguous()
        v_sel_kernel = v_sel.contiguous()

        k_slide_kernel = k_slide.transpose(-1, -2).contiguous()
        v_slide_kernel = v_slide.contiguous()

        # Use autograd wrappers in both train/eval for consistency
        o_compress = CompressionAttention.apply(
            q_kernel,
            k_compressed,
            v_compressed,
            block_ends,
            self.scale,
            self.config,
        )

        o_select = SelectionAttention.apply(
            q_kernel,
            k_sel_kernel,
            v_sel_kernel,
            selected_indices,
            self.scale,
            self.config,
        )

        o_slide = SlidingWindowAttention.apply(
            q_kernel,
            k_slide_kernel,
            v_slide_kernel,
            self.scale,
            self.config.w,
            self.config,
        )

        # 5. GATE AND MIX branches (softmax, deterministic op order)
        gates = self.gate_mlp(hidden_states)  # [B, T, 3*H]
        gates = gates.view(B, T, self.config.n_heads, 3).permute(0, 2, 1, 3)  # [B, H, T, 3]
        gates = torch.softmax(gates, dim=-1)

        o_mixed = o_compress.mul(gates[..., 0:1])
        o_mixed = o_mixed.add(o_select.mul(gates[..., 1:2]))
        o_mixed = o_mixed.add(o_slide.mul(gates[..., 2:3]))  # [B, H, T, dv]

        # 6. OUTPUT PROJECTION
        o_mixed = o_mixed.transpose(1, 2).contiguous()  # [B, T, H, dv]
        o_mixed = o_mixed.view(B, T, self.config.n_heads * self.config.head_dim_v)

        output = self.o_proj(o_mixed)  # [B, T, d_model]
        output = self.dropout(output)

        # Return attention info if requested
        attn_info = None
        if output_attentions:
            attn_info = {
                "compression_scores": compression_scores,
                "selection_scores": selection_scores,
                "selected_indices": selected_indices,
                "gates": gates.permute(0, 2, 1, 3),  # [B, T, H, 3]
            }

        # Restore determinism/matmul settings at the very end (after all dense ops)
        if not self.training:
            try:
                if torch.cuda.is_available() and prev_tf32 is not None:
                    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
                if prev_prec is not None and hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision(prev_prec)
                # Restore previous deterministic mode
                torch.use_deterministic_algorithms(prev_det)
            except Exception:
                pass

        return output, attn_info
