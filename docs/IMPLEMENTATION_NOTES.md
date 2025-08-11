# NSA Implementation Notes

## Critical Implementation Details

### K Tensor Stride Ordering

**THE MOST IMPORTANT BUG FIX**: The kernel expects K strides in a specific order that differs from the tensor layout.

With K tensor shape `[B, G, dk, T]`, the kernel expects:
- `stride_kn` = stride for time axis (k.stride(3))
- `stride_kk` = stride for dk axis (k.stride(2))

**Always pass K strides as**: `k.stride(0), k.stride(1), k.stride(3), k.stride(2)`

This was the root cause of 140-168% gradient errors that took extensive debugging to identify.

### dK Tensor Shape

For backward pass, dK must be allocated as `[B, G, T, dk]` to match kernel's atomic accumulation pattern:
```python
dk_kernel = torch.zeros(B, G, T, dk, device='cuda', dtype=torch.float32)
```

After kernel execution, transpose to `[B, G, dk, T]` if needed for comparison with reference.

### Grouped Query Attention (GQA)

The kernels support GQA with atomic accumulation across heads in a group:
- Q shape: `[B, H, T, dk]` 
- K shape: `[B, G, dk, T]` where G = number of KV groups
- V shape: `[B, G, T, dv]`
- Heads per group = H // G

### Softmax Precision

All softmax operations use float32 for numerical stability:
- M (max) and L (logsumexp) statistics are always float32
- Intermediate exponentials computed in float32
- Critical for avoiding NaN/inf in gradients

### Block Alignment

Sliding window uses block-aligned iteration:
```python
lo = tl.maximum(0, col_idx * BLOCK_N - WINDOW_SIZE + 1)
hi = tl.minimum(N_CTX, col_idx * BLOCK_N + BLOCK_N)
```

### Scale Factor

The scale factor (1/√dk) is applied consistently:
- Forward: `q_scaled = q * sm_scale` before computing QK^T
- Backward: Scale preserved through chain rule

## Performance Characteristics

On NVIDIA A100:
- Sliding window: ~219K tokens/sec
- Compression: ~3K tokens/sec
- Block size: 32x32 for optimal occupancy
- Shared memory usage: Critical constraint for larger blocks

## Testing Requirements

Production accuracy requirement: <0.001 relative error for all gradients
- Forward pass: < 0.001 relative error achieved
- dQ gradient: < 0.001 relative error achieved  
- dK gradient: < 0.001 relative error achieved
- dV gradient: < 0.001 relative error achieved

## Common Pitfalls

1. **Wrong K stride order**: Most common error, causes 140%+ gradient errors
2. **Wrong dK shape**: Kernel expects [B,G,T,dk] not [B,G,dk,T]
3. **Float16 softmax**: Causes numerical instability
4. **Unaligned blocks**: Can cause NaN in edge cases
5. **Missing atomic locks**: Required for GQA gradient accumulation

## Debugging Tips

1. Test with identity K first (diagonal matrix)
2. Check stride ordering carefully
3. Verify tensor shapes match kernel expectations
4. Use float32 for initial testing
5. Start with small dimensions (B=1, H=1, T=32)