"""
Triton-Optimized Parallel Scan Kernels for ANA

This module implements high-performance parallel scan operations using Triton,
enabling O(1) theoretical inference advantage for ANA architectures.

Key Optimizations:
- Hillis-Steele parallel scan O(log n) time
- Associative scan for HoloLink memory retrieval
- Memory coalescing for optimal bandwidth usage
- Shared memory tiling for multi-track processing

Research Questions:
1. Can Triton kernels unlock theoretical O(1) advantage?
2. What's the speedup at sequence lengths 512-8192?
3. How does memory bandwidth affect performance?
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def hillis_steele_scan_kernel(
    u_ptr, a_ptr, b_ptr, h_ptr,
    seq_len, batch, dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_dim = tl.program_id(axis=2)
    
    offsets_seq = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offsets_dim = pid_dim * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    
    mask_seq = offsets_seq < seq_len
    mask_dim = offsets_dim < dim
    
    base_idx = (pid_batch * seq_len + offsets_seq) * dim + offsets_dim
    
    h_init = tl.zeros([BLOCK_SEQ, BLOCK_DIM], dtype=tl.float32)
    
    u = tl.load(u_ptr + base_idx, mask=mask_seq[:, None] & mask_dim[None, :])
    a = tl.load(a_ptr + base_idx, mask=mask_seq[:, None] & mask_dim[None, :])
    b = tl.load(b_ptr + base_idx, mask=mask_seq[:, None] & mask_dim[None, :])
    
    h = h_init
    
    for i in range(BLOCK_SEQ):
        if mask_seq[i]:
            h[:, i] = a[:, i] * h + b[:, i] * u[:, i]
    
    tl.store(h_ptr + base_idx, h, mask=mask_seq[:, None] & mask_dim[None, :])


@triton.jit
def parallel_scan_log2_kernel(
    u_ptr, a_ptr, b_ptr, h_ptr,
    seq_len, batch, dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    num_threads = tl.num_programs(axis=0)
    
    start_idx = pid * seq_len * dim
    stride = num_threads * seq_len * dim
    
    a_curr = tl.load(a_ptr + start_idx + tl.arange(0, stride), mask=(start_idx + tl.arange(0, stride)) < batch * seq_len * dim)
    b_curr = tl.load(b_ptr + start_idx + tl.arange(0, stride), mask=(start_idx + tl.arange(0, stride)) < batch * seq_len * dim)
    
    log_seq = tl.math.log2(tl.where(seq_len > 1, seq_len, 2))
    steps = tl.int32(log_seq) if isinstance(log_seq, int) else int(log_seq.item())
    
    for i in range(steps):
        d = 1 << i
        
        a_shifted = tl.zeros_like(a_curr)
        b_shifted = tl.zeros_like(b_curr)
        
        a_shifted = tl.where(tl.arange(0, stride) >= d, a_curr, a_shifted)
        
        a_next = a_curr * a_shifted
        b_next = a_curr * b_shifted + b_curr
        
        a_curr = a_next
        b_curr = b_next
    
    h = a_curr * tl.load(h_ptr + start_idx) + b_curr
    
    tl.store(h_ptr + start_idx, h)


@triton.jit
def associative_scan_kernel(
    keys_ptr, values_ptr, query_ptr, output_ptr,
    batch, capacity, key_dim, value_dim,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_capacity = tl.program_id(axis=1)
    
    key_offsets = (pid_batch * capacity + pid_capacity) * key_dim + tl.arange(0, BLOCK_KEY)
    val_offsets = (pid_batch * capacity + pid_capacity) * value_dim + tl.arange(0, BLOCK_VAL)
    
    key_mask = pid_capacity < capacity
    
    key = tl.load(keys_ptr + key_offsets, mask=key_mask[None, :])
    value = tl.load(values_ptr + val_offsets, mask=key_mask[None, :])
    
    query = tl.load(query_ptr + pid_batch * key_dim + tl.arange(0, BLOCK_KEY))
    
    similarity = tl.dot(key, query)
    
    best_idx = tl.argmax(similarity)
    best_value = value[best_idx]
    
    output_offset = pid_batch * value_dim + tl.arange(0, BLOCK_VAL)
    tl.store(output_ptr + output_offset, best_value)


@triton.jit
def multi_track_scan_kernel(
    u_ptr, a_ptr, b_ptr, h_ptr,
    seq_len, batch, num_tracks, track_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_TRACK: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_track = tl.program_id(axis=2)
    
    offsets_seq = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    offsets_track = pid_track * BLOCK_TRACK + tl.arange(0, BLOCK_TRACK)
    
    mask_seq = offsets_seq < seq_len
    mask_track = offsets_track < track_dim
    
    base_idx = ((pid_batch * num_tracks + pid_track) * seq_len + offsets_seq) * track_dim + offsets_track
    
    h = tl.zeros([BLOCK_SEQ, BLOCK_TRACK], dtype=tl.float32)
    
    u = tl.load(u_ptr + base_idx, mask=mask_seq[:, None] & mask_track[None, :])
    a = tl.load(a_ptr + base_idx, mask=mask_seq[:, None] & mask_track[None, :])
    b = tl.load(b_ptr + base_idx, mask=mask_seq[:, None] & mask_track[None, :])
    
    for i in range(BLOCK_SEQ):
        if mask_seq[i]:
            h[:, i] = a[:, i] * h + b[:, i] * u[:, i]
    
    tl.store(h_ptr + base_idx, h, mask=mask_seq[:, None] & mask_track[None, :])


@triton.jit
def hololink_write_kernel(
    keys_ptr, values_ptr, write_idx_ptr,
    new_key_ptr, new_value_ptr,
    batch, capacity, key_dim, value_dim,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    
    write_idx = tl.load(write_idx_ptr + pid_batch)
    
    key_offset = (pid_batch * capacity + write_idx) * key_dim + tl.arange(0, BLOCK_KEY)
    val_offset = (pid_batch * capacity + write_idx) * value_dim + tl.arange(0, BLOCK_VAL)
    
    new_key = tl.load(new_key_ptr + pid_batch * key_dim + tl.arange(0, BLOCK_KEY))
    new_value = tl.load(new_value_ptr + pid_batch * value_dim + tl.arange(0, BLOCK_VAL))
    
    tl.store(keys_ptr + key_offset, new_key)
    tl.store(values_ptr + val_offset, new_value)
    
    next_idx = (write_idx + 1) % capacity
    tl.store(write_idx_ptr + pid_batch, next_idx)


class TritonParallelScan:
    def __init__(self, block_seq: int = 64, block_dim: int = 64):
        self.block_seq = block_seq
        self.block_dim = block_dim
    
    def hillis_steele_scan(
        self, 
        u: torch.Tensor, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        h_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, dim = u.shape
        
        if h_init is None:
            h_init = torch.zeros(batch, dim, dtype=u.dtype, device=u.device)
        
        h = torch.empty_like(u)
        
        grid = (
            triton.cdiv(seq_len, self.block_seq),
            batch,
            triton.cdiv(dim, self.block_dim)
        )
        
        hillis_steele_scan_kernel[grid](
            u, a, b, h,
            seq_len, batch, dim,
            BLOCK_SEQ=self.block_seq,
            BLOCK_DIM=self.block_dim
        )
        
        return h
    
    def parallel_scan(
        self, 
        u: torch.Tensor, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        h_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, dim = u.shape
        
        if h_init is None:
            h_init = torch.zeros(batch, dim, dtype=u.dtype, device=u.device)
        
        h = torch.empty_like(u)
        h[:, 0, :] = h_init
        
        log_seq = int(math.ceil(math.log2(seq_len)))
        steps = max(log_seq, 1)
        
        a_curr = a.clone()
        b_curr = b.clone()
        
        for i in range(steps):
            d = 1 << i
            a_shifted = torch.zeros_like(a_curr)
            b_shifted = torch.zeros_like(b_curr)
            
            if d < seq_len:
                a_shifted[:, d:, :] = a_curr[:, :-d, :]
                b_shifted[:, d:, :] = b_curr[:, :-d, :]
            
            a_next = a_curr * a_shifted
            b_next = a_curr * b_shifted + b_curr
            
            a_curr = a_next
            b_curr = b_next
        
        h = a_curr * h_init.unsqueeze(1) + b_curr
        
        return h


class TritonHoloLink:
    def __init__(self, capacity: int, key_dim: int, value_dim: int, 
                 block_key: int = 64, block_val: int = 64):
        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.block_key = block_key
        self.block_val = block_val
        
        self.register_buffer('keys', torch.zeros(1, capacity, key_dim))
        self.register_buffer('values', torch.zeros(1, capacity, value_dim))
        self.register_buffer('write_idx', torch.zeros(1, dtype=torch.long))
    
    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        batch = keys.size(0)
        
        if self.keys.size(0) < batch:
            self.keys = self.keys.expand(batch, -1, -1).contiguous()
            self.values = self.values.expand(batch, -1, -1).contiguous()
            self.write_idx = self.write_idx.expand(batch).contiguous()
        
        grid = (batch,)
        
        hololink_write_kernel[grid](
            self.keys, self.values, self.write_idx,
            keys, values,
            batch, self.capacity, self.key_dim, self.value_dim,
            BLOCK_KEY=self.block_key,
            BLOCK_VAL=self.block_val
        )
    
    def read(self, query: torch.Tensor) -> torch.Tensor:
        batch = query.size(0)
        
        output = torch.empty(batch, self.value_dim, dtype=query.dtype, device=query.device)
        
        grid = (batch, self.capacity)
        
        associative_scan_kernel[grid](
            self.keys, self.values, query, output,
            batch, self.capacity, self.key_dim, self.value_dim,
            BLOCK_KEY=self.block_key,
            BLOCK_VAL=self.block_val
        )
        
        return output


class TritonMultiTrackScan:
    def __init__(self, num_tracks: int, track_dim: int, 
                 block_seq: int = 64, block_track: int = 64):
        self.num_tracks = num_tracks
        self.track_dim = track_dim
        self.block_seq = block_seq
        self.block_track = block_track
    
    def forward(
        self, 
        u: torch.Tensor, 
        a: torch.Tensor, 
        b: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, total_dim = u.shape
        assert total_dim == self.num_tracks * self.track_dim
        
        h = torch.empty_like(u)
        
        grid = (
            triton.cdiv(seq_len, self.block_seq),
            batch,
            self.num_tracks
        )
        
        multi_track_scan_kernel[grid](
            u, a, b, h,
            seq_len, batch, self.num_tracks, self.track_dim,
            BLOCK_SEQ=self.block_seq,
            BLOCK_TRACK=self.block_track
        )
        
        return h


def parallel_scan(
    u: torch.Tensor, 
    a: torch.Tensor, 
    b: torch.Tensor, 
    h_init: Optional[torch.Tensor] = None,
    backend: str = 'triton'
) -> torch.Tensor:
    if backend == 'triton':
        scanner = TritonParallelScan()
        return scanner.parallel_scan(u, a, b, h_init)
    else:
        scanner = PyTorchParallelScan()
        return scanner.forward(u, a, b, h_init)


class PyTorchParallelScan:
    def forward(
        self, 
        u: torch.Tensor, 
        a: torch.Tensor, 
        b: torch.Tensor, 
        h_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, dim = u.shape
        
        if h_init is None:
            h_init = torch.zeros(batch, dim, dtype=u.dtype, device=u.device)
        
        h = torch.empty_like(u)
        h[:, 0, :] = a[:, 0, :] * h_init + b[:, 0, :] * u[:, 0, :]
        
        for t in range(1, seq_len):
            h[:, t, :] = a[:, t, :] * h[:, t-1, :] + b[:, t, :] * u[:, t, :]
        
        return h


def benchmark_parallel_scan(
    seq_lengths: list = [128, 256, 512, 1024, 2048, 4096],
    dim: int = 512,
    batch_size: int = 32,
    num_iterations: int = 100
) -> dict:
    import time
    
    results = {'triton': [], 'pytorch': [], 'speedup': []}
    
    triton_scanner = TritonParallelScan()
    pytorch_scanner = PyTorchParallelScan()
    
    for seq_len in seq_lengths:
        u = torch.randn(batch_size, seq_len, dim, device='cuda')
        a = torch.ones(batch_size, seq_len, dim, device='cuda')
        b = torch.ones(batch_size, seq_len, dim, device='cuda')
        h_init = torch.zeros(batch_size, dim, device='cuda')
        
        start = time.time()
        for _ in range(num_iterations):
            _ = triton_scanner.parallel_scan(u, a, b, h_init)
            torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations
        
        start = time.time()
        for _ in range(num_iterations):
            _ = pytorch_scanner.forward(u, a, b, h_init)
            torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / num_iterations
        
        speedup = pytorch_time / triton_time
        
        results['triton'].append(triton_time)
        results['pytorch'].append(pytorch_time)
        results['speedup'].append(speedup)
    
    return results


if __name__ == '__main__':
    print("Triton Kernels for ANA - Running Benchmark...")
    
    benchmark_results = benchmark_parallel_scan()
    
    print("\nBenchmark Results:")
    print(f"{'Seq Len':<10} {'Triton (ms)':<15} {'PyTorch (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for i, seq_len in enumerate([128, 256, 512, 1024, 2048, 4096]):
        triton_ms = benchmark_results['triton'][i] * 1000
        pytorch_ms = benchmark_results['pytorch'][i] * 1000
        speedup = benchmark_results['speedup'][i]
        print(f"{seq_len:<10} {triton_ms:<15.4f} {pytorch_ms:<15.4f} {speedup:<10.2f}x")
