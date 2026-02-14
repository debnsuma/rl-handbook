# Chapter 2 - Rollouts: Known Issues & Fixes

## CUBLAS_STATUS_INVALID_VALUE Runtime Error

### Symptom

Running `sample.py` (or the equivalent notebook cells) crashes during the
first `model.generate()` call with:

```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling
`cublasGemmEx(...)` or `cublasSgemmStridedBatched(...)`
```

The error occurs in any linear-layer or batched-matmul operation that uses
half-precision (fp16 / bf16).  It also affects fp32 *batched* matmuls
(`torch.bmm`, `torch.matmul` on 3-D+ tensors) while plain 2-D `torch.mm`
in fp32 still works.

### Root Cause

The environment has a **CUDA toolkit / driver version mismatch** that breaks
the default cuBLAS backend:

| Component | Version |
|---|---|
| PyTorch | 2.10.0+cu128 (compiled against CUDA 12.8) |
| NVIDIA driver | 550.163.01 (exposes CUDA 12.9 runtime) |
| bitsandbytes | 0.49.1 |
| transformers | 5.1.0 |
| GPU | NVIDIA L40S (48 GB) |

PyTorch 2.10 bundles its own `libcublas.so` built for CUDA 12.8.  When the
host driver advertises CUDA 12.9, the **strided-batched GEMM** entry points
in that library return `CUBLAS_STATUS_INVALID_VALUE` for every call.  Plain
single-matrix GEMM (`cublasSgemm`) still works, which is why 2-D fp32
`torch.mm` succeeds while everything else fails.

### Diagnosis Steps

A minimal reproducer that does **not** involve model loading:

```python
import torch

# 2-D fp32 matmul -- uses cublasSgemm, works fine
a = torch.randn(256, 256, device="cuda", dtype=torch.float32)
b = torch.randn(256, 256, device="cuda", dtype=torch.float32)
torch.mm(a, b)  # OK

# 3-D fp32 batched matmul -- uses cublasSgemmStridedBatched, FAILS
a = torch.randn(4, 256, 256, device="cuda", dtype=torch.float32)
b = torch.randn(4, 256, 256, device="cuda", dtype=torch.float32)
torch.bmm(a, b)  # CUBLAS_STATUS_INVALID_VALUE

# Any fp16/bf16 matmul -- uses cublasGemmEx, FAILS
a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
torch.mm(a, b)  # CUBLAS_STATUS_INVALID_VALUE
```

This confirms the issue is at the PyTorch/CUDA level, not in
bitsandbytes, PEFT, or the model itself.

### Fix Applied

PyTorch exposes an alternative BLAS backend, **cublasLt** (cuBLAS Light),
which routes through different entry points that are not affected by the
version mismatch.  Adding one line before any CUDA operation resolves the
problem entirely:

```python
import torch
torch.backends.cuda.preferred_blas_library("cublaslt")
```

This line has been added to the top of `sample.py` and to the relevant
notebook cells.

### Alternative Fixes

If the above workaround is not acceptable, the following would also resolve
the issue:

1. **Reinstall PyTorch matching the driver CUDA version:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu129
   ```
2. **Downgrade the NVIDIA driver** to one that ships CUDA 12.8.
3. **Set the environment variable** (equivalent to the Python call):
   ```bash
   export TORCH_BLAS_PREFER_CUBLASLT=1
   ```
