#!/usr/bin/env python3
import ctypes as ct
import numpy as np
import os
import time

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lib_tensor = ct.CDLL(os.path.join(ROOT, 'flash_attention_tensor_cores', 'libflash_attention.so'))

lib_cuda_cores = None
lib_naive = None
cuda_path = os.path.join(ROOT, 'flash_attention_cuda_cores', 'libflash_attention_cuda_cores.so')
naive_path = os.path.join(ROOT, 'naive_gpu_attention', 'libnaive_attention.so')
if os.path.exists(cuda_path):
    lib_cuda_cores = ct.CDLL(cuda_path)
else:
    print('[warn] CUDA-cores shared lib not found at', cuda_path)
if os.path.exists(naive_path):
    lib_naive = ct.CDLL(naive_path)
else:
    print('[warn] Naive shared lib not found at', naive_path)

HalfPtr = ct.POINTER(ct.c_uint16)
FloatPtr = ct.POINTER(ct.c_float)

# Signatures
lib_tensor.run_tensor_flash_attention_host_half.argtypes = [
    HalfPtr, HalfPtr, HalfPtr, HalfPtr,
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_void_p,
    FloatPtr,
]
lib_tensor.run_tensor_flash_attention_host_half.restype = None

if lib_cuda_cores is not None:
    lib_cuda_cores.run_cuda_cores_flash_attention_host_half.argtypes = [
        HalfPtr, HalfPtr, HalfPtr, HalfPtr,
        ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_int, ct.c_int,
        ct.c_void_p, FloatPtr,
    ]
    lib_cuda_cores.run_cuda_cores_flash_attention_host_half.restype = None

if lib_naive is not None:
    lib_naive.run_naive_attention_host_half.argtypes = [
        HalfPtr, HalfPtr, HalfPtr, HalfPtr,
        ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_int,
        ct.c_void_p, FloatPtr,
    ]
    lib_naive.run_naive_attention_host_half.restype = None


def as_half_ptr(arr: np.ndarray) -> HalfPtr:
    assert arr.dtype == np.float16 and arr.flags['C_CONTIGUOUS']
    return arr.ctypes.data_as(HalfPtr)


def metrics(ref32: np.ndarray, test16: np.ndarray):
    a = ref32.astype(np.float32).reshape(-1)
    b = test16.astype(np.float32).reshape(-1)
    diff = a - b
    ad = np.abs(diff)
    mae = float(np.mean(ad))
    nz = np.abs(a) > 0
    mre = float(np.mean(np.abs(diff[nz]) / np.abs(a[nz]))) if np.any(nz) else float('nan')
    return mae, mre


def main():
    B, H, L, D = 32, 16, 512, 128
    tile = 64
    KQVO_block_y = 8
    warps = 4

    size = B * H * L * D
    rng = np.random.default_rng(0)
    Q = rng.standard_normal(size, dtype=np.float32).astype(np.float16)
    K = rng.standard_normal(size, dtype=np.float32).astype(np.float16)
    V = rng.standard_normal(size, dtype=np.float32).astype(np.float16)

    O_tc = np.zeros_like(Q)
    O_cc = np.zeros_like(Q)
    O_nv = np.zeros_like(Q)

    O_ref = None
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.device('cuda')
        Q_t = torch.from_numpy(Q.reshape(B,H,L,D)).to(device=device, dtype=torch.float16)
        K_t = torch.from_numpy(K.reshape(B,H,L,D)).to(device=device, dtype=torch.float16)
        V_t = torch.from_numpy(V.reshape(B,H,L,D)).to(device=device, dtype=torch.float16)
        with torch.no_grad():
            scores = torch.matmul(Q_t.to(torch.float32), K_t.transpose(-1, -2).to(torch.float32)) * (1.0/np.sqrt(D))
            attn = torch.softmax(scores, dim=-1)
            O_ref_t = torch.matmul(attn, V_t.to(torch.float32)).to(torch.float16)
        O_ref = O_ref_t.detach().cpu().numpy().reshape(-1)
    else:
        print('[warn] Torch CUDA not available; metrics will be relative to tensor-cores output.')

    elapsed_tc = ct.c_float(0.0)
    lib_tensor.run_tensor_flash_attention_host_half(
        as_half_ptr(Q), as_half_ptr(K), as_half_ptr(V), as_half_ptr(O_tc),
        B, H, L, D, tile, None, ct.byref(elapsed_tc))

    elapsed_cc = ct.c_float(0.0)
    if lib_cuda_cores is not None:
        lib_cuda_cores.run_cuda_cores_flash_attention_host_half(
            as_half_ptr(Q), as_half_ptr(K), as_half_ptr(V), as_half_ptr(O_cc),
            B, H, L, D, KQVO_block_y, warps, None, ct.byref(elapsed_cc))

    elapsed_nv = ct.c_float(0.0)
    if lib_naive is not None:
        lib_naive.run_naive_attention_host_half(
            as_half_ptr(Q), as_half_ptr(K), as_half_ptr(V), as_half_ptr(O_nv),
            B, H, L, D, 0, None, ct.byref(elapsed_nv))

    if O_ref is None:
        ref = O_tc
    else:
        ref = O_ref

    def print_metrics(name, out):
        mae, mre = metrics(ref.astype(np.float32), out)
        print(f"{name}: MAE={mae:.6f} MRE={mre:.6e}")

    print("\nPrecision metrics vs", ('Torch ref' if O_ref is not None else 'Tensor Cores'))
    print_metrics('Tensor Cores', O_tc)
    if lib_cuda_cores is not None:
        print_metrics('CUDA Cores', O_cc)
    if lib_naive is not None:
        print_metrics('Naive', O_nv)

    print("\nSpeed (ms)")
    print(f"Tensor Cores: {elapsed_tc.value:.3f} ms")
    if lib_cuda_cores is not None:
        print(f"CUDA Cores:   {elapsed_cc.value:.3f} ms")
    if lib_naive is not None:
        print(f"Naive:        {elapsed_nv.value:.3f} ms")

if __name__ == '__main__':
    main()
