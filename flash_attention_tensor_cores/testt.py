import ctypes as ct
import numpy as np
import os

# Load the shared library (adjust path if needed)
lib_path = os.path.join(os.getcwd(), "libflash_attention.so")
lib = ct.CDLL(lib_path)

# Define types
# __half* → use POINTER(c_uint16) (binary-compatible with float16)
HalfPtr = ct.POINTER(ct.c_uint16)
FloatPtr = ct.POINTER(ct.c_float)

# Function signatures (extern "C" in wrapper.cuh)
# void run_tensor_flash_attention_host_half(
#     const __half* hQ, const __half* hK, const __half* hV, __half* hO,
#     int B, int H, int L, int D, int tile, cudaStream_t stream = 0, float* elapsed_ms = nullptr);
lib.run_tensor_flash_attention_host_half.argtypes = [
    HalfPtr, HalfPtr, HalfPtr, HalfPtr,
    ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int,
    ct.c_void_p,  # stream (nullptr)
    ct.POINTER(ct.c_float)  # elapsed_ms (nullable)
]
lib.run_tensor_flash_attention_host_half.restype = None

# Prepare inputs
B, H, L, D, tile = 32, 16, 512, 128, 64
size = B * H * L * D

# Create numpy float16 buffers
Q = (np.random.randn(size).astype(np.float16))
K = (np.random.randn(size).astype(np.float16))
V = (np.random.randn(size).astype(np.float16))
O = np.zeros(size, dtype=np.float16)

# Get ctypes pointers (uint16 underlying storage)
Q_p = Q.ctypes.data_as(HalfPtr)
K_p = K.ctypes.data_as(HalfPtr)
V_p = V.ctypes.data_as(HalfPtr)
O_p = O.ctypes.data_as(HalfPtr)

# elapsed time capture (optional)
elapsed = ct.c_float(0.0)
elapsed_p = ct.pointer(elapsed)

# Call the function (stream = None/0)
lib.run_tensor_flash_attention_host_half(
    Q_p, K_p, V_p, O_p,
    B, H, L, D, tile,
    None,
    elapsed_p
)

print("Elapsed ms:", elapsed.value)
print("Output sample:", O[:10])