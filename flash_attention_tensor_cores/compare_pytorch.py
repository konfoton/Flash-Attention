import torch
import numpy as np
import os

def compare_results():
    # Dimensions from testing.cu
    B = 1
    H = 128
    L = 256 # 4 * 64
    D = 128

    print(f"Loading data with shape B={B}, H={H}, L={L}, D={D}...")

    # Check if files exist
    files = ["Q.bin", "K.bin", "V.bin", "O_gpu.bin"]
    for f in files:
        if not os.path.exists(f):
            print(f"Error: {f} not found. Please run the C++ benchmark first to generate data.")
            return

    # Load binary files (saved as float32 from C++)
    Q = np.fromfile("Q.bin", dtype=np.float32).reshape(B, H, L, D)
    K = np.fromfile("K.bin", dtype=np.float32).reshape(B, H, L, D)
    V = np.fromfile("V.bin", dtype=np.float32).reshape(B, H, L, D)
    O_gpu = np.fromfile("O_gpu.bin", dtype=np.float32).reshape(B, H, L, D)

    # Convert to PyTorch tensors
    # We use float32 for the reference computation to match the accumulation precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.from_numpy(Q).to(device)
    k = torch.from_numpy(K).to(device)
    v = torch.from_numpy(V).to(device)
    o_gpu = torch.from_numpy(O_gpu).to(device)

    print("Computing PyTorch Flash Attention reference...")

    # Flash Attention requires FP16/BF16 on CUDA
    q_half = q.half()
    k_half = k.half()
    v_half = v.half()

    # Use PyTorch's scaled_dot_product_attention which wraps Flash Attention
    # We force Flash Attention backend if available
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            o_ref = torch.nn.functional.scaled_dot_product_attention(q_half, k_half, v_half)
            print("Using Flash Attention backend.")
    except RuntimeError as e:
        print(f"Flash Attention backend failed or not available: {e}")
        print("Falling back to default implementation (may use Math or MemEfficient)...")
        o_ref = torch.nn.functional.scaled_dot_product_attention(q_half, k_half, v_half)

    # Convert back to float32 for comparison with C++ output
    o_ref = o_ref.float()

    # Compare
    diff = (o_ref - o_gpu).abs()
    mean_error = diff.mean().item()
    max_error = diff.max().item()

    print("-" * 40)
    print(f"Mean Error (GPU vs PyTorch): {mean_error:.6f}")
    print(f"Max Error (GPU vs PyTorch):  {max_error:.6f}")
    print("-" * 40)

    # Detailed check for first few elements
    print("\nFirst 10 elements comparison:")
    o_ref_flat = o_ref.flatten().cpu().numpy()
    o_gpu_flat = o_gpu.flatten().cpu().numpy()
    
    for i in range(10):
        print(f"Idx {i}: PyTorch={o_ref_flat[i]:.6f}, GPU={o_gpu_flat[i]:.6f}, Diff={abs(o_ref_flat[i] - o_gpu_flat[i]):.6f}")

    # Compare CPU implementation (if available)
    # The C++ code doesn't dump O_cpu by default, but if you modified testing.cu to dump it:
    if os.path.exists("O_cpu.bin"):
        O_cpu = np.fromfile("O_cpu.bin", dtype=np.float32).reshape(B, H, L, D)
        o_cpu = torch.from_numpy(O_cpu).to(device)
        
        diff_cpu = (o_ref - o_cpu).abs()
        mean_error_cpu = diff_cpu.mean().item()
        max_error_cpu = diff_cpu.max().item()
        
        print("\n" + "-" * 40)
        print(f"Mean Error (CPU vs PyTorch): {mean_error_cpu:.6f}")
        print(f"Max Error (CPU vs PyTorch):  {max_error_cpu:.6f}")
        print("-" * 40)
    else:
        print("\nO_cpu.bin not found. Skipping CPU vs PyTorch comparison.")

if __name__ == "__main__":
    compare_results()
