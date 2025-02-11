import torch
from exllama_gemm import quantized_matmul

# Create test inputs
a = torch.randn(128, 512, dtype=torch.float16, device='cuda')
b_q = torch.randint(0, 16, (64, 64), dtype=torch.int32, device='cuda')  # 512/8 = 64 rows
b_scale = torch.ones(64, dtype=torch.float16, device='cuda')

# Perform quantized matrix multiplication
result = quantized_matmul(a, b_q, b_scale)

print(result)