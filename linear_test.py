# # %%
# from exllamav2 import ext
# import torch
# q_weight = torch.randint(0, 2**16, (640, 4096), dtype=torch.int32).cuda()
# q_scale = torch.randint(0, 2**16, (32, 512), dtype=torch.int32).cuda()
# q_perm = torch.arange(0, 4096, dtype=torch.int32).cuda()
# # q_invperm should be inverse of q_perm
# q_invperm = torch.argsort(q_perm)
# q_scale_max = torch.randn(32, dtype=torch.float16).cuda()
# q_groups = torch.tensor([5, 0, 5, 20, 5, 40, 5, 60, 5, 80, 5, 100, 5, 120, 5, 140, 5, 160, 5, 180, 5, 200, 5, 220, 5, 240, 5, 260, 5, 280, 5, 300, 5, 320, 5, 340, 5, 360, 5, 380, 5, 400, 5, 420, 5, 440, 5, 460, 5, 480, 5, 500, 5, 520, 5, 540, 5, 560, 5, 580, 5, 600, 5, 620], dtype=torch.int16).cuda()

# w = {
#     "q_weight": q_weight,
#     "q_scale": q_scale,
#     "q_groups": q_groups,
#     "q_perm": q_perm,
#     "q_invperm": q_invperm,
#     "q_scale_max": q_scale_max
# }
# temp_dq = torch.zeros(16777216, dtype=torch.float16).cuda()
# q_handle = ext.make_q_matrix(w,
#                             temp_dq,
#                             prescale = 1,
#                             max_dq_rows = 131072,
#                             offset_qzeros = False)
# # %%
# import time
# from exllamav2.ext import exllamav2_ext as ext_c
# # input shape: 16x14848x4096
# hidden_states = torch.randn(16,14848, 4096, dtype=torch.float16).cuda()
# hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
# output_shape = hidden_states.shape[:-1] + (4096,)
# output = torch.empty((hidden_states.shape[0], 4096), dtype=torch.half).cuda()
# start_time = time.time()
# ext_c.gemm_half_q_half(hidden_states, q_handle, output, True)
# end_time = time.time()
# print(f"Time taken: {end_time - start_time} seconds")
# %%
from exllamav2 import ext
import torch
from exllamav2.ext import exllamav2_ext as ext_c
torch.manual_seed(0)

def make_q_group(input_features, pack_size):
    q_groups = []
    unit_size = pack_size * 4
    for i in range(input_features // unit_size):
        q_groups.append(pack_size)
        q_groups.append(i * unit_size)
    return torch.tensor(q_groups, dtype=torch.int16).cuda()

def kernel_latency(m,n,k,bit=4,kernel=None, warmup_times=4, run_times=10, fast_flush=False):
    q_weight = torch.randint(0, 2**16, ((k//32)*bit, n), dtype=torch.int32).cuda()
    q_scale = torch.randint(0, 2**16, (k//128, n//8), dtype=torch.int32).cuda()
    # q_perm = torch.arange(0, 4096, dtype=torch.int32).cuda()
    q_perm = None
    # q_invperm should be inverse of q_perm
    # q_invperm = torch.argsort(q_perm)
    q_invperm = None
    q_scale_max = torch.randn(k//128, dtype=torch.float16).cuda()
    q_groups = make_q_group((k//32)*bit, bit)
    w = {
        "q_weight": q_weight,
        "q_scale": q_scale,
        "q_groups": q_groups,
        "q_scale_max": q_scale_max
    }
        # "q_perm": q_perm,
        # "q_invperm": q_invperm,
    temp_dq = torch.zeros(n*k, dtype=torch.float16).cuda()
    q_handle = ext.make_q_matrix(w,
                                temp_dq,
                                prescale = 0,
                                max_dq_rows = 536870912//n,
                                offset_qzeros = False)
    hidden_states = torch.ones(m, k, dtype=torch.float16).cuda()
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    output = torch.empty((hidden_states.shape[0], n), dtype=torch.half).cuda()
    for _ in range(warmup_times):
        ext_c.gemm_half_q_half(hidden_states, q_handle, output, True)
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Estimate the runtime of the function
    output = torch.empty((hidden_states.shape[0], n), dtype=torch.half).cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(run_times):
        if fast_flush:
            cache.zero_()
        ext_c.gemm_half_q_half(hidden_states, q_handle, output, True)
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / run_times

    return estimate_ms

shapes = [
            # [64, 4096*4, 4096*2],
            # [64, 4096*2, 4096],
            # [32, 4096*4, 4096*2],
            # [32, 4096*2, 4096],
            # [16, 4096*4, 4096*2],
            # [16, 4096*2, 4096],
            # [8, 4096*4, 4096*2],
            # [8, 4096*2, 4096],
            # [1, 4096*4, 4096*2],
            # [1, 4096*2, 4096],
            [32, 8192, 22016],
            [16, 8192, 22016],
            [8, 8192, 22016],
            [4, 8192, 22016],
            [2, 8192, 22016],
            [1, 8192, 22016],
]

# batch = 4
for m, n, k in shapes:
    lat = kernel_latency(m,n,k)

    flops = 1 * m *n * k * 2 / lat / 1e9
    print(f"m={m}, n={n}, k={k}, flops={flops: .4f} TFLOPs, lat={lat:.4f} ms")
# %%
import torch

def kernel_latency(kernel, warmup_times=5, run_times=10, fast_flush=False):
    for _ in range(warmup_times):
        kernel()

    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(run_times):
        if fast_flush:
            cache.zero_()
        kernel()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / run_times

    return estimate_ms
"""
        cute::tuple<int, int, int>(4096, 5120 / 4, 3584),
        cute::tuple<int, int, int>(4096 * 8, 5120 / 4, 3584),
        cute::tuple<int, int, int>(4096, 5120 / 4 * 8, 3584),

        cute::tuple<int, int, int>(4096, 3584, 2560 / 4),
        cute::tuple<int, int, int>(4096 * 8, 3584, 2560 / 4),

"""
# shapes = [
#             [8, 4096*4, 4096*2],
#             [8, 4096*2, 4096],
#             [7, 4096*4, 4096*2],
#             [7, 4096*2, 4096],
#             [6, 4096*4, 4096*2],
#             [6, 4096*2, 4096],
#             [5, 4096*4, 4096*2],
#             [5, 4096*2, 4096],
#             [4, 4096*4, 4096*2],
#             [4, 4096*2, 4096],
#             [3, 4096*4, 4096*2],
#             [3, 4096*2, 4096],
#             [2, 4096*4, 4096*2],
#             [2, 4096*2, 4096],
#             [1, 4096*4, 4096*2],
#             [1, 4096*2, 4096],
#         ]

batch = 1
for m, n, k in shapes:
    a = torch.randn(batch, m, k, dtype=torch.half, device=torch.device("cuda"))
    b = torch.randn(batch, n, k, dtype=torch.half, device=a.device)

    kernel = lambda: torch.matmul(a, b.transpose(1, 2))
    lat = kernel_latency(kernel)

    flops = batch * m *n * k * 2 / lat / 1e9
    print(f"m={m}, n={n}, k={k}, flops={flops: .4f} TFLOPs, lat={lat:.4f} ms")
# %%
