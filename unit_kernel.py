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

def kernel_latency(m,n,k,bit=4,kernel=None, warmup_times=4, run_times=3, fast_flush=False):
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
"""
        cute::tuple<int, int, int>(4096, 5120 / 4, 3584),
        cute::tuple<int, int, int>(4096 * 8, 5120 / 4, 3584),
        cute::tuple<int, int, int>(4096, 5120 / 4 * 8, 3584),

        cute::tuple<int, int, int>(4096, 3584, 2560 / 4),
        cute::tuple<int, int, int>(4096 * 8, 3584, 2560 / 4),

"""
shapes = [
          # [4096, 5120 // 4, 3584],
          # [4096 * 8, 5120 // 4, 3584],
          # [4096, 5120 // 4 * 8, 3584],
          # [4096, 3584, 2560 // 4],
          # [4096 * 8, 3584, 2560 // 4],
            # [30, 1280, 3584],
            # [30, 1280, 640],
            # [30, 1280 * 8, 3584],
            # [30, 4096, 4096],
            # [30, 12288, 4096],
            # [30, 22016, 4096],
            # [30, 4096, 11008],
            # [4096, 4096, 4096],
            # [5120, 5120, 5120],
            # [8192, 8192, 8192],
            # [5120, 8192, 4096],
            # [5120, 6400, 5120],
            # [5120, 5120, 5120],
            # [5129, 256, 4096],
            # [16384, 16384, 16384],
            
            [16, 14848, 8192],
            [16, 8192, 7424],
            # [16, 4096, 4096*2],
            # [16, 4096*2, 4096*2],
            # [16, 4096*4, 4096*2],
        ]

# batch = 4
for m, n, k in shapes:
    # a = torch.randn(batch, m, k, dtype=torch.half, device=torch.device("cuda"))
    # b = torch.randn(batch, n, k, dtype=torch.half, device=a.device)
    # batch = 4
    # m = 16
    # n = 14848
    # k = 8192
    # kernel = lambda: torch.matmul(a, b.transpose(1, 2))
    lat = kernel_latency(m,n,k)

    flops = 1 * m *n * k * 2 / lat / 1e9
    print(f"{flops: .4f} TFLOPs {lat:.4f} ms")