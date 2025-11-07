import torch
from sgl_kernel_esimd import esimd_kernel_uni

dev = torch.device("xpu", 3)

torch.xpu.set_device(3)

real_total_count = 43337
real_total_count_reserved = 160*1024
total_count_stride = 20480
groups = (real_total_count + total_count_stride - 1) // total_count_stride
real_total_count_stride_aligned = groups * total_count_stride
topk = 2048

# index_score = torch.rand(groups, total_count, device=dev, dtype=torch.float16) * 10
# index_score_int = torch.randint(low=0, high=65535, size=(groups, total_count), device=dev, dtype=torch.uint16)

index_score_rsv = torch.zeros(real_total_count_reserved, device=dev, dtype=torch.float16) * 10
index_score_int_rsv = torch.zeros(real_total_count_reserved, device=dev, dtype=torch.uint16)

index_score_rsv[:real_total_count] = torch.rand(real_total_count, device=dev, dtype=torch.float16) * 10
index_score_int_rsv[:real_total_count] = torch.randint(low=0, high=65535, size=(1, real_total_count), device=dev, dtype=torch.uint16)[0]

index_score = index_score_rsv[:real_total_count_stride_aligned].view(groups, total_count_stride)
index_score_int = index_score_int_rsv[:real_total_count_stride_aligned].view(groups, total_count_stride)

out, outidx = index_score.topk(2048, dim=-1)

input = index_score
input_idx = torch.arange(real_total_count_reserved, device=dev).to(torch.uint32)

out_ordered =  torch.zeros(groups, topk, device=dev, dtype=input.dtype)
out_idx = torch.zeros(groups, topk, device=dev, dtype=torch.uint32)

debug_buf = torch.ones(64, 16, device=dev, dtype=torch.uint32)
debug_buf1 = torch.ones(16, 4, device=dev, dtype=torch.uint32)
debug_buf2 = torch.ones(64, 16, device=dev, dtype=torch.uint32)

esimd_kernel_uni(
    input,
    input_idx,
    debug_buf,
    debug_buf1,
    debug_buf2,
    out_ordered,
    out_idx, out_idx, out_idx, out_idx,
    3311,
    total_count_stride,
    groups,
    topk, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
)

breakpoint()

dl = []
out_list = []
buf_range = 100
for i in range(buf_range):
    dl.append([input.clone(),
        input_idx.clone(),
        out_ordered.clone(),
        out_idx.clone()])

breakpoint()

import time
cnt = 10000
torch.xpu.synchronize()
tic = time.perf_counter()
for _ in range(cnt):
    # dummyBuf[...] = 2
    idx = _%buf_range
    esimd_kernel_uni(
        dl[idx][0],
        dl[idx][1],
        debug_buf,
        debug_buf1,
        debug_buf2,
        dl[idx][2],
        dl[idx][3], dl[idx][3], dl[idx][3], dl[idx][3],
        3311,
        total_count_stride,
        groups,
        topk, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    # out, outidx = dl[idx][0].topk(2048, dim=-1)

torch.xpu.synchronize()
latency = time.perf_counter() - tic

print("total latency is ", latency*1000, "ms")
print("each latency is ", latency*1000/cnt, "ms")
# breakpoint()