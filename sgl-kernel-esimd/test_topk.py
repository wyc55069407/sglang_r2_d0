import torch
from sgl_kernel_esimd import esimd_kernel_uni

dev = torch.device("xpu", 3)

torch.xpu.set_device(3)

batch_n = 2
real_total_count = 32667
real_total_count_reserved = 160*1024
groups = 1
if real_total_count > 20480:
    groups = 4
if real_total_count > 30720:
    groups = 6
if real_total_count > 40960:
    groups = 8
if real_total_count > 51200:
    groups = 10
total_count_stride = ((real_total_count // groups) + 4095) // 4096 * 4096
real_total_count_stride_aligned = groups * total_count_stride
topk = 2048

# index_score = torch.rand(groups, total_count, device=dev, dtype=torch.float16) * 10
# index_score_int = torch.randint(low=0, high=65535, size=(groups, total_count), device=dev, dtype=torch.uint16)

index_score_rsv = torch.zeros(batch_n, real_total_count_reserved, device=dev, dtype=torch.float16) - 65504
index_score_int_rsv = torch.zeros(batch_n, real_total_count_reserved, device=dev, dtype=torch.uint16)

index_score_rsv[:,:real_total_count] = torch.rand(batch_n, real_total_count, device=dev, dtype=torch.float16) * 10 - 5
index_score_int_rsv[:,:real_total_count] = torch.randint(low=0, high=65535, size=(batch_n, real_total_count), device=dev, dtype=torch.uint16)[0]

index_score = index_score_rsv[:,:real_total_count_stride_aligned].view(batch_n, groups, total_count_stride)
index_score_int = index_score_int_rsv[:,:real_total_count_stride_aligned].view(batch_n, groups, total_count_stride)

out, outidx = index_score.topk(2048, dim=-1)

# input = index_score
# input_idx = torch.arange(real_total_count_reserved, device=dev).to(torch.uint32)

out_ordered =  torch.zeros(batch_n, 10, topk, device=dev, dtype=index_score.dtype)
out_idx = torch.zeros(batch_n, 10, topk, device=dev, dtype=torch.uint32)

final_out_ordered =  torch.zeros(batch_n, topk, device=dev, dtype=index_score.dtype)
final_out_idx = torch.zeros(batch_n, topk, device=dev, dtype=torch.uint32)

output_final_out = 1


breakpoint()

esimd_kernel_uni(
    index_score_rsv,
    index_score_rsv,
    out_ordered,
    out_idx, 
    final_out_ordered, 
    final_out_idx, out_idx, out_idx, out_idx, out_idx,
    3311,
    total_count_stride,
    groups,
    topk, 
    output_final_out,
    batch_n, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
)

breakpoint()

dl = []
out_list = []
buf_range = 30
for i in range(buf_range):
    dl.append([index_score_rsv.clone(),
        index_score_rsv.clone(), final_out_ordered.clone(), final_out_idx.clone()])

# breakpoint()

import time
time.sleep(5)

cnt = 10000
torch.xpu.synchronize()
tic = time.perf_counter()
for _ in range(cnt):
    # dummyBuf[...] = 2
    idx = _%buf_range
    esimd_kernel_uni(
        dl[idx][0],
        dl[idx][1],
        out_ordered,
        out_idx, 
        dl[idx][2], 
        dl[idx][3], out_idx, out_idx, out_idx, out_idx,
        3311,
        total_count_stride,
        groups,
        topk, output_final_out, batch_n, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    # out, outidx = dl[idx][0].topk(2048, dim=-1)

torch.xpu.synchronize()
latency = time.perf_counter() - tic

print("total latency is ", latency*1000, "ms")
print("each latency is ", latency*1000/cnt, "ms")
# breakpoint()