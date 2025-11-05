import torch
from sgl_kernel_esimd import esimd_kernel_uni

index_score = torch.rand(1, 32768, device="xpu", dtype=torch.float16)


out, outidx = index_score.topk(2048, dim=-1)

index_buf = torch.zeros(1, device="xpu", dtype=torch.int32)
input = index_buf.clone()

total_count = 777
out_ordered = torch.zeros(total_count, device="xpu", dtype=torch.int32)
out_idx = out_ordered.clone()

topk = 2048
esimd_kernel_uni(
    input,
    index_buf,
    out_ordered,
    out_idx,
    input,input,input,input,input,input,
    3322,
    total_count, # HD
    topk,
    0, #seq_len
    0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
)

breakpoint()

print(f"out_ordered should in range {total_count * 2} ~ {total_count * 2 * 2}")
print("torch.min(out_ordered) = ", torch.min(out_ordered))
print("torch.max(out_ordered) = ", torch.max(out_ordered))

print(f"out_idx should in range {total_count * 2 - 500} ~ {total_count * 3 - 500}")
print("torch.min(out_idx) = ", torch.min(out_idx))
print("torch.max(out_idx) = ", torch.max(out_idx))
