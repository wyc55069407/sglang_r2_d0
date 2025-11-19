from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import add_prefix

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import global_server_args_dict

from sgl_kernel_esimd import esimd_kernel_uni, esimd_kernel_uni_huge_params

# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class V32LayerNorm(nn.Module):
    """
    Layer Normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):

        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x_mean = x.mean(dim=-1, keepdim=True)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = (x - x_mean) * torch.rsqrt(variance + self.eps)
        
        if self.bias is not None:
            x = (x * self.weight + self.bias.to(torch.float32)).to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        return x
        
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)

global topk_indices_final
topk_indices_final = None
global index_score_rsv
index_score_rsv = None
real_total_count_reserved = 160*1024
global out_idx
out_idx = None
global out_ordered
out_ordered = None
class Indexer(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.enable_quant = True

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.k_norm = V32LayerNorm(self.head_dim)
        # NOTE: weight_proj is not quantized
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    def _forward_fake(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ):
        bs = x.shape[0]
        assert self.index_topk == 2048
        ans = torch.arange(0, self.index_topk, dtype=torch.int32, device=x.device)[
            None, ...
        ].repeat(bs, 1)
        if forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.seq_lens_cpu is not None
            )
            which = 0
            for i, (kv_len, qo_len) in enumerate(
                zip(
                    forward_batch.seq_lens_cpu.tolist(),
                    forward_batch.extend_seq_lens_cpu,
                    strict=True,
                )
            ):
                for j in range(kv_len - qo_len, kv_len):
                    ans[which, j + 1 :] = -1
                    which += 1
            assert which == ans.shape[0]
        else:
            assert forward_batch.seq_lens_cpu is not None
            for i, seq_len in enumerate(forward_batch.seq_lens_cpu.tolist()):
                ans[i, seq_len:] = -1

        return ans

    def _get_logits_head_gate(self, x: torch.Tensor, q_scale):
        weights, _ = self.weights_proj(x)
        # weights = weights * self.n_heads**-0.5
        if self.enable_quant:
            weights = weights.unsqueeze(-1) * q_scale.unsqueeze(-1) * self.softmax_scale
        else:
            weights = weights.unsqueeze(-1) * self.softmax_scale  # disable quant
        return weights
    
    def oneline_quant(self, input: torch.Tensor, group_size: int):
        if input.device.type != "meta":
            grouped_input = input.reshape(-1, group_size)
            max_vals, max_indices = grouped_input.abs().max(-1)

            grouped_max = grouped_input[torch.arange(max_indices.numel(), device=input.device), max_indices]
            scales = grouped_max / 127
            scales = scales.abs()
            iscales = torch.nan_to_num(1 / scales).unsqueeze(-1)
            scales = scales.half().reshape((input.size(0), -1))

            qinput = torch.clamp(torch.round(grouped_input * iscales), -128, 127)
            qinput = qinput.to(torch.int8).reshape(input.shape)

        return qinput, scales

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)

        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        key, _ = self.wk(x)
        key = self.k_norm(key)

        key = key.unsqueeze(-2)

        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        query[..., : self.rope_head_dim] = q_rope
        key[..., : self.rope_head_dim] = k_rope

        # query = rotate_activation(query)
        # key = rotate_activation(key)
        query_q, query_scale = self.oneline_quant(query, self.head_dim)
        key_q, key_scale = self.oneline_quant(key, self.head_dim)

        # breakpoint()

        # deq_query = query_q * query_scale.unsqueeze(-1)
        # deq_key = key_q * key_scale.unsqueeze(-1)

        # query = deq_query
        # key = deq_key
        if self.enable_quant:
            return query_q, key_q, query_scale, key_scale
        else:
            return query, key, query_scale, key_scale # disable quant
        # breakpoint()

        # return query, key
    
    def mega_dsa_forward(self, x, q_lora, positions, kv_indptr, kv_indices, forward_batch):

        kv_indptr_updated = torch.empty_like(kv_indptr)
        kv_indices_new = torch.empty(forward_batch.batch_size * 2048, device=kv_indices.device, dtype=kv_indices.dtype)
        
        max_q_len = 0
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            if seq_len > max_q_len:
                max_q_len = seq_len
        forward_batch.tbo_start_batch_idx = 0
        if forward_batch.req_pool_indices.shape[0] != self.k_cache.shape[0]:
            forward_batch.tbo_start_batch_idx = forward_batch.req_pool_indices_cpu[0].item()
        seq_lens_cpu_addr = forward_batch.seq_lens_cpu.data_ptr()

        knorm_weight = self.k_norm.weight
        knorm_bias = self.k_norm.bias
        knorm_eps = self.k_norm.eps
        tokens_per_batch = self.k_cache.shape[1]
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        k_cache_out = self.k_cache
        k_scale_out = self.k_scale

        real_total_count = max_q_len
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

        global out_idx
        global out_ordered
        topk = self.index_topk
        if out_idx is None:
            out_ordered = torch.zeros(8, 10, topk, device=x.device, dtype=torch.float16)
            out_idx = torch.zeros(8, 10, topk, device=x.device, dtype=torch.uint32)
        global topk_indices_final
        if topk_indices_final is None:
            # assume max batch is 8
            topk_indices_final = torch.zeros(8, topk, device=x.device, dtype=torch.uint32)
        
        global index_score_rsv
        if index_score_rsv is None:
            index_score_rsv = torch.zeros(8, real_total_count_reserved, device=x.device, dtype=torch.float16) - 65504

        batch_num = forward_batch.batch_size
        query = torch.empty(batch_num, self.n_heads, self.head_dim, device=x.device, dtype=torch.float16)
        key = torch.empty(batch_num, self.head_dim, device=x.device, dtype=torch.float16)
        
        query_out = torch.empty(query.shape, device=query.device, dtype=torch.int8)
        q_scale = torch.empty(query.shape[0], query.shape[1], device=query.device, dtype=torch.float16)

        weights = torch.empty(q_scale.shape[0], self.weights_proj.weight.shape[0], device=x.device, dtype=x.dtype)

        esimd_kernel_uni_huge_params(
            x,
            q_lora,
            self.wq_b.weight,
            self.wq_b.weight_scale_inv,
            self.wk.weight,
            self.wk.weight_scale_inv,
            query,
            key,
            knorm_weight,
            knorm_bias,
            cos_sin_cache,
            positions,
            k_cache_out,
            k_scale_out,
            query_out,
            q_scale,
            weights,
            self.weights_proj.weight,
            topk_indices_final,
            index_score_rsv,
            out_idx,
            out_ordered,
            kv_indptr,
            kv_indices,
            kv_indptr_updated,
            kv_indices_new,
            2000,
            batch_num,
            max_q_len,
            forward_batch.tbo_start_batch_idx,
            seq_lens_cpu_addr,
            tokens_per_batch,
            total_count_stride,
            real_total_count_reserved,
            groups, 0,
            knorm_eps,
            self.softmax_scale, 1.0, 1.0, 1.0
        )

        return kv_indptr_updated, kv_indices_new
        

    def forward_indexer_bs_1(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:

        # topk_indices_list = []
        global topk_indices_final
        if topk_indices_final is None:
            # assume max batch is 8
            topk_indices_final = torch.zeros(8, topk, device=query.device, dtype=torch.uint32)
        
        global index_score_rsv
        if index_score_rsv is None:
            index_score_rsv = torch.zeros(8, real_total_count_reserved, device=query.device, dtype=torch.float16) - 65504

        q_len_start = 0
        max_q_len = 0

        dsa_qk_attn_fuse = True
        if dsa_qk_attn_fuse:
            # seq_lens_cpu = forward_batch.seq_lens.to("cpu")
            seq_lens_cpu_addr = forward_batch.seq_lens_cpu.data_ptr()

            # uint8_t* query,    // [b, 64, 128]
            # uint8_t* k_cache,  // [b, k_cache_batch_stride, 128]
            # uint8_t* k_scale,  // [b, k_cache_batch_stride]
            # uint8_t* weights,  // [b, 1, 64]
            # uint8_t* index_score_rsv,
            # int64_t batch_num,
            # int64_t k_cache_batch_stride,
            # int64_t seq_lens_cpu_addr,  // [3260, 5136,   10,  778]
            k_cache_batch_stride = self.k_cache.shape[-2]
            # index_score_rsv_int = torch.zeros(16384, 64, device=query.device, dtype=torch.float16) - 65504
            index_score_rsv_int = index_score_rsv[forward_batch.tbo_start_batch_idx:]

            esimd_kernel_uni(
                    query,
                    self.k_cache,
                    self.k_scale,
                    weights, 
                    index_score_rsv_int, 
                    index_score_rsv_int, index_score_rsv_int, index_score_rsv_int, index_score_rsv_int, index_score_rsv_int,
                    4004,
                    forward_batch.batch_size,
                    k_cache_batch_stride,
                    seq_lens_cpu_addr, 
                    0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
                )

            # breakpoint()

            for i in range(forward_batch.batch_size):
                seq_len = forward_batch.seq_lens_cpu[i].item()
                if seq_len > max_q_len:
                    max_q_len = seq_len
        else:
            for i in range(forward_batch.batch_size):
                seq_len = forward_batch.seq_lens[i].item()
                q_len = (
                    forward_batch.extend_seq_lens_cpu[i]
                    if forward_batch.forward_mode.is_extend()
                    else 1
                )
                q_len_end = q_len_start + q_len

                query_partial = query[q_len_start:q_len_end]
                query_partial = query_partial.unsqueeze(0).contiguous()

                weights_partial = weights[q_len_start:q_len_end]
                weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

                start_pos = 0  # always start from 0 for k cache
                end_pos = seq_len

                ii = i + forward_batch.tbo_start_batch_idx

                key = self.k_cache[ii:ii+1, start_pos:end_pos]
                k_scale = self.k_scale[ii:ii+1, start_pos:end_pos]
                # index_score = fp8_index(
                #     query_partial,
                #     weights_partial,
                #     key
                # )
                """
                Perform index score using FP8 precision.

                Args:
                    q (torch.Tensor): The Q tensor, must be contiguous.
                    q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
                    k (torch.Tensor): The K tensor, must be contiguous.
                    k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

                    fp8 q @ fp8 k -> fp32 logits
                    relu(fp32 logits) * q_s (weights) -> fp32 logits
                    fp32 logits -> fp32 logits_sum
                    fp32 logits_sum * k_s (e8m0) -> fp32 index_score
                """
                qp_shape = query_partial.shape
                query_partial = query_partial.reshape(qp_shape[0], qp_shape[1]*qp_shape[2], qp_shape[3])

                query_partial = query_partial.to(torch.float32)
                key = key.transpose(1, 2).to(torch.float32)
                
                logits = torch.matmul(query_partial, key)
                logits = logits.reshape(qp_shape[0], qp_shape[1], self.n_heads, logits.shape[-1])

                if self.enable_quant:
                    logits = logits * k_scale
                logits = logits.to(torch.float16)

                index_score_rsv[i, :seq_len] = (torch.relu(logits) * weights_partial.unsqueeze(-1)).sum(dim=-2)

                # breakpoint()
                
                # index_score = (torch.relu(logits) * weights_partial.unsqueeze(-1)).sum(dim=-2)
                if seq_len > max_q_len:
                    max_q_len = seq_len
                
                q_len_start = q_len_end
        
        topk_fuse_opt = True
        if topk_fuse_opt:
            real_total_count = max_q_len
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
            # real_total_count_stride_aligned = groups * total_count_stride
            
            global out_idx
            global out_ordered
            if out_idx is None:
                out_ordered = torch.zeros(8, 10, topk, device=query.device, dtype=torch.float16)
                out_idx = torch.zeros(8, 10, topk, device=query.device, dtype=torch.uint32)

            output_final_out = 0
            esimd_kernel_uni(
                index_score_rsv[forward_batch.tbo_start_batch_idx:],
                index_score_rsv[forward_batch.tbo_start_batch_idx:],
                out_ordered,
                out_idx, 
                out_ordered, 
                topk_indices_final, out_ordered, out_ordered, out_ordered, out_ordered,
                3311,
                total_count_stride,
                groups,
                topk, 
                output_final_out, 
                forward_batch.batch_size, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0,
            )
        else:
            for i in range(forward_batch.batch_size):
                seq_len = forward_batch.seq_lens[i].item()
                end_pos = seq_len

                index_score = index_score_rsv[i, :seq_len]
                
                topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

                pad_len = align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
                topk_indices = torch.nn.functional.pad(
                    topk_indices, (0, pad_len), "constant", -1
                )

                topk_indices_final[i] = topk_indices
                # topk_indices_list.append(topk_indices)

        # topk_indices = torch.cat(topk_indices_list, dim=0)
        # breakpoint()

        return topk_indices_final

    def forward_indexer(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self.forward_indexer_bs_1(query, weights, forward_batch, topk, layer_id)

    def _forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not hasattr(self, "k_cache"):
            k_dtype = x.dtype
            if self.enable_quant:
                k_dtype = torch.int8
            # YC WA
            # assume even divide for each batch
            tokens_per_batch = forward_batch.token_to_kv_pool.size // forward_batch.batch_size + forward_batch.batch_size
            print("dsa tokens_per_batch = ", tokens_per_batch)
            self.k_cache = torch.zeros(forward_batch.batch_size, tokens_per_batch, self.head_dim,
                dtype=k_dtype, device=x.device)
            self.k_scale = torch.zeros(forward_batch.batch_size, tokens_per_batch, dtype=x.dtype, device=x.device)
        
        opt_rope_quant_updatek = True
        if not forward_batch.forward_mode.is_extend() and self.enable_quant and opt_rope_quant_updatek:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            key, _ = self.wk(x)

            forward_batch.tbo_start_batch_idx = 0
            if forward_batch.req_pool_indices.shape[0] != self.k_cache.shape[0]:
                forward_batch.tbo_start_batch_idx = forward_batch.req_pool_indices_cpu[0].item()
            seq_lens_cpu_addr = forward_batch.seq_lens_cpu.data_ptr()

            if True:
                knorm_weight = self.k_norm.weight
                knorm_bias = self.k_norm.bias
                knorm_eps = self.k_norm.eps
                tokens_per_batch = self.k_cache.shape[1]
                cos_sin_cache = self.rotary_emb.cos_sin_cache
                k_cache_out = self.k_cache
                k_scale_out = self.k_scale
                query_out = torch.empty(query.shape, device=query.device, dtype=torch.int8)
                q_scale = torch.empty(query.shape[0], query.shape[1], device=query.device, dtype=torch.float16)

                esimd_kernel_uni(
                    query, key,
                    knorm_weight, knorm_bias,
                    positions, cos_sin_cache,
                    query_out, q_scale, k_cache_out, k_scale_out,
                    3991,
                    seq_lens_cpu_addr,
                    forward_batch.tbo_start_batch_idx, 
                    tokens_per_batch, 
                    forward_batch.batch_size, 0, 0, 0, 0, 0,
                    knorm_eps, 1.0, 1.0, 1.0, 1.0,
                )
                # breakpoint()

                query = query_out
            else:
                # input query 
                # fuse ----------------------------------
                q_rope, _ = torch.split(query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
                key = self.k_norm(key)
                key = key.unsqueeze(-2)
                k_rope, _ = torch.split(
                    key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
                )
                q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

                query[..., : self.rope_head_dim] = q_rope
                key[..., : self.rope_head_dim] = k_rope

                query_q, query_scale = self.oneline_quant(query, self.head_dim)
                key_q, key_scale = self.oneline_quant(key, self.head_dim)

                query = query_q
                key = key_q
                q_scale = query_scale
                k_scale = key_scale

                key = key.squeeze(-2)
                k_scale = k_scale.squeeze(-1)

                q_len_start = 0
                for i in range(forward_batch.batch_size):
                    seq_len = forward_batch.seq_lens_cpu[i].item()
                    q_len = (
                        forward_batch.extend_seq_lens_cpu[i]
                        if forward_batch.forward_mode.is_extend()
                        else 1
                    )

                    q_len_end = q_len_start + q_len

                    start_pos = seq_len - q_len
                    end_pos = seq_len

                    ii = i + forward_batch.tbo_start_batch_idx

                    self.k_cache[ii:ii+1, start_pos:end_pos] = key[q_len_start:q_len_end]
                    self.k_scale[ii:ii+1, start_pos:end_pos] = k_scale[q_len_start:q_len_end]

                    print("layer", layer_id, " batch", ii, " dsa: update k at: ", start_pos, "~", end_pos)

                    q_len_start = q_len_end
                # fuse ----------------------------------

                breakpoint()
        else:
            query, key, q_scale, k_scale = self._get_q_k_bf16(q_lora, x, positions)

            key = key.squeeze(-2)
            k_scale = k_scale.squeeze(-1)

            forward_batch.tbo_start_batch_idx = 0
            if forward_batch.req_pool_indices.shape[0] != self.k_cache.shape[0]:
                forward_batch.tbo_start_batch_idx = forward_batch.req_pool_indices_cpu[0].item()
            # print("forward_batch.tbo_start_batch_idx = ", forward_batch.tbo_start_batch_idx)

            q_len_start = 0
            for i in range(forward_batch.batch_size):
                seq_len = forward_batch.seq_lens_cpu[i].item()
                q_len = (
                    forward_batch.extend_seq_lens_cpu[i]
                    if forward_batch.forward_mode.is_extend()
                    else 1
                )

                q_len_end = q_len_start + q_len

                start_pos = seq_len - q_len
                end_pos = seq_len

                ii = i + forward_batch.tbo_start_batch_idx

                self.k_cache[ii:ii+1, start_pos:end_pos] = key[q_len_start:q_len_end]
                self.k_scale[ii:ii+1, start_pos:end_pos] = k_scale[q_len_start:q_len_end]

                # print("layer", layer_id, " batch", ii, " dsa: update k at: ", start_pos, "~", end_pos)

                q_len_start = q_len_end

        if forward_batch.forward_mode.is_extend():
            global index_score_rsv
            if index_score_rsv is not None and layer_id == 0:
                index_score_rsv[...] = -65504.
            # do not cal indexer for prefill
            return None

        opt_weight_qscale_fuse = True
        if opt_weight_qscale_fuse:
            weights = torch.empty(q_scale.shape[0], self.weights_proj.weight.shape[0], device=x.device, dtype=x.dtype)
            esimd_kernel_uni(
                x, self.weights_proj.weight, q_scale, weights,
                weights, weights, weights, weights, weights, weights,
                3876,
                q_scale.shape[0], # input_len
                0, 0, 0, 0, 0, 0, 0, 0, self.softmax_scale, 1.0, 1.0, 1.0, 1.0,
            )
        else:
            weights = self._get_logits_head_gate(x, q_scale)
            
            assert len(weights.shape) == 3
            weights = weights.squeeze(-1)

        topk_result = self.forward_indexer(
            query,
            weights,
            forward_batch,
            topk=self.index_topk,
            layer_id=layer_id,
        )

        return topk_result

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        return self._forward(x, q_lora, positions, forward_batch, layer_id)
