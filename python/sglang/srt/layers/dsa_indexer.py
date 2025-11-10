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
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


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

    def _get_logits_head_gate(self, x: torch.Tensor):
        weights, _ = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        weights = weights.unsqueeze(-1) * self.softmax_scale
        return weights

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

        return query, key

    def forward_indexer_bs_1(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        topk_indices_list = []

        q_len_start = 0

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
            logits = torch.matmul(query_partial, key.transpose(1, 2))
            logits = logits.reshape(qp_shape[0], qp_shape[1], self.n_heads, logits.shape[-1])

            index_score = (torch.relu(logits) * weights_partial.unsqueeze(-1)).sum(dim=-2)

            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)

        return topk_indices

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
        query, key = self._get_q_k_bf16(q_lora, x, positions)

        if not hasattr(self, "k_cache"):
            # YC WA
            # assume even divide for each batch
            tokens_per_batch = forward_batch.token_to_kv_pool.size // forward_batch.batch_size + forward_batch.batch_size
            print("dsa tokens_per_batch = ", tokens_per_batch)
            self.k_cache = torch.zeros(forward_batch.batch_size, tokens_per_batch, self.head_dim,
                dtype=query.dtype, device=query.device)
        
        # key = key.reshape(forward_batch.batch_size, key.shape[-3] // forward_batch.batch_size, key.shape[-1])

        key = key.squeeze(-2)

        forward_batch.tbo_start_batch_idx = 0
        if forward_batch.req_pool_indices.shape[0] != self.k_cache.shape[0]:
            forward_batch.tbo_start_batch_idx = forward_batch.req_pool_indices[0].item()

        q_len_start = 0
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
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

            # print("layer", layer_id, " batch", ii, " dsa: update k at: ", start_pos, "~", end_pos)

            q_len_start = q_len_end

        if forward_batch.forward_mode.is_extend():
            # do not cal indexer for prefill
            return None

        weights = self._get_logits_head_gate(x)

        topk_result = self.forward_indexer(
            query.contiguous(),
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
