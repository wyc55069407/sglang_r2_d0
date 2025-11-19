import torch
import torch.nn.functional as F

dev = torch.device("xpu", 3)
torch.xpu.set_device(3)

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = 1 / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    # breakpoint()
    for i in range(num_heads):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    
    
    # breakpoint()

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


query = torch.rand(1, 1, 32, 128, dtype=torch.float32, device=dev)
key = torch.rand(1, 1, 32, 128, dtype=torch.float32, device=dev)
value = torch.rand(1, 1, 32, 128, dtype=torch.float32, device=dev)
g = torch.rand(1, 1, 32, dtype=torch.float32, device=dev)
beta = torch.rand(1, 1, 32, dtype=torch.float32, device=dev)


last_recurrent_state = torch.rand(1, 32, 128, 128, dtype=torch.float32, device=dev)

core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, last_recurrent_state, output_final_state=True, use_qk_l2norm_in_kernel=False
)

#--------------------------------------------------------------------------------------------------------------------------------------

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]  # 128
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)  # 128->64 64
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state  # 128->128 64
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state    # 128->128  64
        core_attn_out[:, :, i] = attn_inter + attn @ v_new  # 64->128 64
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new  # 64->128 128
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


query = torch.rand(1, 256, 32, 128, dtype=torch.float32, device=dev) - 0.5
key = torch.rand(1, 256, 32, 128, dtype=torch.float32, device=dev) - 0.5
value = torch.rand(1, 256, 32, 128, dtype=torch.float32, device=dev) - 0.5
g = (torch.rand(1, 256, 32, dtype=torch.float32, device=dev) - 0.5) / 1000
beta = torch.rand(1, 256, 32, dtype=torch.float32, device=dev) - 0.5

core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
)

core_attn_out2, last_recurrent_state2 = torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=8,
    initial_state=None,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
)


# breakpoint()

last_recurrent_state3 = None
for i in range(256):
    core_attn_out3, last_recurrent_state3 = torch_recurrent_gated_delta_rule(
        query[:,i:i+1,:,:],
        key[:,i:i+1,:,:],
        value[:,i:i+1,:,:],
        g[:,i:i+1,:],
        beta[:,i:i+1,:],
        initial_state=last_recurrent_state3,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

breakpoint()