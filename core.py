import torch
from torch import Tensor
from einops import rearrange


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor,pe_q = None, attention_mask = None) -> Tensor:
    """
    计算多头自注意力机制。

    参数:
        q (torch.Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        k (torch.Tensor): 键张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        v (torch.Tensor): 值张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        pe (torch.Tensor): 位置编码张量，形状为 (seq_len, head_dim)。
        pe_q (Optional[torch.Tensor]): 查询的位置编码张量。如果为None，则使用相同的 `pe` 作为 `q` 和 `k` 的位置编码。默认为None。
        attention_mask (Optional[torch.Tensor]): 注意力掩码张量，用于屏蔽某些注意力权重。默认为None。

    返回:
        torch.Tensor: 注意力机制的输出，形状为 (batch_size, num_heads, seq_len, head_dim)。
    """
    if pe_q is None:
        # 如果没有单独的位置编码用于查询，则对查询和键应用旋转位置编码（RoPE）
        q, k = apply_rope(q, k, pe) 
        # 计算缩放点积注意力
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=attention_mask) 
        # 重塑输出张量形状为 (batch_size, seq_len, num_heads * head_dim)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x
    else: 
        # 如果有单独的位置编码用于查询，则分别对查询和键应用旋转位置编码
        q, k = apply_rope_qk(q, k, pe_q, pe) 
        # 计算缩放点积注意力
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=attention_mask)
        # 重塑输出张量形状为 (batch_size, seq_len, num_heads * head_dim)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    生成旋转位置编码（RoPE）。

    参数:
        pos (torch.Tensor): 位置张量，形状为 (..., seq_len)。
        dim (int): 位置编码的维度，必须为偶数。
        theta (int): 旋转角度的基数。

    返回:
        torch.Tensor: 旋转位置编码张量，形状为 (..., seq_len, dim, 2, 2)。
    """
    assert dim % 2 == 0
    # 生成缩放因子，形状为 (dim // 2,)
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim  # dim = 16 + 56 + 56 
    # 计算 omega，形状为 (dim // 2,)
    omega = 1.0 / (theta**scale) # 64 omega
    # 计算位置编码的相位部分，形状为 (..., seq_len, dim // 2)
    out = torch.einsum("...n,d->...nd", pos, omega)
    # 计算正弦和余弦，形状为 (..., seq_len, dim // 2, 2)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    # 重塑为 (..., seq_len, dim, 2, 2)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    应用旋转位置编码（RoPE）到查询和键张量。

    参数:
        xq (torch.Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        xk (torch.Tensor): 键张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        freqs_cis (torch.Tensor): 复数形式的旋转位置编码，形状为 (seq_len, head_dim // 2, 2)。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 应用RoPE后的查询和键张量，形状均为 (batch_size, num_heads, seq_len, head_dim)。
    """
    # 重塑查询和键张量以适应复数运算
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2) 
    # 应用RoPE到查询张量
    xq_out = freqs_cis[:, :, :xq_.shape[2], :, :, 0] * xq_[..., 0] + freqs_cis[:, :, :xq_.shape[2], :, :, 1] * xq_[..., 1] 
    # 应用RoPE到键张量
    xk_out = freqs_cis[:, :, :xk_.shape[2], :, :, 0] * xk_[..., 0] + freqs_cis[:, :, :xk_.shape[2], :, :, 1] * xk_[..., 1]
    # 重塑回原始形状
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def apply_rope_qk(xq: Tensor, xk: Tensor, freqs_cis_q: Tensor,freqs_cis_k: Tensor) -> tuple[Tensor, Tensor]:
    """
    应用不同的旋转位置编码（RoPE）到查询和键张量。

    参数:
        xq (torch.Tensor): 查询张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        xk (torch.Tensor): 键张量，形状为 (batch_size, num_heads, seq_len, head_dim)。
        freqs_cis_q (torch.Tensor): 查询的复数形式的旋转位置编码，形状为 (seq_len, head_dim // 2, 2)。
        freqs_cis_k (torch.Tensor): 键的复数形式的旋转位置编码，形状为 (seq_len, head_dim // 2, 2)。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 应用RoPE后的查询和键张量，形状均为 (batch_size, num_heads, seq_len, head_dim)。
    """
    # 重塑查询和键张量以适应复数运算
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2) 
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2) 
    # 应用RoPE到查询张量
    xq_out = freqs_cis_q[:, :, :xq_.shape[2], :, :, 0] * xq_[..., 0] + freqs_cis_q[:, :, :xq_.shape[2], :, :, 1] * xq_[..., 1]  
    # 应用RoPE到键张量
    xk_out = freqs_cis_k[:, :, :xk_.shape[2], :, :, 0] * xk_[..., 0] + freqs_cis_k[:, :, :xk_.shape[2], :, :, 1] * xk_[..., 1] 
    # 重塑回原始形状
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
