import math
from dataclasses import dataclass
import torch
from einops import rearrange
from torch import Tensor, nn

from core import attention, rope,apply_rope


class EmbedND(nn.Module):
    """
    多维嵌入模块，用于生成多维位置嵌入。

    参数:
        dim (int): 嵌入的维度。
        theta (int): 旋转角度的基数，用于生成旋转位置编码（RoPE）。
        axes_dim (List[int]): 每个轴的维度列表，用于生成每个轴的位置编码。
    """
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        # 嵌入的维度
        self.dim = dim
        # 旋转角度的基数
        self.theta = theta
        # 每个轴的维度列表
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        前向传播函数，用于生成多维位置嵌入。

        参数:
            ids (torch.Tensor): 输入的索引张量，形状为 (..., n_axes)。

        返回:
            torch.Tensor: 生成的多维位置嵌入，形状为 (..., n_axes, dim, 2, 2)。
        """
        # 获取轴的数量
        n_axes = ids.shape[-1]
        # 对每个轴生成旋转位置编码，并沿最后一个轴拼接
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # 在第二个维度上添加一个维度，并返回
        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    生成时间步的正弦位置嵌入。

    参数:
        t (torch.Tensor): 一个1维张量，包含N个时间步索引，每个批次元素一个。这些可能是分数。
        dim (int): 输出的维度。
        max_period (int): 控制嵌入的最小频率，默认为10000。
        time_factor (float): 时间因子，用于调整时间步的尺度，默认为1000.0。

    返回:
        torch.Tensor: 位置嵌入张量，形状为 (N, D)。
    """
    # 调整时间步的尺度
    t = time_factor * t
    # 计算一半的维度
    half = dim // 2
    # 生成频率张量，形状为 (half,)
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    # 计算角度
    args = t[:, None].float() * freqs[None]
    # 生成正弦和余弦嵌入，并拼接，形状为 (N, dim)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        # 如果维度为奇数，则在末尾添加一个零张量，形状为 (N, 1)
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        # 如果输入张量是浮点类型，则将嵌入张量转换为相同的类型
        embedding = embedding.to(t)
    # 返回位置嵌入
    return embedding


class MLPEmbedder(nn.Module):
    """
    多层感知机嵌入器，用于将输入嵌入到高维空间。

    参数:
        in_dim (int): 输入的维度。
        hidden_dim (int): 隐藏层的维度。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # 输入线性层
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        # SiLU激活函数
        self.silu = nn.SiLU()
        # 输出线性层
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 嵌入后的张量。
        """
        # 应用输入层、SiLU激活函数和输出层
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    """
    均方根归一化层（RMSNorm），用于对输入张量进行归一化。

    参数:
        dim (int): 输入张量的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 获取输入张量的数据类型
        x_dtype = x.dtype
        x = x.float()
        # 计算均方根归一化因子
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # 应用归一化并乘以缩放因子
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """
    查询-键归一化层（QKNorm），用于对查询和键张量进行归一化。

    参数:
        dim (int): 输入张量的维度。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 查询归一化层
        self.query_norm = RMSNorm(dim)
        # 键归一化层
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播函数。

        参数:
            q (torch.Tensor): 查询张量。
            k (torch.Tensor): 键张量。
            v (torch.Tensor): 值张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 归一化后的查询和键张量。
        """
        # 对查询张量进行归一化
        q = self.query_norm(q)
        # 对键张量进行归一化
        k = self.key_norm(k)
        # 返回归一化后的查询和键张量，并确保它们与值张量在同一设备上
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    """
    自注意力机制模块，用于捕捉输入序列中不同位置之间的关系。

    参数:
        dim (int): 输入和输出的维度。
        num_heads (int): 多头注意力机制中的头数，默认为8。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        # 多头数
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = dim // num_heads

        # 查询、键和值线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 查询-键归一化层
        self.norm = QKNorm(head_dim)
        # 输出投影层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            pe (torch.Tensor): 位置编码张量，形状为 (sequence_length, head_dim)。

        返回:
            torch.Tensor: 自注意力机制的输出，形状为 (batch_size, sequence_length, dim)。
        """
        # 计算查询、键和值
        qkv = self.qkv(x)
        # 重塑张量形状为 (3, batch_size, num_heads, sequence_length, head_dim)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 应用查询-键归一化
        q, k = self.norm(q, k, v)
        # 计算自注意力
        x = attention(q, k, v, pe=pe)
        # 应用输出投影层
        x = self.proj(x)
        # 返回输出
        return x


@dataclass
class ModulationOut:
    """
    调制输出类，用于存储调制的偏移量、缩放因子和门控值。

    参数:
        shift (torch.Tensor): 偏移量张量。
        scale (torch.Tensor): 缩放因子张量。
        gate (torch.Tensor): 门控值张量。
    """
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    """
    调制模块，用于对输入向量进行缩放和平移调制。

    参数:
        dim (int): 输入向量的维度。
        double (bool): 是否进行双重调制。如果为True，则生成两组调制参数；否则，生成一组。
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        # 是否进行双重调制
        self.is_double = double
        # 调制参数的数量
        self.multiplier = 6 if double else 3
        # 线性变换层
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        """
        前向传播函数。

        参数:
            vec (torch.Tensor): 输入向量，形状为 (batch_size, dim)。

        返回:
            Tuple[ModulationOut, Optional[ModulationOut]]: 
                - 第一组调制参数。
                - 如果是双重调制，则返回第二组调制参数；否则，返回None。
        """
        # 应用SiLU激活函数后，通过线性变换生成调制参数
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]), # 第一组调制参数
            ModulationOut(*out[3:]) if self.is_double else None, # 第二组调制参数（如果适用）
        )


class DoubleStreamBlock(nn.Module):
    """
    双流块模块，结合图像和文本的注意力机制和前馈神经网络。

    参数:
        hidden_size (int): 隐藏层的大小。
        num_heads (int): 多头注意力机制中的头数。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        # MLP隐藏层维度
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 多头数
        self.num_heads = num_heads
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 图像调制模块（双重调制）
        self.img_mod = Modulation(hidden_size, double=True)
        # 图像归一化层1
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像自注意力机制
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # 图像归一化层2
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 图像前馈神经网络
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本调制模块（双重调制）
        self.txt_mod = Modulation(hidden_size, double=True)
        # 文本归一化层1
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本自注意力机制
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # 文本归一化层2
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 文本前馈神经网络
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, sequence_length, hidden_size)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, sequence_length, hidden_size)。
            vec (torch.Tensor): 输入向量张量，形状为 (batch_size, hidden_size)。
            pe (torch.Tensor): 位置编码张量，形状为 (sequence_length, head_dim)。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像和文本的输出张量，形状均为 (batch_size, sequence_length, hidden_size)。
        """
        # 对图像进行调制（双重调制）
        img_mod1, img_mod2 = self.img_mod(vec)
        # 对文本进行调制（双重调制）
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # 准备图像用于注意力机制
        # 应用图像归一化层1
        img_modulated = self.img_norm1(img)
        # 应用调制
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # 计算图像查询、键和值
        img_qkv = self.img_attn.qkv(img_modulated)
        # 重塑张量形状
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads) 

        # 应用查询-键归一化
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # 准备文本用于注意力机制
        # 应用文本归一化层1
        txt_modulated = self.txt_norm1(txt)
        # 应用调制
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # 计算文本查询、键和值
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        # 重塑张量形状
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 应用查询-键归一化
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # 运行实际的注意力机制
        # 拼接文本和图像的查询张量
        q = torch.cat((txt_q, img_q), dim=2) 
        # 拼接文本和图像的键张量
        k = torch.cat((txt_k, img_k), dim=2)
        # 拼接文本和图像的值张量
        v = torch.cat((txt_v, img_v), dim=2)
        # 计算注意力
        attn = attention(q, k, v, pe=pe)

        # 分离文本和图像的注意力输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # 计算图像块
        # 应用图像注意力输出
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # 应用图像前馈神经网络
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # 计算文本块
        # 应用文本注意力输出
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # 应用文本前馈神经网络
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # 返回图像和文本的输出
        return img, txt
    

class SingleStreamBlock(nn.Module):
    """
    单流块模块，描述了具有并行线性层的DiT块，并适应了调制接口。

    参数:
        hidden_size (int): 隐藏层的大小。
        num_heads (int): 多头注意力机制中的头数。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维量的比率，默认为4.0。
        qk_scale (float | None): 键查询缩放因子。如果为None，则使用 head_dim ** -0.5。默认为None。
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        # 隐藏层大小
        self.hidden_dim = hidden_size
        # 多头数
        self.num_heads = num_heads
        # 每个注意力头的维度
        head_dim = hidden_size // num_heads
        # 缩放因子
        self.scale = qk_scale or head_dim**-0.5

        # MLP隐藏层维度
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv和mlp输入线性层
        # 线性变换层
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # 投影和mlp输出线性层
        # 线性变换层
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # 查询-键归一化层
        self.norm = QKNorm(head_dim)

        # 隐藏层大小
        self.hidden_size = hidden_size
        # 预归一化层
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # MLP激活函数
        self.mlp_act = nn.GELU(approximate="tanh")
        # 调制模块（单重调制）
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, hidden_size)。
            vec (torch.Tensor): 输入向量张量，形状为 (batch_size, hidden_size)。
            pe (torch.Tensor): 位置编码张量，形状为 (sequence_length, head_dim)。

        返回:
            torch.Tensor: 单流块的输出，形状为 (batch_size, sequence_length, hidden_size)。
        """
        # 应用调制
        mod, _ = self.modulation(vec)
        # 应用预归一化和调制
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # 分离qkv和mlp输入
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重塑qkv张量形状
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 应用查询-键归一化
        q, k = self.norm(q, k, v)

        # 计算注意力
        attn = attention(q, k, v, pe=pe)
        # 计算激活函数在mlp流中，拼接并运行第二个线性层
        # 应用输出线性层
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # 返回输出
        return x + mod.gate * output


class LastLayer(nn.Module):
    """
    最后一层模块，用于将隐藏表示转换为输出图像。

    参数:
        hidden_size (int): 隐藏层的维度。
        patch_size (int): 补丁的大小，用于确定输出图像的尺寸。
        out_channels (int): 输出图像的通道数。
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # 最后的LayerNorm层，不使用仿射变换
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 线性变换层，将隐藏表示映射到输出图像
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 线性变换层，用于生成AdaLN的缩放和平移参数
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, hidden_size)。
            vec (torch.Tensor): 条件向量张量，形状为 (batch_size, hidden_size)。

        返回:
            torch.Tensor: 输出图像张量，形状为 (batch_size, out_channels, height, width)。
        """
        # 将AdaLN调制参数拆分为缩放和平移
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        # 应用AdaLN调制
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        # 应用线性变换，将隐藏表示映射到输出图像
        x = self.linear(x)
        # 返回输出图像
        return x


class DoubleStreamBlock_kv(DoubleStreamBlock):
    """
    双流块模块的扩展版本，结合了图像和文本的注意力机制，并支持保存和恢复特征。

    参数:
        hidden_size (int): 隐藏层的维度。
        num_heads (int): 多头注意力机制中的头数。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__(hidden_size, num_heads, mlp_ratio, qkv_bias)

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, info) -> tuple[Tensor, Tensor]:
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, sequence_length, hidden_size)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, sequence_length, hidden_size)。
            vec (torch.Tensor): 输入向量张量，形状为 (batch_size, hidden_size)。
            pe (torch.Tensor): 位置编码张量，形状为 (sequence_length, head_dim)。
            info (Dict[str, Any]): 信息字典，包含用于保存和恢复特征的信息。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 图像和文本的输出张量，形状均为 (batch_size, sequence_length, hidden_size)。
        """
        # 对图像进行调制（双重调制）
        img_mod1, img_mod2 = self.img_mod(vec)
        # 对文本进行调制（双重调制）
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # 准备图像用于注意力机制
        # 应用图像归一化层1
        img_modulated = self.img_norm1(img)
        # 应用调制
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # 计算图像查询、键和值
        img_qkv = self.img_attn.qkv(img_modulated)
        # 重塑张量形状
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads) 

        # 应用查询-键归一化
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        # 准备文本用于注意力机制
        # 应用文本归一化层1
        txt_modulated = self.txt_norm1(txt)
        # 应用调制
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # 计算文本查询、键和值
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        # 重塑张量形状
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 应用查询-键归一化
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)   
        
        # 保存特征用于反向过程
        # 特征键名称
        feature_k_name = str(info['t']) + '_' + str(info['id']) + '_' + 'MB' + '_' + 'K'
        # 特征值名称
        feature_v_name = str(info['t']) + '_' + str(info['id']) + '_' + 'MB' + '_' + 'V'

        if info['inverse']:
            info['feature'][feature_k_name] = img_k.cpu() # 保存图像键特征
            info['feature'][feature_v_name] = img_v.cpu() # 保存图像值特征
            q = torch.cat((txt_q, img_q), dim=2) # 拼接文本和图像的查询张量
            k = torch.cat((txt_k, img_k), dim=2) # 拼接文本和图像的键张量
            v = torch.cat((txt_v, img_v), dim=2) # 拼接文本和图像的值张量
            if 'attention_mask' in info:
                # 计算注意力（带掩码）
                attn = attention(q, k, v, pe=pe,attention_mask=info['attention_mask'])
            else:
                # 计算注意力（不带掩码）
                attn = attention(q, k, v, pe=pe)
    
        else:
            # 获取保存的图像键特征
            source_img_k = info['feature'][feature_k_name].to(img.device)
            # 获取保存的图像值特征
            source_img_v = info['feature'][feature_v_name].to(img.device)

            # 获取掩码索引
            mask_indices = info['mask_indices'] 
            # 更新图像键特征
            source_img_k[:, :, mask_indices, ...] = img_k
            # 更新图像值特征
            source_img_v[:, :, mask_indices, ...] = img_v
            
            # 拼接文本和图像的查询张量
            q = torch.cat((txt_q, img_q), dim=2)
            # 拼接文本和更新后的图像键张量
            k = torch.cat((txt_k, source_img_k), dim=2)
            # 拼接文本和更新后的图像值张量
            v = torch.cat((txt_v, source_img_v), dim=2)
            # 计算注意力（带位置编码）
            attn = attention(q, k, v, pe=pe, pe_q = info['pe_mask'])

        # 分离文本和图像的注意力输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # 计算图像块
        # 应用图像注意力输出
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # 应用图像前馈神经网络
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # 计算文本块
        # 应用文本注意力输出
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # 应用文本前馈神经网络
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # 返回图像和文本的输出
        return img, txt
    

class SingleStreamBlock_kv(SingleStreamBlock):
    """
    单流块模块的扩展版本，结合了图像和文本的注意力机制，并支持保存和恢复特征。

    参数:
        hidden_size (int): 隐藏层的维度。
        num_heads (int): 多头注意力机制中的头数。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率，默认为4.0。
        qk_scale (float | None): 键查询缩放因子。如果为None，则使用 head_dim ** -0.5，默认为None。
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__(hidden_size, num_heads, mlp_ratio, qk_scale)

    def forward(self,x: Tensor, vec: Tensor, pe: Tensor, info) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, hidden_size)。
            vec (torch.Tensor): 输入向量张量，形状为 (batch_size, hidden_size)。
            pe (torch.Tensor): 位置编码张量，形状为 (sequence_length, head_dim)。
            info (Dict[str, Any]): 信息字典，包含用于保存和恢复特征的信息。

        返回:
            torch.Tensor: 单流块的输出，形状为 (batch_size, sequence_length, hidden_size)。
        """
        # 应用调制
        mod, _ = self.modulation(vec)
        # 应用预归一化和调制
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # 分离qkv和mlp输入
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重塑qkv张量形状
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 应用查询-键归一化
        q, k = self.norm(q, k, v)

        # 保存特征用于反向过程
        img_k = k[:, :, 512:, ...] # 图像键特征
        img_v = v[:, :, 512:, ...] # 图像值特征
        
        txt_k = k[:, :, :512, ...] # 文本键特征
        txt_v = v[:, :, :512, ...] # 文本值特征
    
        # 特征键名称
        feature_k_name = str(info['t']) + '_' + str(info['id']) + '_' + 'SB' + '_' + 'K'
        # 特征值名称
        feature_v_name = str(info['t']) + '_' + str(info['id']) + '_' + 'SB' + '_' + 'V'

        if info['inverse']:
            info['feature'][feature_k_name] = img_k.cpu() # 保存图像键特征
            info['feature'][feature_v_name] = img_v.cpu() # 保存图像值特征
            if 'attention_mask' in info:
                # 计算注意力（带掩码）
                attn = attention(q, k, v, pe=pe,attention_mask=info['attention_mask'])
            else:
                # 计算注意力（不带掩码)
                attn = attention(q, k, v, pe=pe)
            
        else:
            # 获取保存的图像键特征
            source_img_k = info['feature'][feature_k_name].to(x.device)
            # 获取保存的图像值特征
            source_img_v = info['feature'][feature_v_name].to(x.device)

            # 获取掩码索引
            mask_indices = info['mask_indices']
            # 更新图像键特征
            source_img_k[:, :, mask_indices, ...] = img_k
            # 更新图像值特征
            source_img_v[:, :, mask_indices, ...] = img_v
            
            # 拼接文本和更新后的图像键张量
            k = torch.cat((txt_k, source_img_k), dim=2)
            # 拼接文本和更新后的图像值张量
            v = torch.cat((txt_v, source_img_v), dim=2)
            # 计算注意力（带位置编码）
            attn = attention(q, k, v, pe=pe, pe_q = info['pe_mask'])

        # 计算激活函数在mlp流中，拼接并运行第二个线性层
        # 应用输出线性层
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        # 返回输出
        return x + mod.gate * output
