from dataclasses import dataclass
import torch
from torch import Tensor, nn

from layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock
from layers import SingleStreamBlock_kv, DoubleStreamBlock_kv, timestep_embedding


@dataclass
class FluxParams:
    """
    Flux模型参数类，用于配置Flux模型的各种参数。

    参数:
        in_channels (int): 输入图像的通道数，例如RGB图像为3。
        vec_in_dim (int): 向量输入的维度。
        context_in_dim (int): 上下文输入的维度，例如文本嵌入的维度。
        hidden_size (int): 隐藏层的维度，用于定义模型中各层的隐藏大小。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率。
        num_heads (int): 多头注意力机制中的头数。
        depth (int): 双流块（DoubleStreamBlock）的数量。
        depth_single_blocks (int): 单流块（SingleStreamBlock）的数量。
        axes_dim (List[int]): 每个轴的维度列表，用于生成多维位置嵌入。
        theta (int): 旋转角度的基数，用于生成旋转位置编码（RoPE）。
        qkv_bias (bool): 查询、键和值线性变换是否使用偏置，默认为False。
        guidance_embed (bool): 是否使用指导嵌入，默认为False。
    """
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Flux模型，用于序列上的流匹配（Flow Matching on Sequences）。

    参数:
        params (FluxParams): Flux模型的参数。
        double_block_cls (nn.Module, optional): 双流块类，默认为DoubleStreamBlock。用于构建双流块。
        single_block_cls (nn.Module, optional): 单流块类，默认为SingleStreamBlock。用于构建单流块。
    """
    def __init__(self, params: FluxParams,double_block_cls=DoubleStreamBlock,single_block_cls=SingleStreamBlock):
        super().__init__()

        # 保存参数
        self.params = params
        # 输入通道数
        self.in_channels = params.in_channels
        # 输出通道数，假设与输入通道数相同
        self.out_channels = self.in_channels
        # 检查隐藏大小是否能被多头数整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        # 计算位置编码的维度
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            # 检查轴维度之和是否等于位置编码维度
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        # 隐藏层大小
        self.hidden_size = params.hidden_size
        # 多头数
        self.num_heads = params.num_heads
        # 多维位置嵌入器
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        # 图像输入线性层
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # 时间步嵌入器
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # 向量输入嵌入器
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        # 指导嵌入器，如果不需要指导嵌入，则使用恒等层
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        # 文本输入线性层
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # 构建双流块列表
        self.double_blocks = nn.ModuleList(
            [
                double_block_cls(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        # 构建单流块列表
        self.single_blocks = nn.ModuleList(
            [
                single_block_cls(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 最后一层，用于将隐藏表示转换为输出图像
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, height, width)。
            img_ids (torch.Tensor): 输入图像的ID张量，形状为 (batch_size, seq_len_img)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, context_in_dim)。
            txt_ids (torch.Tensor): 输入文本的ID张量，形状为 (batch_size, seq_len_txt)。
            timesteps (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 目标向量张量，形状为 (batch_size, vec_in_dim)。
            guidance (Optional[torch.Tensor]): 指导张量，形状为 (batch_size,)。如果不需要指导嵌入，则为None。

        返回:
            torch.Tensor: 输出图像张量，形状为 (batch_size, out_channels, height, width)。
        """
        if img.ndim != 3 or txt.ndim != 3:
            # 检查输入张量的维度
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 对输入图像进行线性变换
        img = self.img_in(img)
        # 生成时间步嵌入并通过MLP嵌入器
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                # 检查是否提供了指导
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            # 添加指导嵌入
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        # 添加向量输入嵌入
        vec = vec + self.vector_in(y)
        # 对文本进行线性变换
        txt = self.txt_in(txt)

        # 拼接文本和图像的ID
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # 生成多维位置嵌入
        pe = self.pe_embedder(ids)

        # 应用双流块
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # 将文本和图像拼接起来
        img = torch.cat((txt, img), 1)
        # 应用单流块
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        # 分离出图像部分
        img = img[:, txt.shape[1] :, ...]

        # 应用最后一层，将隐藏表示转换为输出图像
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        # 返回输出图像
        return img

class Flux_kv(Flux):
    """
    Flux_kv 类，继承自 Flux 类，并重写了 forward 方法。

    参数:
        params (FluxParams): Flux模型的参数。
        double_block_cls (nn.Module, optional): 双流块类，默认为 DoubleStreamBlock_kv。用于构建双流块。
        single_block_cls (nn.Module, optional): 单流块类，默认为 SingleStreamBlock_kv。用于构建单流块。
    """
    def __init__(self, params: FluxParams,double_block_cls=DoubleStreamBlock_kv,single_block_cls=SingleStreamBlock_kv):
        super().__init__(params,double_block_cls,single_block_cls)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor, 
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor, 
        guidance: Tensor | None = None, 
        info: dict = {},
    ) -> Tensor:
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, height, width)。
            img_ids (torch.Tensor): 输入图像的ID张量，形状为 (batch_size, seq_len_img)。
            txt (torch.Tensor): 输入文本张量，形状为 (batch_size, context_in_dim)。
            txt_ids (torch.Tensor): 输入文本的ID张量，形状为 (batch_size, seq_len_txt)。
            timesteps (torch.Tensor): 时间步张量，形状为 (batch_size,)。
            y (torch.Tensor): 目标向量张量，形状为 (batch_size, vec_in_dim)。
            guidance (Optional[torch.Tensor]): 指导张量，形状为 (batch_size,)。如果不需要指导嵌入，则为 None。
            info (Dict[str, Any]): 信息字典，用于在双流块和单流块之间传递信息，默认为空字典。

        返回:
            torch.Tensor: 输出图像张量，形状为 (batch_size, out_channels, height, width)。
        """
        if img.ndim != 3 or txt.ndim != 3:
            # 检查输入张量的维度
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 对输入图像进行线性变换
        img = self.img_in(img)
        # 生成时间步嵌入并通过MLP嵌入器
        vec = self.time_in(timestep_embedding(timesteps, 256)) 
        if self.params.guidance_embed:
            if guidance is None:
                # 检查是否提供了指导
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            # 添加指导嵌入
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256)) 
        # 添加向量输入嵌入
        vec = vec + self.vector_in(y)
        # 对文本进行线性变换
        txt = self.txt_in(txt)

        # 拼接文本和图像的ID
        ids = torch.cat((txt_ids, img_ids), dim=1) 
        # 生成多维位置嵌入
        pe = self.pe_embedder(ids) 
        if not info['inverse']:
            # 获取掩码索引
            mask_indices = info['mask_indices'] 
            # 如果不是反向过程，则拼接位置编码，保留文本部分，并拼接掩码区域的图像部分
            info['pe_mask'] = torch.cat((pe[:, :, :512, ...],pe[:, :, mask_indices+512, ...]),dim=2)

        # 计数器，用于跟踪当前处理的双流块数量
        cnt = 0
        for block in self.double_blocks:
            # 设置当前块ID
            info['id'] = cnt
            # 应用双流块
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, info=info)
            # 计数器递增
            cnt += 1

        # 重置计数器，用于跟踪当前处理的单流块数量
        cnt = 0
        # 拼接文本和图像
        x = torch.cat((txt, img), 1) 
        for block in self.single_blocks:
            # 设置当前块ID
            info['id'] = cnt
            # 应用单流块
            x = block(x, vec=vec, pe=pe, info=info)
            # 计数器递增
            cnt += 1

        # 分离出图像部分
        img = x[:, txt.shape[1] :, ...]

        # 应用最后一层，将隐藏表示转换为输出图像
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        # 返回输出图像
        return img
