from dataclasses import dataclass
import torch
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    """
    自编码器参数类，用于配置自编码器的各种参数。

    参数:
        resolution (int): 输入图像的分辨率，通常为图像的高度或宽度。
        in_channels (int): 输入图像的通道数，例如RGB图像为3。
        ch (int): 初始通道数，用于定义模型中各层的通道数。
        out_ch (int): 输出通道数，通常与输入通道数相同，用于重建图像。
        ch_mult (List[int]): 通道数乘数列表，用于逐步增加或减少通道数。
                             例如，[1, 2, 4] 表示通道数依次乘以1、2、4。
        num_res_blocks (int): 残差块的数量，用于构建残差网络。
        z_channels (int): 潜在空间的通道数，即编码器输出和解码器输入的维度。
        scale_factor (float): 缩放因子，用于调整潜在空间的大小。
        shift_factor (float): 平移因子，用于调整潜在空间的位置。
    """
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


def swish(x: Tensor) -> Tensor:
    """
    Swish激活函数。

    Swish激活函数定义为：f(x) = x * sigmoid(x)。

    参数:
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 应用Swish激活函数后的张量。
    """
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    """
    注意力块（Attention Block），用于在自编码器中引入自注意力机制。

    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 输入通道数
        self.in_channels = in_channels

        # 分组归一化层，组数为32
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # 定义查询、键和值卷积层，卷积核大小为1
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # 定义输出投影卷积层，卷积核大小为1
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """
        计算自注意力机制。

        参数:
            h_ (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
            torch.Tensor: 注意力机制的输出，形状为 (batch_size, channels, height, width)。
        """
        # 应用归一化
        h_ = self.norm(h_)
        # 计算查询
        q = self.q(h_)
        # 计算键
        k = self.k(h_)
        # 计算值
        v = self.v(h_)

        # 获取批量大小、通道数、高度和宽度
        b, c, h, w = q.shape
        # 重塑查询、键和值张量以适应缩放点积注意力计算
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        # 计算缩放点积注意力
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        # 重塑回原始形状
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
            torch.Tensor: 注意力块的输出，形状为 (batch_size, channels, height, width)。
        """
        # 残差连接
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """
    残差块（ResNet Block），用于构建残差网络。

    参数:
        in_channels (int): 输入通道数。
        out_channels (int, optional): 输出通道数。如果为None，则输出通道数与输入通道数相同。默认为None。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 输入通道数
        self.in_channels = in_channels
        # 如果输出通道数为None，则输出通道数与输入通道数相同
        out_channels = in_channels if out_channels is None else out_channels
        # 输出通道数
        self.out_channels = out_channels

        # 定义第一个归一化层和卷积层
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 定义第二个归一化层和卷积层
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 如果输入和输出通道数不同，则定义一个1x1卷积层进行通道数匹配
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 残差块的输出，形状为 (batch_size, out_channels, height, width)。
        """
        # 保留输入张量
        h = x
        # 应用第一个归一化层
        h = self.norm1(h)
        # 应用Swish激活函数
        h = swish(h)
        # 应用第一个卷积层
        h = self.conv1(h)

        # 应用第二个归一化层
        h = self.norm2(h)
        # 应用Swish激活函数
        h = swish(h)
        # 应用第二个卷积层
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            # 如果通道数不同，则应用1x1卷积进行匹配
            x = self.nin_shortcut(x)

        # 残差连接
        return x + h


class Downsample(nn.Module):
    """
    下采样模块，用于在编码器中降低特征图的分辨率。

    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 使用3x3卷积，步幅为2，实现下采样，同时保持特征图尺寸
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 下采样后的张量，形状为 (batch_size, in_channels, height/2, width/2)。
        """
        # 对输入张量进行填充，填充方式为常数填充，填充值为0
        # pad参数为 (左, 右, 上, 下)，这里在右侧和底部各填充1个像素
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        # 应用卷积操作，实现下采样
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """
    上采样模块，用于在解码器中提高特征图的分辨率。

    参数:
        in_channels (int): 输入通道数。
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # 使用3x3卷积，步幅为1，填充为1，保持特征图尺寸
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 上采样后的张量，形状为 (batch_size, in_channels, height*2, width*2)。
        """
        # 使用最近邻插值法进行上采样，放大2倍
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        # 应用卷积操作
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """
    编码器类，用于将输入图像编码为潜在表示。

    参数:
        resolution (int): 输入图像的分辨率，通常为图像的高度或宽度。
        in_channels (int): 输入图像的通道数，例如RGB图像为3。
        ch (int): 初始通道数，用于定义模型中各层的通道数。
        ch_mult (List[int]): 通道数乘数列表，用于逐步增加通道数。
                             例如，[1, 2, 4] 表示通道数依次乘以1、2、4。
        num_res_blocks (int): 每个分辨率级别中残差块的数量。
        z_channels (int): 潜在空间的通道数，即编码器输出和解码器输入的维度。
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        # 初始通道数
        self.ch = ch
        # 分辨率级别的数量
        self.num_resolutions = len(ch_mult)
        # 每个分辨率级别中残差块的数量
        self.num_res_blocks = num_res_blocks
        # 输入图像的分辨率
        self.resolution = resolution
        # 输入图像的通道数
        self.in_channels = in_channels
        # 下采样过程
        # 输入卷积层
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        # 当前分辨率
        curr_res = resolution
        # 计算每个分辨率级别的输入通道数乘数
        in_ch_mult = (1,) + tuple(ch_mult)
        # 保存输入通道数乘数
        self.in_ch_mult = in_ch_mult
        # 下采样模块列表
        self.down = nn.ModuleList()
        # 当前块的输入通道数
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            # 当前分辨率级别的残差块列表
            block = nn.ModuleList()
            # 当前分辨率级别的注意力块列表
            attn = nn.ModuleList()
            # 计算当前块的输入通道数
            block_in = ch * in_ch_mult[i_level]
            # 计算当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out
            # 创建下采样模块
            down = nn.Module()
            # 保存残差块列表
            down.block = block
            # 保存注意力块列表
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # 如果不是最后一个分辨率级别，则添加下采样层
                down.downsample = Downsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res // 2
            # 添加下采样模块到列表中
            self.down.append(down)

        # 中间部分
        # 中间模块
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 注意力块
        self.mid.attn_1 = AttnBlock(block_in)
        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 输出部分
        # 分组归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 输出卷积层
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 编码器的输出，形状为 (batch_size, 2 * z_channels, height / (2 ** num_resolutions), width / (2 ** num_resolutions))。
        """
        # 下采样过程
        # 应用输入卷积层并添加到列表中
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # 应用残差块
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    # 应用注意力块（如果有）
                    h = self.down[i_level].attn[i_block](h)
                # 添加到列表中
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                # 应用下采样层（如果不是最后一个分辨率级别）
                hs.append(self.down[i_level].downsample(hs[-1]))

        # 中间部分
        # 获取最后一个输出
        h = hs[-1]
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用注意力块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # 输出部分
        # 应用分组归一化
        h = self.norm_out(h)
        # 应用Swish激活函数
        h = swish(h)
        # 应用输出卷积层
        h = self.conv_out(h)
        # 返回编码器的输出
        return h


class Decoder(nn.Module):
    """
    解码器类，用于将潜在表示解码为重建图像。

    参数:
        ch (int): 初始通道数，用于定义模型中各层的通道数。
        out_ch (int): 输出通道数，通常与输入图像的通道数相同，例如RGB图像为3。
        ch_mult (List[int]): 通道数乘数列表，用于逐步增加或减少通道数。
                             例如，[1, 2, 4] 表示通道数依次乘以1、2、4。
        num_res_blocks (int): 每个分辨率级别中残差块的数量。
        in_channels (int): 输入通道数，通常为潜在空间的通道数。
        resolution (int): 输出图像的分辨率，通常为图像的高度或宽度。
        z_channels (int): 潜在空间的通道数，即编码器输出和解码器输入的维度。
    """
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        # 初始通道数
        self.ch = ch
        # 分辨率级别的数量
        self.num_resolutions = len(ch_mult)
        # 每个分辨率级别中残差块的数量
        self.num_res_blocks = num_res_blocks
        # 输出图像的分辨率
        self.resolution = resolution
        # 输入通道数
        self.in_channels = in_channels
        # 计算上采样因子
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # 计算最低分辨率级别的输入通道数乘数、块的输入通道数和当前分辨率
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 潜在空间的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # 潜在空间到块的输入通道数
        # 卷积层
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # 中间部分
        # 中间模块
        self.mid = nn.Module()
        # 第一个残差块
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        # 注意力块
        self.mid.attn_1 = AttnBlock(block_in)
        # 第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # 上采样过程
        # 上采样模块列表
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            # 当前分辨率级别的残差块列表
            block = nn.ModuleList()
            # 当前分辨率级别的注意力块列表
            attn = nn.ModuleList()
            # 当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                # 添加残差块
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                # 更新输入通道数
                block_in = block_out
            # 创建上采样模块
            up = nn.Module()
            # 保存残差块列表
            up.block = block
            # 保存注意力块列表
            up.attn = attn
            if i_level != 0:
                # 如果不是第一个分辨率级别，则添加上采样层
                up.upsample = Upsample(block_in)
                # 更新当前分辨率
                curr_res = curr_res * 2
            # 将上采样模块插入到列表的前面，以保持顺序一致
            self.up.insert(0, up)  # prepend to get consistent order

        # 输出部分
        # 分组归一化层
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        # 输出卷积层
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            z (torch.Tensor): 输入的潜在表示，形状为 (batch_size, z_channels, height, width)。

        返回:
            torch.Tensor: 解码器的输出重建图像，形状为 (batch_size, out_ch, resolution, resolution)。
        """
        # 潜在空间到块的输入通道数
        # 应用卷积层
        h = self.conv_in(z)

        # 中间部分
        # 应用第一个残差块
        h = self.mid.block_1(h)
        # 应用注意力块
        h = self.mid.attn_1(h)
        # 应用第二个残差块
        h = self.mid.block_2(h)

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                # 应用残差块
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    # 应用注意力块（如果有）
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # 应用上采样层
                h = self.up[i_level].upsample(h)

        # 输出部分
        # 应用分组归一化
        h = self.norm_out(h)
        # 应用Swish激活函数
        h = swish(h)
        # 应用输出卷积层
        h = self.conv_out(h)
        # 返回重建图像
        return h


class DiagonalGaussian(nn.Module):
    """
    对角高斯分布，用于对潜在表示进行采样或计算均值。

    参数:
        sample (bool): 是否进行采样。如果为True，则从对角高斯分布中采样；否则，返回均值。默认为True。
        chunk_dim (int): 分块维度，用于分割均值和方差。默认为1。
    """
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        # 是否进行采样
        self.sample = sample
        # 分块维度
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            z (torch.Tensor): 输入张量，形状为 (batch_size, ..., 2 * z_channels)。

        返回:
            torch.Tensor: 从对角高斯分布中采样或返回均值，形状为 (batch_size, ..., z_channels)。
        """
        # 将输入张量分割为均值和方差的对数
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        # import pdb;pdb.set_trace()
        if self.sample:
            # 计算标准差
            std = torch.exp(0.5 * logvar)
            # 从对角高斯分布中采样
            return mean #+ std * torch.randn_like(mean)
        else:
            # 返回均值
            return mean


class AutoEncoder(nn.Module):
    """
    自编码器类，结合编码器、解码器和潜在分布。

    参数:
        params (AutoEncoderParams): 自编码器的参数。
    """
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        # 初始化编码器
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        # 初始化解码器
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        # 初始化对角高斯分布
        self.reg = DiagonalGaussian()

        # 缩放因子
        self.scale_factor = params.scale_factor
        # 平移因子
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        """
        编码函数，将输入图像编码为潜在表示。

        参数:
            x (torch.Tensor): 输入图像，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 潜在表示，形状为 (batch_size, z_channels, height / (2 ** num_resolutions), width / (2 ** num_resolutions))。
        """
        # 应用编码器和潜在分布
        z = self.reg(self.encoder(x))
        # 应用缩放和平移
        z = self.scale_factor * (z - self.shift_factor)
        # 返回潜在表示
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        解码函数，将潜在表示解码为重建图像。

        参数:
            z (torch.Tensor): 潜在表示，形状为 (batch_size, z_channels, height / (2 ** num_resolutions), width / (2 ** num_resolutions))。

        返回:
            torch.Tensor: 重建图像，形状为 (batch_size, in_channels, height, width)。
        """
        # 逆向应用缩放和平移
        z = z / self.scale_factor + self.shift_factor
        # 返回重建图像
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入图像，形状为 (batch_size, in_channels, height, width)。

        返回:
            torch.Tensor: 重建图像，形状为 (batch_size, in_channels, height, width)。
        """
        # 先编码再解码，实现自编码过程
        return self.decode(self.encode(x))
