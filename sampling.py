import math
from typing import Callable
import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib import tzip

from model import Flux, Flux_kv
from conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    """
    生成随机噪声张量，用于图像生成过程。

    参数:
        num_samples (int): 生成的样本数量。
        height (int): 图像的高度。
        width (int): 图像的宽度。
        device (torch.device): 张量所在的设备，如 'cuda' 或 'cpu'。
        dtype (torch.dtype): 张量的数据类型，如 torch.float32。
        seed (int): 随机种子，用于控制生成的随机性。

    返回:
        torch.Tensor: 生成的随机噪声张量，形状为 (num_samples, 16, 2 * ceil(height/16), 2 * ceil(width/16))。
    """
    return torch.randn(
        num_samples, # 样本数量
        16, # 通道数，假设为16
        # 允许打包，将高度和宽度向上取整为16的倍数
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed), # 随机数生成器，设定种子
    )
    

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    """
    准备输入数据，用于模型的前向传播。

    参数:
        t5 (HFEmbedder): T5文本嵌入器，用于将文本转换为嵌入向量。
        clip (HFEmbedder): CLIP文本嵌入器，用于将文本转换为嵌入向量。
        img (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
        prompt (str 或 list[str]): 输入提示，可以是单个字符串或字符串列表。

    返回:
        dict[str, torch.Tensor]: 包含准备好的输入数据的字典，包括图像、图像ID、文本、文本ID和向量。
    """
    # 获取批量大小、通道数、高度和宽度
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        # 如果批量大小为1且提示不是字符串，则批量大小为提示列表的长度
        bs = len(prompt)

    # 重塑图像张量，将高度和宽度打包为16的倍数
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        # 如果图像批量大小为1且需要更大的批量，则重复图像
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # 生成图像ID张量，形状为 (h//2, w//2, 3)
    img_ids = torch.zeros(h // 2, w // 2, 3)
    # 为第二维添加高度索引
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    # 为第三维添加宽度索引
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    # 重复图像ID以匹配批量大小
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        # 如果提示是字符串，则转换为列表
        prompt = [prompt]
    # 使用T5嵌入器生成文本嵌入
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        # 如果文本批量大小为1且需要更大的批量，则重复文本
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    # 生成文本ID张量，形状为 (batch_size, seq_len, 3)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    # 使用CLIP嵌入器生成文本向量
    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        # 如果向量批量大小为1且需要更大的批量，则重复向量
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img, # 准备好的图像张量
        "img_ids": img_ids.to(img.device), # 图像ID张量，移动到与图像相同的设备
        "txt": txt.to(img.device), # 文本嵌入张量，移动到与图像相同的设备
        "txt_ids": txt_ids.to(img.device), # 文本ID张量，移动到与图像相同的设备
        "vec": vec.to(img.device), # 向量张量，移动到与图像相同的设备
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    """
    时间步平移函数，用于调整时间步的分布。

    参数:
        mu (float): 平移参数，控制时间步的平移量。
        sigma (float): 缩放参数，控制时间步的缩放量。
        t (torch.Tensor): 输入时间步张量。

    返回:
        torch.Tensor: 平移后的时间步张量。
    """
    # 应用时间步平移公式
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    """
    生成线性函数，用于计算mu参数。

    参数:
        x1 (float): 第一个点的x坐标，默认为256。
        y1 (float): 第一个点的y坐标，默认为0.5。
        x2 (float): 第二个点的x坐标，默认为4096。
        y2 (float): 第二个点的y坐标，默认为1.15。

    返回:
        Callable[[float], float]: 生成的线性函数。
    """
    # 计算斜率
    m = (y2 - y1) / (x2 - x1)
    # 计算截距
    b = y1 - m * x1
    # 返回线性函数
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """
    生成时间步调度列表，用于控制去噪过程的强度。

    参数:
        num_steps (int): 去噪过程的步数。
        image_seq_len (int): 图像序列的长度。
        base_shift (float): 基础平移量，默认为0.5。
        max_shift (float): 最大平移量，默认为1.15。
        shift (bool): 是否进行平移，默认为True。

    返回:
        List[float]: 生成的时间步调度列表。
    """
    # 为零添加额外的时间步
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # 平移调度以偏向高时间步，对高信号图像更有利
    if shift:
        # 基于线性估计在两点之间估计mu
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    # 返回时间步调度列表
    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
):
    """
    去噪函数，用于对图像进行去噪。

    参数:
        model (Flux): Flux模型实例，用于前向传播。
        img (torch.Tensor): 输入图像张量。
        img_ids (torch.Tensor): 输入图像ID张量。
        txt (torch.Tensor): 输入文本张量。
        txt_ids (torch.Tensor): 输入文本ID张量。
        vec (torch.Tensor): 输入向量张量。
        timesteps (List[float]): 时间步列表。
        guidance (float): 指导强度，默认为4.0。

    返回:
        torch.Tensor: 去噪后的图像张量，形状与输入图像相同。
    """
    # 对于 'schnell' 模型，guidance_vec 被忽略
    # 创建指导向量，形状为 (batch_size,)
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # 遍历时间步对
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # 创建当前时间步张量，形状为 (batch_size,)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # 前向传播，获取预测结果
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        # 更新图像张量，应用去噪步长
        img = img + (t_prev - t_curr) * pred
    # 返回去噪后的图像
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    """
    重新排列张量形状，从 (batch_size, seq_len, channels * patch_size * patch_size) 变为 (batch_size, channels, height, width)。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, channels * patch_size * patch_size)。
        height (int): 图像的高度。
        width (int): 图像的宽度。

    返回:
        torch.Tensor: 重新排列后的张量，形状为 (batch_size, channels, height, width)。
    """
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)", # 重塑张量形状
        h=math.ceil(height / 16), # 计算高度上的补丁数量，假设每个补丁大小为16
        w=math.ceil(width / 16), # 计算宽度上的补丁数量
        ph=2, # 补丁高度上的维度
        pw=2, # 补丁宽度上的维度
    )


def denoise_kv(
    model: Flux_kv,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    """
    去噪函数，用于对图像进行去噪，支持反向过程。

    参数:
        model (Flux_kv): Flux_kv模型实例，用于前向传播。
        img (torch.Tensor): 输入图像张量。
        img_ids (torch.Tensor): 输入图像ID张量。
        txt (torch.Tensor): 输入文本张量。
        txt_ids (torch.Tensor): 输入文本ID张量。
        vec (torch.Tensor): 输入向量张量。
        timesteps (List[float]): 时间步列表。
        inverse (bool): 是否为反向过程。
        info (Dict[str, Any]): 信息字典，用于传递中间信息。
        guidance (float): 指导强度，默认为4.0。

    返回:
        Tuple[torch.Tensor, Dict[str, Any]]: 去噪后的图像张量和更新后的信息字典。
    """
    if inverse:
        # 如果是反向过程，则反转时间步列表
        timesteps = timesteps[::-1]
    # 创建指导向量
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    # 遍历时间步对
    for i, (t_curr, t_prev) in enumerate(tzip(timesteps[:-1], timesteps[1:])):
        # 创建当前时间步张量
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # 设置当前时间步
        info['t'] = t_prev if inverse else t_curr
        
        if inverse:
            # 生成图像名称
            img_name = str(info['t']) + '_' + 'img'
            # 保存当前图像到信息字典
            info['feature'][img_name] = img.cpu()
        else:
            # 生成图像名称
            img_name = str(info['t']) + '_' + 'img'
            # 获取保存的图像
            source_img = info['feature'][img_name].to(img.device)
            # 更新图像
            img = source_img[:, info['mask_indices'],...] * (1 - info['mask'][:, info['mask_indices'],...]) + img * info['mask'][:, info['mask_indices'],...]
        
        # 前向传播，获取预测结果
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        # 更新图像张量，应用去噪步长
        img = img + (t_prev - t_curr) * pred
    # 返回去噪后的图像和信息字典
    return img, info


def denoise_kv_inf(
    model: Flux_kv,
    # model input
    img: Tensor,
    img_ids: Tensor,
    source_txt: Tensor,
    source_txt_ids: Tensor,
    source_vec: Tensor,
    target_txt: Tensor,
    target_txt_ids: Tensor,
    target_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    target_guidance: float = 4.0,
    source_guidance: float = 4.0,
    info: dict = {},
):
    """
    去噪函数，用于对图像进行去噪，支持源和目标指导。

    参数:
        model (Flux_kv): Flux_kv模型实例，用于前向传播。
        img (torch.Tensor): 输入图像张量。
        img_ids (torch.Tensor): 输入图像ID张量。
        source_txt (torch.Tensor): 源文本张量。
        source_txt_ids (torch.Tensor): 源文本ID张量。
        source_vec (torch.Tensor): 源向量张量。
        target_txt (torch.Tensor): 目标文本张量。
        target_txt_ids (torch.Tensor): 目标文本ID张量。
        target_vec (torch.Tensor): 目标向量张量。
        timesteps (List[float]): 时间步列表。
        target_guidance (float): 目标指导强度，默认为4.0。
        source_guidance (float): 源指导强度，默认为4.0。
        info (dict): 信息字典，用于传递中间信息。

    返回:
        Tuple[torch.Tensor, dict]: 去噪后的图像张量和更新后的信息字典。
    """
    # 创建目标指导向量 
    target_guidance_vec = torch.full((img.shape[0],), target_guidance, device=img.device, dtype=img.dtype)
    # 创建源指导向量
    source_guidance_vec = torch.full((img.shape[0],), source_guidance, device=img.device, dtype=img.dtype)
    
    # 获取掩码索引
    mask_indices = info['mask_indices']
    # 克隆输入图像
    init_img = img.clone() 
    # 获取掩码区域的图像
    z_fe = img[:, mask_indices,...]
    
    # 初始化噪声列表
    noise_list = []
    for i in range(len(timesteps)):
        # 生成随机噪声
        noise = torch.randn(init_img.size(), dtype=init_img.dtype, 
                        layout=init_img.layout, device=init_img.device,
                        generator=torch.Generator(device=init_img.device).manual_seed(0)) 
        # 添加到噪声列表
        noise_list.append(noise)

    # 遍历时间步对
    for i, (t_curr, t_prev) in enumerate(tzip(timesteps[:-1], timesteps[1:])): 
        # 设置当前时间步
        info['t'] = t_curr
        # 创建当前时间步张量
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # 计算源图像
        z_src = (1 - t_curr) * init_img + t_curr * noise_list[i]
        # 计算目标图像
        z_tar = z_src[:, mask_indices,...] - init_img[:, mask_indices,...] + z_fe
        
        # 设置为反向过程
        info['inverse'] = True
        # 初始化特征字典
        info['feature'] = {}
        # 前向传播，获取源预测结果
        v_src = model(
            img=z_src,
            img_ids=img_ids,
            txt=source_txt,
            txt_ids=source_txt_ids,
            y=source_vec,
            timesteps=t_vec,
            guidance=source_guidance_vec,
            info=info
        )
        # 设置为正向过程
        info['inverse'] = False
        # 前向传播，获取目标预测结果
        v_tar = model(
            img=z_tar,
            img_ids=img_ids,
            txt=target_txt,
            txt_ids=target_txt_ids,
            y=target_vec,
            timesteps=t_vec,
            guidance=target_guidance_vec,
            info=info
        )
        # 计算特征差异
        v_fe = v_tar - v_src[:, mask_indices,...]
        # 更新特征
        z_fe = z_fe + (t_prev - t_curr) * v_fe * info['mask'][:, mask_indices,...]
    # 返回更新后的特征和信息字典
    return z_fe, info
