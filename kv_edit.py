from dataclasses import dataclass
from einops import rearrange,repeat
import torch
import torch.nn.functional as F
from torch import Tensor

from sampling import get_schedule, unpack,denoise_kv,denoise_kv_inf
from util import load_flow_model
from model import Flux_kv


@dataclass
class SamplingOptions:
    """
    采样选项类，用于配置生成过程中的各种参数。

    参数:
        source_prompt (str): 源提示，用于指导生成过程的起始文本。默认为空字符串 ''。
        target_prompt (str): 目标提示，用于指导生成过程的目标文本。默认为空字符串 ''。
        # prompt (str): 提示文本，已注释掉，未使用。
        width (int): 生成图像的宽度，默认为1366像素。
        height (int): 生成图像的高度，默认为768像素。
        inversion_num_steps (int): 反演过程的步数，用于图像反演。默认为0。
        denoise_num_steps (int): 去噪过程的步数，用于图像去噪。默认为0。
        skip_step (int): 跳过的步数，用于跳过生成过程中的某些步骤。默认为0。
        inversion_guidance (float): 反演过程的指导权重，用于控制反演过程的强度。默认为1.0。
        denoise_guidance (float): 去噪过程的指导权重，用于控制去噪过程的强度。默认为1.0。
        seed (int): 随机种子，用于控制生成过程的随机性。默认为42。
        re_init (bool): 是否重新初始化模型参数。默认为False。
        attn_mask (bool): 是否使用注意力掩码。默认为False。
    """
    source_prompt: str = ''
    target_prompt: str = ''
    # prompt: str
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False


class only_Flux(torch.nn.Module): 
    """
    Flux模型类，用于加载Flux模型并创建自定义的注意力掩码。

    参数:
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        name (str): 模型名称，默认为 'flux-dev'。
    """
    def __init__(self, device,name='flux-dev'):
        self.device = device
        self.name = name
        super().__init__()
        # 加载Flux模型，指定设备和Flux_kv类
        self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux_kv)
        
    def create_attention_mask(self,seq_len, mask_indices, text_len=512, device='cuda'):
        """
        创建自定义的注意力掩码。

        该方法根据给定的序列长度和掩码区域索引，生成一个注意力掩码张量。
        掩码用于控制模型在生成过程中对不同区域的关注程度。

        参数:
            seq_len (int): 序列长度，即输入序列的总长度。
            mask_indices (List[int]): 图像令牌中掩码区域的索引列表。
            text_len (int): 文本令牌的长度，默认为512。
            device (str): 设备类型，如 'cuda' 或 'cpu'，默认为 'cuda'。

        返回:
            torch.Tensor: 生成的注意力掩码张量，形状为 (1, seq_len, seq_len)。
        """
        # 初始化掩码为全False，形状为 (seq_len, seq_len)
        attention_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # 生成文本令牌的索引，范围从0到text_len-1
        text_indices = torch.arange(0, text_len, device=device)

        # 生成图像掩码令牌的索引，范围从text_len到seq_len-1
        mask_token_indices = torch.tensor([idx + text_len for idx in mask_indices], device=device)

        # 生成背景令牌的索引，排除掩码区域的索引
        all_indices = torch.arange(text_len, seq_len, device=device)
        background_token_indices = torch.tensor([idx for idx in all_indices if idx not in mask_token_indices])

        # 设置文本查询可以关注所有键
        # 文本查询关注所有键
        attention_mask[text_indices.unsqueeze(1).expand(-1, seq_len)] = True
        # 文本查询关注文本键
        attention_mask[text_indices.unsqueeze(1), text_indices] = True
        # 文本查询关注背景键
        attention_mask[text_indices.unsqueeze(1), background_token_indices] = True
        
        # 设置掩码查询可以关注文本键和掩码键
        attention_mask[mask_token_indices.unsqueeze(1), text_indices] = True  # 关注文本
        attention_mask[mask_token_indices.unsqueeze(1), mask_token_indices] = True  # 关注掩码区域

        # 设置背景查询可以关注文本键和掩码键
        # attention_mask[background_token_indices.unsqueeze(1).expand(-1, seq_len), :] = False
        attention_mask[background_token_indices.unsqueeze(1), mask_token_indices] = True  # 关注掩码
        attention_mask[background_token_indices.unsqueeze(1), text_indices] = True  # 关注文本
        attention_mask[background_token_indices.unsqueeze(1), background_token_indices] = True  # 关注背景区域

        # 返回形状为 (1, seq_len, seq_len) 的注意力掩码
        return attention_mask.unsqueeze(0)
    

class Flux_kv_edit_inf(only_Flux):
    """
    Flux_kv_edit_inf 类，继承自 only_Flux，用于在推理阶段对图像进行编辑。

    参数:
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        name (str): 模型名称，用于指定使用的Flux模型。
    """
    def __init__(self, device,name):
        super().__init__(device,name)

    @torch.inference_mode()
    def forward(self,inp,inp_target,mask:Tensor,opts):
        """
        前向传播函数，用于在推理阶段对图像进行编辑。

        参数:
            inp (Dict[str, torch.Tensor]): 输入字典，包含图像、图像ID、文本、文本ID和向量等信息。
                - inp["img"]: 输入图像，形状为 (batch_size, L, d)。
                - inp['img_ids']: 图像ID。
                - inp['txt']: 输入文本。
                - inp['txt_ids']: 输入文本ID。
                - inp['vec']: 输入向量。
            inp_target (Dict[str, torch.Tensor]): 目标输入字典，包含文本、文本ID和向量等信息。
                - inp_target['txt']: 目标文本。
                - inp_target['txt_ids']: 目标文本ID。
                - inp_target['vec']: 目标向量。
            mask (torch.Tensor): 掩码张量，形状为 (batch_size, 1, H, W)。
            opts (SamplingOptions): 采样选项，包含生成过程的配置参数。

        返回:
            torch.Tensor: 编辑后的图像，形状为 (batch_size, 3, height, width)。
        """
        # 初始化信息字典，用于存储中间信息
        info = {}
        # 初始化特征信息
        info['feature'] = {}
        # 获取批量大小、序列长度和维度
        bs, L, d = inp["img"].shape
        # 计算高度
        h = opts.height // 8
        # 计算宽度
        w = opts.width // 8
        # 对掩码进行插值，使其与特征图大小匹配
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        # 将掩码二值化，大于0的像素设为1
        mask[mask > 0] = 1
        
        # 重复掩码以匹配模型输入的通道数（假设为16）
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
        # 重塑掩码以适应模型输入的维度
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        # 将掩码存储在信息字典中
        info['mask'] = mask
        # 生成布尔掩码，标记哪些位置需要被处理
        bool_mask = (mask.sum(dim=2) > 0.5)
        # 获取需要处理的索引
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1] 
        # 如果使用注意力掩码且存在需要处理的位置，则创建注意力掩码
        if opts.attn_mask and (~bool_mask).any():
            attention_mask = self.create_attention_mask(L+512, info['mask_indices'], device=self.device)
        else:
            attention_mask = None   
        # 将注意力掩码存储在信息字典中
        info['attention_mask'] = attention_mask
        
        # 获取去噪的时间步长
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        # 根据跳步参数调整时间步长
        denoise_timesteps = denoise_timesteps[opts.skip_step:]

        # 获取输入图像
        z0 = inp["img"]

        with torch.no_grad():
            # 标记注入状态
            info['inject'] = True
            # 调用去噪函数进行推理
            z_fe, info = denoise_kv_inf(self.model, img=inp["img"], img_ids=inp['img_ids'], 
                                    source_txt=inp['txt'], source_txt_ids=inp['txt_ids'], source_vec=inp['vec'],
                                    target_txt=inp_target['txt'], target_txt_ids=inp_target['txt_ids'], target_vec=inp_target['vec'],
                                    timesteps=denoise_timesteps, source_guidance=opts.inversion_guidance, target_guidance=opts.denoise_guidance,
                                    info=info)
        # 获取掩码索引
        mask_indices = info['mask_indices'] 

        # 将去噪后的特征替换到原始图像中
        z0[:, mask_indices,...] = z_fe

        # 解包图像数据
        z0 = unpack(z0.float(),  opts.height, opts.width)

        # 删除信息字典
        del info

        # 返回编辑后的图像
        return z0


class Flux_kv_edit(only_Flux):
    """
    Flux_kv_edit 类，继承自 only_Flux，用于对图像进行编辑。

    参数:
        device (str): 设备类型，如 'cuda' 或 'cpu'。
        name (str): 模型名称，用于指定使用的Flux模型。
    """
    def __init__(self, device,name):
        super().__init__(device,name)
    
    @torch.inference_mode()
    def forward(self,inp,inp_target,mask:Tensor,opts):
        """
        前向传播函数，用于对图像进行编辑。

        参数:
            inp (Dict[str, torch.Tensor]): 输入字典，包含图像、图像ID、文本、文本ID和向量等信息。
                - inp["img"]: 输入图像，形状为 (batch_size, L, d)。
                - inp['img_ids']: 图像ID。
                - inp['txt']: 输入文本。
                - inp['txt_ids']: 输入文本ID。
                - inp['vec']: 输入向量。
            inp_target (Dict[str, torch.Tensor]): 目标输入字典，包含文本、文本ID和向量等信息。
                - inp_target['txt']: 目标文本。
                - inp_target['txt_ids']: 目标文本ID。
                - inp_target['vec']: 目标向量。
            mask (torch.Tensor): 掩码张量，形状为 (batch_size, 1, H, W)。
            opts (SamplingOptions): 采样选项，包含生成过程的配置参数。

        返回:
            torch.Tensor: 编辑后的图像，形状为 (batch_size, 3, height, width)。
        """
        # 进行反演过程
        z0,zt,info = self.inverse(inp,mask,opts)
        # 进行去噪过程
        z0 = self.denoise(z0,zt,inp_target,mask,opts,info)
        # 返回编辑后的图像
        return z0
    
    @torch.inference_mode()
    def inverse(self,inp,mask,opts):
        """
        反演函数，用于对图像进行反演。

        参数:
            inp (Dict[str, torch.Tensor]): 输入字典，包含图像、图像ID、文本、文本ID和向量等信息。
            mask (torch.Tensor): 掩码张量，形状为 (batch_size, 1, H, W)。
            opts (SamplingOptions): 采样选项，包含生成过程的配置参数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: 
                - z0 (torch.Tensor): 输入图像，形状为 (batch_size, L, d)。
                - zt (torch.Tensor): 加噪后的图像，形状为 (batch_size, L, d)。
                - info (Dict[str, Any]): 信息字典，包含中间信息。
        """
        # 初始化信息字典
        info = {}
        # 初始化特征信息
        info['feature'] = {}
        # 获取批量大小、序列长度和维度
        bs, L, d = inp["img"].shape
        # 计算高度
        h = opts.height // 8
        # 计算宽度
        w = opts.width // 8

        if opts.attn_mask:
            # 对掩码进行插值，使其与特征图大小匹配
            mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
            # 将掩码二值化，大于0的像素设为1
            mask[mask > 0] = 1
            
            # 重复掩码以匹配模型输入的通道数（假设为16）
            mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
            # 重塑掩码以适应模型输入的维度
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            # 生成布尔掩码
            bool_mask = (mask.sum(dim=2) > 0.5)
            # 获取需要处理的索引
            mask_indices = torch.nonzero(bool_mask)[:,1] 
            
            # 确保掩码不是全False或全True
            assert not (~bool_mask).all(), "mask is all false"
            assert not (bool_mask).all(), "mask is all true"
            # 创建注意力掩码
            attention_mask = self.create_attention_mask(L+512, mask_indices, device=mask.device)
            # 将注意力掩码存储在信息字典中
            info['attention_mask'] = attention_mask
    
        # 获取去噪的时间步长
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        # 根据跳步参数调整时间步长
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
        
        # 加噪过程
        # 克隆输入图像
        z0 = inp["img"].clone()     
        # 标记反演状态   
        info['inverse'] = True
        zt, info = denoise_kv(self.model, **inp, timesteps=denoise_timesteps, guidance=opts.inversion_guidance, inverse=True, info=info)
        
        # 返回输入图像、加噪后的图像和信息字典
        return z0,zt,info
    
    @torch.inference_mode()
    def denoise(self,z0,zt,inp_target,mask:Tensor,opts,info):
        """
        去噪函数，用于对图像进行去噪。

        参数:
            z0 (torch.Tensor): 输入图像，形状为 (batch_size, L, d)。
            zt (torch.Tensor): 加噪后的图像，形状为 (batch_size, L, d)。
            inp_target (Dict[str, torch.Tensor]): 目标输入字典，包含文本、文本ID和向量等信息。
            mask (torch.Tensor): 掩码张量，形状为 (batch_size, 1, H, W)。
            opts (SamplingOptions): 采样选项，包含生成过程的配置参数。
            info (Dict[str, Any]): 信息字典，包含中间信息。

        Returns:
            torch.Tensor: 去噪后的图像，形状为 (batch_size, 3, height, width)。
        """
        # 计算高度
        h = opts.height // 8
        # 计算宽度
        w = opts.width // 8
        
        # 对掩码进行插值，使其与特征图大小匹配
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        # 将掩码二值化，大于0的像素设为1
        mask[mask > 0] = 1
        
        # 重复掩码以匹配模型输入的通道数（假设为16）
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
      
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        # 将掩码存储在信息字典中
        info['mask'] = mask
        # 生成布尔掩码
        bool_mask = (mask.sum(dim=2) > 0.5)
        # 获取需要处理的索引
        info['mask_indices'] = torch.nonzero(bool_mask)[:,1]
        
        # 获取去噪的时间步长
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))
        # 根据跳步参数调整时间步长
        denoise_timesteps = denoise_timesteps[opts.skip_step:]

        # 获取掩码索引
        mask_indices = info['mask_indices']
        if opts.re_init:
            # 如果需要重新初始化，则添加噪声
            noise = torch.randn_like(zt)
            t  = denoise_timesteps[0]
            zt_noise = z0 *(1 - t) + noise * t
            inp_target["img"] = zt_noise[:, mask_indices,...]
        else:
            # 否则，直接使用加噪后的图像
            inp_target["img"] = zt[:, mask_indices,...]

        # 标记去噪状态
        info['inverse'] = False
        # 调用去噪函数进行去噪
        x, _ = denoise_kv(self.model, **inp_target, timesteps=denoise_timesteps, guidance=opts.denoise_guidance, inverse=False, info=info)

        # 将去噪后的图像替换到原始图像中
        z0[:, mask_indices,...] = z0[:, mask_indices,...] * (1 - info['mask'][:, mask_indices,...]) + x * info['mask'][:, mask_indices,...]
        
        # 解包图像数据
        z0 = unpack(z0.float(),  opts.height, opts.width)

        # 删除信息字典
        del info

        # 返回去噪后的图像
        return z0
