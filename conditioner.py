from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    """
    HFEmbedder 类，用于加载预训练的Hugging Face模型并生成文本嵌入。

    参数:
        version (str): 预训练模型的版本或名称，例如 'black-forest-labs/FLUX.1-dev'。
        max_length (int): 文本的最大长度，用于截断或填充文本。
        is_clip (bool): 是否使用CLIP模型。如果为True，则使用CLIP模型；否则，使用T5模型。
        **hf_kwargs: 其他关键字参数，传递给Hugging Face模型的 `from_pretrained` 方法。
    """
    def __init__(self, version: str, max_length: int, is_clip, **hf_kwargs):
        super().__init__()
        # 是否使用CLIP模型
        self.is_clip = is_clip
        # 文本的最大长度
        self.max_length = max_length
        # 根据是否使用CLIP模型，设置输出的键名
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if version == 'black-forest-labs/FLUX.1-dev':
            if self.is_clip:
                # 如果使用CLIP模型，加载CLIP分词器和CLIP文本编码器模型
                self.tokenizer: T5Tokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length, subfolder="tokenizer")
                self.hf_module: T5EncoderModel = CLIPTextModel.from_pretrained(version,subfolder='text_encoder' , **hf_kwargs)
            else:
                # 如果不使用CLIP模型，加载T5分词器和T5编码器模型
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length, subfolder="tokenizer_2")
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version,subfolder='text_encoder_2' , **hf_kwargs)
        else:
            if self.is_clip:
                # 如果使用CLIP模型，加载CLIP分词器和CLIP文本编码器模型
                self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
                self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
            else:
                # 如果不使用CLIP模型，加载T5分词器和T5编码器模型
                self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
                self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)
        # 将模型设置为评估模式，并冻结其参数
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        """
        前向传播函数，用于生成文本嵌入。

        参数:
            text (List[str]): 输入的文本列表。

        返回:
            torch.Tensor: 生成的文本嵌入，形状为 (batch_size, hidden_dim)。
        """
        # 对输入文本进行编码，生成输入张量
        batch_encoding = self.tokenizer(
            text,
            truncation=True, # 是否截断文本
            max_length=self.max_length, # 最大序列长度
            return_length=False, # 是否返回序列长度
            return_overflowing_tokens=False, # 是否返回溢出的标记
            padding="max_length", # 填充方式，填充到最大长度
            return_tensors="pt", # 返回的格式为PyTorch张量
        )
        
        # 获取模型的输出
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device), # 将输入张量移动到模型所在的设备（CPU或GPU）
            attention_mask=None, # 如果模型需要注意力掩码，则传递，否则为None
            output_hidden_states=False, # 是否返回隐藏状态
        )
        
        return outputs[self.output_key]
