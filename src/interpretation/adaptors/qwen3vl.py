import torch
import torch.nn as nn
from qwen_vl_utils import process_vision_info
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from typing import List, Union, Optional

class Qwen3VLAdaptor(nn.Module):
    """
    Qwen3-VL 模型适配器，用于 EAGLE 归因分析。
    实现了将图像输入映射到目标 Token 概率预测的逻辑。
    """
    def __init__(self, model: nn.Module, processor: torch.nn.Module, device: str = "cuda"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        
        # 归因状态变量
        self.generated_ids = None
        self.target_token_positions = None
        self.target_token_ids = None
        self.prompt = None

    def set_context(self, 
                   prompt: str, 
                   generated_ids: torch.Tensor, 
                   target_token_positions: Union[List[int], np.ndarray], 
                   target_token_ids: Union[List[int], np.ndarray]):
        """
        设置归因上下文。
        
        Args:
            prompt: 原始用户提示词。
            generated_ids: 完整的输入 ID（包含上下文和已生成的回答）。
            target_token_positions: 需要解释的 Token 在 generated_ids 中的位置。
            target_token_ids: 这些 Token 对应的词表 ID。
        """
        self.prompt = prompt
        self.generated_ids = generated_ids.to(self.device)
        self.target_token_positions = np.array(target_token_positions)
        self.target_token_ids = torch.tensor(target_token_ids).to(self.device)

    def forward(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        EAGLE 调用的前向传播方法。
        根据输入图像（可能是经过掩码处理的），返回目标 Token 的概率。
        
        Args:
            image: PIL 图像或张量。
            
        Returns:
            torch.Tensor: 目标 Token 的概率分布，形状为 [num_targets]。
        """
        # 将张量转换为 PIL 图像（适应 EAGLE 的 Submodular Solver）
        if isinstance(image, torch.Tensor):
            if image.max() <= 1.0:
                image = image * 255
            
            if image.dim() == 4: # [B, C, H, W]
                image = image.squeeze(0)
                
            if image.shape[0] == 3: # [C, H, W]
                pass
            elif image.shape[-1] == 3: # [H, W, C]
                image = image.permute(2, 0, 1)
            
            image = image.byte().cpu()
            image = TF.to_pil_image(image)

        # 准备 Qwen3-VL 输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 教师强制 (Teacher Forcing)：使用预生成的序列覆盖 input_ids
        # 确保 input_ids 长度一致
        inputs['input_ids'] = self.generated_ids
        inputs['attention_mask'] = torch.ones_like(self.generated_ids)
        
        # 将所有张量移动到设备上
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # [batch, seq_len, vocab_size]

        # 提取目标 Token 的 Logits
        # 预测位置通常在目标 Token 所在位置的前一位
        target_logits = logits[:, self.target_token_positions - 1] 
        probs = self.softmax(target_logits) # [batch, num_targets, vocab_size]

        # 获取特定词表 ID 的概率
        # indices 形状: [1, num_targets, 1]
        indices = self.target_token_ids.unsqueeze(0).unsqueeze(-1) 
        target_probs = probs.gather(dim=2, index=indices).squeeze(-1) # [1, num_targets]

        return target_probs[0] # [num_targets]
