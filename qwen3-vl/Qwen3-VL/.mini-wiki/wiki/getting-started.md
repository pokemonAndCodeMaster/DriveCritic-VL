# 快速开始 (Getting Started)

本指南将帮助您在本地环境快速搭建 Qwen3-VL，并运行您的第一个多模态推理示例。

## 📋 环境要求

| 组件 | 最低要求 | 推荐配置 |
| :--- | :--- | :--- |
| **Python** | 3.8+ | 3.10+ |
| **Transformers** | 4.57.0+ | 最新版本 |
| **PyTorch** | 2.1.0+ | 2.4.0+ (CUDA 12.x) |
| **GPU 显存** | 8GB (2B-Int4) | 24GB+ (8B/32B BF16) |

## ⚙️ 安装步骤

### 1. 克隆项目并安装依赖
```bash
git clone https://github.com/QwenLM/Qwen3-VL.git
cd Qwen3-VL
pip install -r requirements_web_demo.txt
```

### 2. 核心包安装 (Hugging Face)
```bash
pip install "transformers>=4.57.0" accelerate qwen-vl-utils
```

### 3. 加速库 (可选但推荐)
为了在处理多图或视频时获得更好的性能，建议安装 **Flash-Attention 2**：
```bash
pip install -U flash-attn --no-build-isolation
```

## 🚀 第一个示例：图像描述

您可以直接使用以下脚本来测试 Qwen3-VL 的基础图像理解能力。

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 1. 初始化模型与处理器 (以 2B 指令微调版为例)
model_id = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 准备交错输入
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "图中描述了什么场景？请详细说明。"},
        ],
    }
]

# 3. 处理输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = process_vision_info(messages, image_patch_size=16)

inputs = processor(
    text=text, 
    images=images, 
    videos=videos, 
    do_resize=False, 
    return_tensors="pt"
).to(model.device)

# 4. 生成回复
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("-" * 20)
print(output_text[0])
```

## 🎬 进阶：视频理解

Qwen3-VL 处理视频的方式非常简洁，您只需将 `type` 改为 `video` 并提供视频路径。

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video", 
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
                "fps": 2.0  # 每秒抽样帧数
            },
            {"type": "text", "text": "这段视频的内容是什么？"},
        ],
    }
]
# 处理逻辑同上
```

## 🌐 交互式 Web Demo

项目内置了一个基于 Gradio 的 Web 界面，方便您直观体验模型的各种能力。

```bash
# 确保已安装 web_demo 依赖
pip install -r requirements_web_demo.txt

# 启动 Web UI
python web_demo_mm.py -c /path/to/your/model_weights
```

## ❓ 常见问题 FAQ

**Q: 显存不足 (OOM) 怎么办？**
- 尝试使用 `torch_dtype=torch.float16` 或加载量化后的模型版本（如 FP8/AWQ）。
- 降低视觉输入的像素：在 `process_vision_info` 中设置 `max_pixels`。

**Q: 如何启用 Thinking (思维) 模式？**
- 请加载带有 `-Thinking` 后缀的模型（如 `Qwen3-VL-2B-Thinking`）。这类模型会自动在回复前输出推理链。

## 💡 下一步计划
- [阅读 Cookbooks](https://github.com/QwenLM/Qwen3-VL/tree/main/cookbooks): 学习 2D/3D 定位、OCR、Agent 等高级用法。
- [微调指南](qwen-vl-finetune/README.md): 了解如何根据自己的数据微调 Qwen3-VL。

---
**相关文档**
- [项目首页](index.md)
- [系统架构](architecture.md)

*由 [Mini-Wiki v3.0.6](https://github.com/trsoliu/mini-wiki) 自动生成 | 2026-03-01*
