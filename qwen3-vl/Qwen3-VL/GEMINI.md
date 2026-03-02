# Qwen3-VL 项目上下文

## 项目概述
Qwen3-VL 是 Qwen 系列中最新的视觉语言模型，在文本理解、视觉感知、推理和智能体（Agent）交互方面进行了全面升级。它支持多种架构（Dense 和 MoE）以及专门的版本（Instruct 指令微调版和 Thinking 推理增强版）。

### 核心特性：
- **视觉智能体 (Visual Agent)**：能够操作 PC/手机 GUI 并调用工具。
- **视觉编程 (Visual Coding)**：根据视觉输入生成代码（HTML/CSS/JS/Draw.io）。
- **空间推理 (Spatial Reasoning)**：增强的 2D/3D Grounding 和空间感知能力。
- **长文本支持 (Long Context)**：原生支持 256K 上下文，最高可扩展至 1M，支持数小时的视频理解。
- **多模态推理**：在 STEM 和数学视觉推理方面表现强劲。
- **全球化 OCR**：支持 32 种语言及复杂的文档结构解析。

### 核心架构升级：
- **Interleaved-MRoPE**：优化的位置嵌入，增强长视频推理。
- **DeepStack**：融合多级 ViT 特征，实现精细化对齐。
- **文本-时间戳对齐**：视频中精准的时间事件定位。

---

## 目录结构
- `web_demo_mm.py`：基于 Gradio 的主交互界面。
- `qwen-vl-utils/`：视觉处理工具包（图像/视频预处理）。
- `qwen-vl-finetune/`：模型微调框架，支持 SFT 和 LoRA。
- `evaluation/`：评测脚本（MathVision, MMMU, VideoMME 等）。
- `cookbooks/`：Jupyter Notebook 教程，演示 Grounding、OCR 等功能。
- `docker/`：容器化部署的 Dockerfile 和脚本。

---

## 构建与运行

### 环境准备
项目要求 Python 3.8+ 以及 `transformers >= 4.57.0`。
```bash
# 核心依赖
pip install "transformers>=4.57.0" accelerate qwen-vl-utils

# Web 演示依赖
pip install -r requirements_web_demo.txt
```

### Web 演示
启动交互式界面：
```bash
python web_demo_mm.py -c /path/to/model_weights
```

### 模型微调
微调框架位于 `qwen-vl-finetune/`，基于 `torchrun` 和 `deepspeed`。
```bash
# 训练启动示例（参考 qwen-vl-finetune/scripts/）
bash qwen-vl-finetune/scripts/sft_qwen3_4b.sh
```
关键训练参数包括 `--tune_mm_llm` (微调 LLM), `--tune_mm_vision` (微调视觉模块), `--tune_mm_mlp` (微调投影层)。

### 评测
`evaluation/` 下的每个基准测试都有独立的设置。大多数使用 vLLM 进行推理，并使用 LLM（如 GPT-4o）作为评委。
```bash
# 以 MathVision 为例
cd evaluation/MathVision
bash infer_instruct.sh  # 第一阶段：推理
bash eval_instruct.sh   # 第二阶段：GPT-4o 评分
```

---

## 开发规范

### 代码风格与标准
- **框架基础**：主要基于 PyTorch 和 Hugging Face Transformers。
- **视觉处理**：务必使用 `qwen_vl_utils`（特别是 `process_vision_info`）进行统一的图像/视频处理。
- **交错输入**：数据结构通常为包含 `role` 和 `content` 的列表，其中 `content` 包含交错的 `text`, `image`, `video` 类型。
- **Thinking 模型**：专门的“思维”模型会在 `<think>...</think>` 标签内输出推理过程。

### 验证与测试
- **Cookbooks**：通过 `cookbooks/` 中的 Notebook 验证具体功能，如 2D/3D Grounding 或 OCR。
- **结果复现**：官方评测结果可通过 `evaluation/` 目录下的脚本复现。

### 视觉输入限制
- **图像**：通过 `min_pixels` 和 `max_pixels` 控制。
- **视频**：通过 `fps`, `num_frames` 或 `total_pixels` 控制。
- **后端**：推荐使用 `torchcodec` 作为视频解码后端。
