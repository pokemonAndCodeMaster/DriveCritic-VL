# DriveCritic-VL 项目文档

## 1. 项目简介
DriveCritic-VL 是一个基于 Qwen3-VL 的驾驶场景理解与自动评估系统。它不仅能够对复杂的交通场景进行描述和决策建议，还通过集成 EAGLE 解释性算法，深入分析模型决策背后的视觉依据，从而提升自动驾驶决策的可解释性并辅助幻觉检测。

## 2. 核心架构
项目采用模块化设计，主要分为以下层次：
- **Core 层**: 模型加载与推理封装 (`QwenModelWrapper`)。
- **Data 层**: 多样化数据集集成 (`nuscenes`, `LingoQA`)。
- **Interpretation 层**: 解释性引擎，负责生成决策依据的热力图。
- **Pipeline 层**: 串联数据加载、模型推理与结果保存的自动化流水线。

## 3. 快速开始
### 环境要求
- Python 3.11+
- Transformers >= 4.57.0
- qwen-vl-utils
- opencv-python-headless

### 基础用法
运行全量推理：
```bash
python main.py --config configs/inference.yaml --mode dataset
```

运行归因分析演示：
```bash
python debug/demo_explain.py
```

## 4. 技术特性
- **高性能归因**: 利用 GPU 张量加速，支持大规模区域分割下的快速归因。
- **时空归因**: 业内领先的视频级解释能力，可定位视频中特定时间段、特定空间的决策依据。
- **Clean Code 设计**: 严格遵循适配器模式与策略模式，易于扩展至其他 VLM 模型。
