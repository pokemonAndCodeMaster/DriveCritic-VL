# DriveCritic-VL 项目概览

DriveCritic-VL 是一个专为自动驾驶场景分析设计的视觉-语言模型（VLM）框架。它基于 **Qwen3-VL** 架构，能够理解多视角的驾驶视频数据，并提供驾驶行为分析、场景描述以及对自我车辆（Ego-Car）的决策评价。

## 🚀 快速开始 (Quick Start)

### 1. 推理功能 (Inference)

本项目提供两种推理模式：**基础脚本模式**（适合快速测试）和**流水线模式**（适合生产与科研）。

#### A. 基础脚本模式 (infer_base.py)
适合无需复杂配置的单视频快速测试。

*   **单视频推理：**
    ```bash
    python infer_base.py single --input_path assets/demo.mp4 --prompt_file prompt_demo.txt
    ```
*   **文件夹批量推理：**
    ```bash
    python infer_base.py folder --input_path data/my_videos/ --output_path results.json
    ```

#### B. 流水线模式 (main.py) [推荐]
基于 `configs/inference.yaml` 配置文件的完整流水线，支持自动数据加载、视频合成和注意力可视化。

*   **单样本调试 (Single Mode)：**
    通过 ID 运行特定样本（自动查找并加载其多视角图片/视频数据）。
    ```bash
    python main.py --config configs/inference.yaml --mode single --id scene-0061
    ```
*   **全量数据集评测 (Dataset Mode)：**
    运行配置文件中指定的数据集全集。
    ```bash
    python main.py --config configs/inference.yaml --mode dataset
    ```

---

### 2. 模块使用指南 (Module Guide)

#### 📦 数据加载与准备 (Data Loader & Preparation)
核心代码位于 `src/data/` 和 `src/tools/`。

*   **生成视频缓存：**
    自动驾驶数据集通常是图片序列。使用此工具将它们预处理为 MP4 视频，加速后续训练/推理。
    ```bash
    # 为 nuscenes 数据集生成视频
    python src/tools/prepare_videos.py --config configs/datasets.yaml --dataset_key nuscenes --fps 2
    ```
*   **检查数据集完整性：**
    ```bash
    python src/tools/check_datasets.py
    ```

#### 🖼️ 多视角视频合成 (Video Composer)
核心代码位于 `src/utils/video_composer/`。
该模块负责将 NuScenes/DriveLM 的 6 路环视相机图片拼接成单个视频帧。
*   **自动触发：** 在 `main.py` 中，如果检测到本地没有视频缓存，流水线会自动调用 Composer 实时合成。
*   **手动触发：** 使用上面的 `prepare_videos.py` 工具。

#### 🧠 注意力可视化 (Attention Visualization)
核心代码位于 `src/visualizer.py`。
想要查看模型关注画面的哪些区域（如行人、红绿灯）：
1.  修改 `configs/inference.yaml`：
    ```yaml
    visualization:
      enabled: true
      save_dir: "heatmaps"
      keywords: ["pedestrian", "traffic light"] # 关注的关键词
    ```
2.  运行 `main.py`，结果将保存在 `outputs/heatmaps/`。

---

## 🛠 功能扩展指南 (Extension Guide)

如果你需要添加新的数据集或视频处理逻辑，请参考以下规范：

### 1. 添加新数据集 (Dataset)
1.  **实现类：** 在 `src/data/datasets.py` 中继承 `BaseDataset`。
2.  **实现 `_load_data` 方法：** 读取你的数据源，并将每个样本封装为 `DataSample` 对象。
    *   `image_paths`: 图片路径列表（如果是多视角，则是列表的列表）。
    *   `qa_pairs`: QA 对。
3.  **注册：** 在 `src/data/factory.py` 的 `_REGISTRY` 中注册你的新类名。
4.  **配置：** 在 `configs/datasets.yaml` 中添加对应的数据集配置。

### 2. 添加新视频合成布局 (Layout)
1.  **实现类：** 在 `src/utils/video_composer/` 中新建文件或在现有文件中继承 `BaseComposer`。
2.  **实现 `compose_layout` 方法：** 输入图片列表，输出拼接好的 `numpy` 图像（H, W, 3）。
3.  **注册：** 在 `src/utils/video_composer/__init__.py` 的 `get_composer` 函数中添加映射。

---

## 📂 项目结构 (Structure)

*   `src/core/`: 核心引擎
    *   `model_wrapper.py`: Qwen 模型封装，处理输入 Tensor 构建。
    *   `pipeline.py`: 推理主循环，串联数据、模型和可视化。
*   `src/data/`: 数据管理
    *   `datasets.py`: 各类数据集加载逻辑 (NuScenes, DriveLM 等)。
    *   `factory.py`: 简单工厂模式，统一创建数据集实例。
*   `src/utils/`: 工具库
    *   `video_composer/`: 视频拼接与合成逻辑。
*   `configs/`: 配置文件
    *   `inference.yaml`: 推理参数（模型路径、Prompt、可视化开关）。
    *   `datasets.yaml`: 数据集路径与元数据。
*   `LLaMA-Factory/`: 训练子模块 (Submodule)。
*   `outputs/`: 默认输出目录。

## 📝 开发规范
*   **配置优先：** 尽量通过修改 `yaml` 文件来调整参数，避免硬编码。
*   **调试技巧：** 遇到模型输入相关问题，查看 `outputs/debug_inspect/` 下生成的 `model_seen_input.mp4`，这是模型实际“看到”的视频内容。
