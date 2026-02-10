import os
import json
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


# ==========================================
# 1. 统一数据样本容器 (DataSample)
# ==========================================
@dataclass
class DataSample:
    """
    通用数据样本。
    兼容：单帧图片 / 多帧图片列表 / 预生成的视频文件 / QA问答对
    """
    id: str

    # 核心媒体数据
    # 如果是 NuScenes，这里存的是 6张图的路径列表 或 连续帧的列表
    image_paths: List[Union[str, List[str]]]

    # 预生成的视频路径 (可选)
    video_path: Optional[str] = None

    # 文本/标签数据
    qa_pairs: List[Dict[str, str]] = field(default_factory=list)  # [{'q': '...', 'a': '...'}]
    description: str = ""  # 用于 Caption 任务或关键信息描述

    # 元数据 (用于画框、相机参数、时间戳等)
    meta_info: Dict[str, Any] = field(default_factory=dict)

    def get_model_input(self):
        """
        [智能加载逻辑]
        如果有生成的视频且文件存在 -> 返回视频路径 (str)
        否则 -> 返回原始图片路径列表 (List)
        """
        if self.video_path and os.path.exists(self.video_path):
            return self.video_path
        return self.image_paths

    def get_prompt_text(self):
        """
        将 QA 对拼接成模型输入的 Prompt 格式
        """
        if not self.qa_pairs:
            return self.description

        # 简单拼接示例，具体格式可根据模型要求调整
        conversation = []
        for qa in self.qa_pairs:
            conversation.append(f"Q: {qa['q']}\nA: {qa['a']}")
        return "\n".join(conversation)


# ==========================================
# 2. 视频索引管理器 (VideoIndexManager)
# ==========================================
class VideoIndexManager:
    """管理 Sample ID 到 Video Path 的映射"""

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.mapping = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self.mapping = json.load(f)
            except Exception as e:
                print(f"[Warn] Failed to load video index: {e}")

    def get_video_path(self, sample_id: str) -> Optional[str]:
        return self.mapping.get(str(sample_id))

    def update(self, sample_id: str, video_path: str):
        self.mapping[str(sample_id)] = video_path

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(self.mapping, f, indent=2)


# ==========================================
# 3. 数据集抽象基类 (BaseDataset)
# ==========================================
class BaseDataset(Dataset, ABC):
    def __init__(self, cfg: dict):
        """
        :param cfg: 来自 config.yaml 的 dataset 部分配置
        """
        self.cfg = cfg
        self.data_root = cfg.get('data_root', '')
        self.samples: List[DataSample] = []

        # 尝试初始化视频索引管理器 (默认在 data_root 下找 video_index.json)
        # 子类可以覆盖这个路径
        index_file = cfg.get('video_index_file', 'video_index.json')
        self.indexer = VideoIndexManager(os.path.join(self.data_root, index_file))

        # 执行加载逻辑
        self._load_data()

    @abstractmethod
    def _load_data(self):
        """解析文件结构，填充 self.samples"""
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> DataSample:
        return self.samples[idx]