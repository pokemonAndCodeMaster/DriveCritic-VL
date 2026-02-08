# src/utils/composer/single_view.py

import cv2
import numpy as np
import os
from .base import BaseComposer

class SingleViewComposer(BaseComposer):
    """
    单视角时序合成器。
    适用于 LingoQA 或普通视频帧序列。
    """
    def compose_layout(self, img_source):
        """
        :param img_source: 在 BaseComposer 循环中，这里接收的是单个时间步的数据。
                           对于单视角，这里就是一个图片路径字符串 (str)。
        """
        if not isinstance(img_source, str):
            # 容错：如果意外传来了列表（比如错误配置了NuScenes），取第一个
            if isinstance(img_source, list) and len(img_source) > 0:
                img_source = img_source[0]
            else:
                return None

        if not os.path.exists(img_source):
            return None

        # 直接读取并返回，无需拼接
        return cv2.imread(img_source)