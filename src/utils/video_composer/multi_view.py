import cv2
import numpy as np
from .base import BaseComposer


class NuScenesGridComposer(BaseComposer):
    """
    将 NuScenes 的 6 个摄像头数据拼接成 2x3 网格。
    输入 img_source 预期为 list: [Front, FL, FR, Back, BL, BR] (顺序需与Dataset一致)
    """

    def compose_layout(self, img_source):
        if not isinstance(img_source, list) or len(img_source) < 6:
            return None  # 数据格式不对

        # 加载图片
        imgs = [cv2.imread(p) for p in img_source]
        if any(img is None for img in imgs): return None

        # 统一 Resize 尺寸
        target_w, target_h = 800, 450
        resized = [cv2.resize(img, (target_w, target_h)) for img in imgs]

        # 布局逻辑:
        # 上排: FrontLeft, Front, FrontRight
        # 下排: BackLeft,  Back,  BackRight
        # 注意: Dataset 传进来的顺序至关重要，这里假设是 [F, FL, FR, B, BL, BR]
        # 根据实际情况调整索引
        top = np.hstack([resized[1], resized[0], resized[2]])
        bot = np.hstack([resized[4], resized[3], resized[5]])

        grid = np.vstack([top, bot])
        return grid


class NuScenesBEVComposer(NuScenesGridComposer):
    """
    高级版：左边是 6 视图拼接，右边是生成的 BEV 视图
    """

    def compose_layout(self, img_source):
        # 1. 先拿网格图
        grid = super().compose_layout(img_source)
        if grid is None: return None

        # 2. 生成或读取 BEV (此处为模拟)
        h, w = grid.shape[:2]
        bev_canvas = np.zeros((h, h, 3), dtype=np.uint8)  # 正方形 BEV
        cv2.putText(bev_canvas, "BEV MAP PLACEHOLDER", (50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # 3. 左右拼接
        final = np.hstack([grid, bev_canvas])
        return final