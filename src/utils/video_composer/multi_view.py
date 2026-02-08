import cv2
import numpy as np
from .base import BaseComposer


class NuScenesGridComposer(BaseComposer):
    """
    将 6 视图拼接，并添加视角标签和时间戳
    """

    def compose_layout(self, img_source):
        # 顺序: [Front Left, Front, Front Right, Back Left, Back, Back Right]
        if not isinstance(img_source, list) or len(img_source) < 6:
            return None

        imgs = [cv2.imread(p) for p in img_source]
        if any(img is None for img in imgs): return None

        # 1. 统一 Resize (提升分辨率以看清文字)
        # 单图 800x450 -> 拼接后 2400x900
        target_w, target_h = 800, 450
        resized = [cv2.resize(img, (target_w, target_h)) for img in imgs]

        # 2. 定义标签
        labels = [
            "Front Left", "Front", "Front Right",
            "Back Left", "Back", "Back Right"
        ]

        # 3. 绘制标签 (每个子图内)
        for img, label in zip(resized, labels):
            # 加个半透明黑底背景，保证字能看清
            # 放在子图的底部居中，避免遮挡顶部天空信息
            h, w = img.shape[:2]

            # 标签背景条
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (200, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

            # 白色文字
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        # 4. 拼接
        # 上排: FL, F, FR
        top = np.hstack(resized[:3])
        # 下排: BL, B, BR
        bot = np.hstack(resized[3:])

        grid = np.vstack([top, bot])
        return grid

    def draw_overlays(self, image, meta, t_idx):
        """覆盖时间戳 (移到底部)"""
        h, w = image.shape[:2]

        # 计算时间戳 (假设 2Hz)
        timestamp = t_idx * (1.0 / self.fps)
        text = f"Time: {timestamp:.1f}s | Frame: {t_idx}"

        # 背景条 (放在整个画面的最底部)
        # 黑色实底，高 60 像素
        cv2.rectangle(image, (0, h - 60), (500, h), (0, 0, 0), -1)

        # 黄色文字
        cv2.putText(image, text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 3)

        return image


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