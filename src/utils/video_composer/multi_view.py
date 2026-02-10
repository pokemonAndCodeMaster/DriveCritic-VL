import cv2
import numpy as np
from .base import BaseComposer


class NuScenesGridComposer(BaseComposer):
    """
    将 6 视图拼接 (输入顺序已在 Dataset 层统一为 FL, F, FR, BL, B, BR)
    """

    def compose_layout(self, img_source):
        # 此时 img_source 必定是:
        # [Front Left, Front, Front Right, Back Left, Back, Back Right]

        if not isinstance(img_source, list) or len(img_source) < 6:
            return None

        imgs = [cv2.imread(p) for p in img_source]
        if any(img is None for img in imgs): return None

        # 1. 统一 Resize
        target_w, target_h = 800, 450
        resized = [cv2.resize(img, (target_w, target_h)) for img in imgs]

        # 2. 定义标签 (与 Dataset 输出顺序严格对应)
        labels = [
            "Front Left", "Front", "Front Right",
            "Back Left", "Back", "Back Right"
        ]

        # 3. 绘制标签
        for img, label in zip(resized, labels):
            h, w = img.shape[:2]
            overlay = img.copy()

            # 标签背景条：放在底部居中，半透明黑色
            # 避免遮挡上方天空，也避免遮挡左上角可能存在的原始数据
            cv2.rectangle(overlay, (0, h - 40), (220, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # 文字
            cv2.putText(img, label, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        # 4. 拼接 (直接按顺序切分)
        # 上排: Indices 0, 1, 2
        top = np.hstack(resized[:3])
        # 下排: Indices 3, 4, 5
        bot = np.hstack(resized[3:])

        grid = np.vstack([top, bot])
        return grid

    def draw_overlays(self, image, meta, t_idx):
        """覆盖时间戳 (放在整个画面的左上角，显眼位置)"""
        h, w = image.shape[:2]
        timestamp = t_idx * (1.0 / self.fps)
        text = f"T: {timestamp:.1f}s | F: {t_idx}"

        # 大背景条 (顶部)
        cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
        # 亮黄色文字
        cv2.putText(image, text, (30, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 255), 2)

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