from abc import ABC, abstractmethod
import cv2
import os
import numpy as np


class BaseComposer(ABC):
    def __init__(self, output_dir, fps=2):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def process(self, sample) -> str:
        """模板方法：处理视频生成流程"""
        save_path = os.path.join(self.output_dir, f"{sample.id}.mp4")
        if os.path.exists(save_path):
            return save_path

        frames = []
        # sample.image_paths 可能是 List[str] 也可能是 List[List[str]]
        for t_idx, img_source in enumerate(sample.image_paths):
            # 1. 布局合成 (Layout)
            canvas = self.compose_layout(img_source)
            if canvas is None: continue

            # 2. 元素绘制 (Overlay: 时间戳、框、辅助线)
            canvas = self.draw_overlays(canvas, sample.meta_info, t_idx)
            frames.append(canvas)

        if not frames: return None

        # 写入视频
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, self.fps, (w, h))
        for f in frames: writer.write(f)
        writer.release()

        return save_path

    @abstractmethod
    def compose_layout(self, img_source) -> np.ndarray:
        """子类实现：如何把输入的图片源（单张或多张）拼成一帧"""
        pass

    def draw_overlays(self, image, meta, t_idx):
        """通用绘制逻辑：画时间戳、帧号等"""
        # 默认画个时间戳
        h, w = image.shape[:2]
        text = f"Frame: {t_idx}"
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 如果 meta 里有 2d_box，也可以在这里画
        # boxes = meta.get('boxes', []) ...
        return image