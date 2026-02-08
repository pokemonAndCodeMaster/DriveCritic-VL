import cv2
import os
import numpy as np
from tqdm import tqdm


class VideoComposer:
    def __init__(self, output_dir, fps=2):
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def process_sample(self, sample: 'DataSample', layout_mode='single', draw_func=None) -> str:
        """
        将 DataSample 的图片转为视频。
        :param layout_mode: 'single' (单视角序列) 或 'nusc_6view' (NuScenes 6视图拼接)
        :param draw_func: 自定义回调函数，接收 (image, meta_info) 用于画框
        :return: 生成的视频绝对路径
        """
        save_path = os.path.join(self.output_dir, f"{sample.id}.mp4")
        if os.path.exists(save_path):
            return save_path  # 缓存命中

        # 准备写入器
        writer = None

        # 展平时间步
        time_steps = sample.image_paths

        for t_idx, img_source in enumerate(time_steps):
            frame = self._compose_frame(img_source, layout_mode)

            # 自定义绘制 (例如画 2D Box)
            if draw_func:
                frame = draw_func(frame, sample.meta_info, t_idx)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(save_path, fourcc, self.fps, (w, h))

            writer.write(frame)

        if writer:
            writer.release()

        return save_path

    def _compose_frame(self, img_source, mode):
        """处理单帧的图片组合"""
        if mode == 'single':
            # img_source 是单个路径
            path = img_source if isinstance(img_source, str) else img_source[0]
            return cv2.imread(path)

        elif mode == 'nusc_6view':
            # img_source 是 6 个路径的列表 [Front, FL, FR, Back, BL, BR]
            # 这里做一个简单的 2x3 拼接示例
            # 注意：实际顺序需要根据 Dataset 里的读取顺序对应
            images = [cv2.resize(cv2.imread(p), (800, 450)) for p in img_source]

            # 拼图逻辑: 上三张，下三张
            top = np.hstack(images[:3])
            bot = np.hstack(images[3:])
            return np.vstack((top, bot))

        return np.zeros((100, 100, 3), dtype=np.uint8)