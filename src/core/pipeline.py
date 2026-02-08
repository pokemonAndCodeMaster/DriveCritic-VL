import os
import torch
from src.data.base import DataSample


class InferencePipeline:
    def __init__(self, model_wrapper, composer, visualizer=None):
        self.model = model_wrapper
        self.composer = composer
        self.viz = visualizer

    def run_sample(self, sample: DataSample):
        """处理单个样本的全流程"""
        result = {
            "id": sample.id,
            "prompt": sample.get_prompt_text(),
            "video_path": None,
            "prediction": None,
            "error": None
        }

        try:
            # 1. 准备视频 (懒加载/实时生成)
            video_path = self._prepare_video_resource(sample)
            if not video_path:
                result['error'] = "Video generation failed"
                return result

            result['video_path'] = video_path

            # 2. 准备模型输入
            inputs = self.model.prepare_inputs(video_path, result['prompt'])

            # 3. 推理生成
            output_text = self.model.generate(inputs)
            result['prediction'] = output_text

            # 4. 可视化 (可选)
            if self.viz:
                self._run_visualization(sample.id, video_path, inputs)

        except Exception as e:
            print(f"[Pipeline Error] Sample {sample.id}: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)

        return result

    def _prepare_video_resource(self, sample):
        """处理 DataSample 的多态输入 (List -> MP4)"""
        video_input = sample.get_model_input()

        # 如果是图片列表，调用 composer 转 MP4
        if isinstance(video_input, list):
            return self.composer.process(sample)

        # 如果已经是路径，直接返回
        if isinstance(video_input, str) and os.path.exists(video_input):
            return video_input

        return None

    def _run_visualization(self, task_id, video_path, inputs):
        """执行可视化逻辑"""
        try:
            outputs = self.model.forward_for_viz(inputs)
            self.viz.process(
                task_id=task_id,
                video_path=video_path,
                engine_inputs=inputs,
                model_outputs=outputs
            )
        except Exception as e:
            print(f"[Viz Warn] Failed to visualize {task_id}: {e}")
            import traceback
            traceback.print_exc()