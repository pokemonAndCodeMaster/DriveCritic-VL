import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenModelWrapper:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        print(f"[Model] Loading: {self.cfg['path']}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg['path'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager" if self.cfg.get('viz_enabled', True) else "flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg['path'])
        # Qwen 的 Processor 通常有两个地方限制像素
        # 我们把它们都设为一个巨大的值 (比如 1000万像素)
        # huge_pixel_limit = 1280 * 28 * 28 * 20  # 约 2000万像素
        #
        # if hasattr(self.processor, "image_processor"):
        #     # 1. 解锁 max_pixels (针对图片)
        #     if hasattr(self.processor.image_processor, "max_pixels"):
        #         print(
        #             f"[Model] Unlocking image_processor.max_pixels limit: {self.processor.image_processor.max_pixels} -> {huge_pixel_limit}")
        #         self.processor.image_processor.max_pixels = huge_pixel_limit
        #
        #     # 2. 解锁 video_max_pixels (针对视频) - 这就是导致你报错的元凶
        #     if hasattr(self.processor.image_processor, "video_max_pixels"):
        #         print(
        #             f"[Model] Unlocking image_processor.video_max_pixels limit: {self.processor.image_processor.video_max_pixels} -> {huge_pixel_limit}")
        #         self.processor.image_processor.video_max_pixels = huge_pixel_limit
        #
        #     # 3. 还有可能有 fps 限制
        #     if hasattr(self.processor.image_processor, "fps"):
        #         self.processor.image_processor.fps = 2.0  # 强制默认 fps

        self.model.eval()

    def prepare_inputs(self, video_path, prompt_text):
        """将视频路径和文本转换为模型输入 Tensor"""

        # 1. 从配置读取参数，给默认值兜底
        sample_fps = self.cfg.get('sample_fps', 1.0)
        min_pixels = self.cfg.get('min_pixels', 256 * 28 * 28)
        # max_pixels = self.cfg.get('max_pixels', 1280 * 28 * 28)
        total_pixels = self.cfg.get('total_pixels', 20480 * 32 * 32)

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": sample_fps,
                    "sample_fps": sample_fps,
                    # 动态注入配置
                    "min_pixels": min_pixels,
                    # "max_pixels": max_pixels,
                    "total_pixels": total_pixels
                },
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )
        # 分离视频数据和元数据
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        return inputs

    def generate(self, inputs):
        """执行推理"""
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.get('max_new_tokens', 512)
            )

        # Trim tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    def forward_for_viz(self, inputs):
        """为了可视化，单独跑一次 Forward 获取 Attention"""
        with torch.no_grad():
            return self.model(**inputs, output_attentions=True)
