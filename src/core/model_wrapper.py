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
        self.model.eval()

    def prepare_inputs(self, video_path, prompt_text, fps=2.0):
        """将视频路径和文本转换为模型输入 Tensor"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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