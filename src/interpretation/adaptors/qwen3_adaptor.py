import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2

class Qwen3Adaptor(nn.Module):
    """
    Adaptor for Qwen3-VL to work with EAGLE (Submodular Interpretability).
    Handles logit extraction for specific target tokens given masked visual inputs.
    """
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.device = device
        print(f"[Qwen3Adaptor] Loading model from {model_path}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        self.softmax = nn.Softmax(dim=-1)
        
        # State to be set before calling forward (greedy search loop)
        self.generated_ids = None # Full input_ids up to target
        self.target_token_position = None # Absolute position of target token
        self.target_token_id = None # Token ID to explain
        self.prompt_text = ""
        
    def set_target(self, generated_ids, target_token_position, target_token_id, prompt_text):
        """
        Sets the context for attribution.
        generated_ids: torch.Tensor, the full sequence of tokens (including prompt and generated tokens)
        target_token_position: int, the position of the token to be explained (0-indexed)
        target_token_id: int, the ID of the token at target_token_position
        prompt_text: str, the original prompt text
        """
        self.generated_ids = generated_ids
        self.target_token_position = target_token_position
        self.target_token_id = target_token_id
        self.prompt_text = prompt_text

    def forward(self, inputs):
        """
        Forward pass for one or more (potentially masked) visual inputs.
        Supports both single images and videos.
        
        Args:
            inputs: list of (PIL.Image or torch.Tensor) or a single input.
                    If tensor is 4D (T, H, W, 3), it's treated as a video.
                    If tensor is 3D (H, W, 3), it's treated as an image.
        """
        if not isinstance(inputs, (list, torch.Tensor)):
            inputs = [inputs]

        batch_messages = []
        for visual_input in inputs:
            # Handle Video (4D Tensor or list of images)
            is_video = False
            if isinstance(visual_input, torch.Tensor) and visual_input.ndim == 4:
                is_video = True
            elif isinstance(visual_input, list) and len(visual_input) > 0 and isinstance(visual_input[0], Image.Image):
                is_video = True
            
            content = []
            if is_video:
                if isinstance(visual_input, torch.Tensor):
                    # Convert T,H,W,3 BGR Tensor to list of RGB PIL Images
                    with torch.no_grad():
                        # Swap BGR to RGB
                        video_rgb = visual_input[..., [2, 1, 0]]
                        if video_rgb.max() <= 1.0:
                            video_rgb = (video_rgb * 255).byte()
                        else:
                            video_rgb = video_rgb.byte()
                        
                        frames_np = video_rgb.cpu().numpy()
                        frames_pil = [Image.fromarray(f) for f in frames_np]
                else:
                    frames_pil = visual_input
                
                content.append({"type": "video", "video": frames_pil})
            else:
                # Handle Image
                if isinstance(visual_input, torch.Tensor):
                    with torch.no_grad():
                        image_rgb = visual_input[..., [2, 1, 0]]
                        if image_rgb.max() <= 1.0:
                            image_rgb = (image_rgb * 255).byte()
                        else:
                            image_rgb = image_rgb.byte()
                        image_np = image_rgb.cpu().numpy()
                        image_pil = Image.fromarray(image_np)
                else:
                    image_pil = visual_input
                
                content.append({"type": "image", "image": image_pil})
            
            content.append({"type": "text", "text": self.prompt_text})
            batch_messages.append([{"role": "user", "content": content}])

        # Process vision info for the whole batch
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # Prepare final model inputs
        # Note: video_inputs in Qwen3-VL processor needs special care
        # If it's a batch of videos, it's a list of tensors.
        
        proc_kwargs = {
            "text": texts,
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt"
        }
        
        # Clean up None values to avoid processor errors
        proc_kwargs = {k: v for k, v in proc_kwargs.items() if v is not None}
        
        inputs_dict = self.processor(**proc_kwargs)
        
        # Inject the cached generated_ids up to the target position
        batch_size = len(batch_messages)
        current_input_ids = self.generated_ids[:, :self.target_token_position]
        inputs_dict['input_ids'] = current_input_ids.repeat(batch_size, 1)
        inputs_dict['attention_mask'] = torch.ones_like(inputs_dict['input_ids'])
        inputs_dict = inputs_dict.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict, return_dict=True, use_cache=True)
            last_logits = outputs.logits[:, -1, :] # [batch, vocab_size]
            probs = self.softmax(last_logits)
            target_probs = probs[:, self.target_token_id]
            
        return target_probs

    @property
    def model_device(self):
        return self.model.device
