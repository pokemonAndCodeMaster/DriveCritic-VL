import os
import torch
import numpy as np
import cv2
from .adaptors.qwen3_adaptor import Qwen3Adaptor
from .processors.visual_processor import VisualProcessor
from .strategies.eagle_strategy import EagleStrategy
from .visualizer import InterpretationVisualizer
from qwen_vl_utils import process_vision_info

class InterpretationEngine:
    """
    Orchestrates the interpretability pipeline for DriveCritic-VL.
    Supports both images and videos using the EAGLE method.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.adaptor = Qwen3Adaptor(model_path, device)
        self.visual_processor = VisualProcessor(mode="grid", patch_size=32)
        self.strategy = EagleStrategy(self.adaptor, lambda1=1.0, lambda2=1.0, batch_size=8)

    def explain(self, path: str, prompt_text: str, target_phrase: str = None, max_steps: int = 64):
        """
        Runs the full EAGLE attribution pipeline on an image or video.
        
        Args:
            path: Path to the input image or video.
            prompt_text: The user prompt.
            target_phrase: The specific phrase in the model's response to explain.
            max_steps: Number of regions to select for the attribution map.
        """
        # Determine if it's a video based on extension
        is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        if is_video:
            return self.explain_video(path, prompt_text, target_phrase, max_steps)
        else:
            return self.explain_image(path, prompt_text, target_phrase, max_steps)

    def explain_image(self, image_path: str, prompt_text: str, target_phrase: str = None, max_steps: int = 64):
        """
        Internal method for image attribution.
        """
        # 1. Initial inference to get the model's response
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        print(f"[Engine] Initial inference for prompt: '{prompt_text}'")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = self.adaptor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.adaptor.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.adaptor.model_device)

        with torch.no_grad():
            output_ids = self.adaptor.model.generate(**inputs, max_new_tokens=128)
            
        response = self.adaptor.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"[Engine] Model Response: '{response}'")

        # 2. Identify target token
        target_token_idx, target_token_id, decoded_tokens = self._find_target_token(output_ids, inputs.input_ids.shape[1], response, target_phrase)
        absolute_target_pos = inputs.input_ids.shape[1] + target_token_idx
        
        print(f"[Engine] Explaining token '{decoded_tokens[target_token_idx]}' at pos {absolute_target_pos}")

        # 3. Setup adaptor
        self.adaptor.set_target(output_ids, absolute_target_pos, target_token_id, prompt_text)

        # 4. Divide image into regions
        print("[Engine] Segmenting image...")
        regions = self.visual_processor.divide_regions(image)
        print(f"[Engine] Created {len(regions)} regions.")

        # 5. Run EAGLE Greedy Search
        print("[Engine] Starting EAGLE Greedy Search...")
        selected_indices, scores = self.strategy.run(image, regions, max_steps=max_steps)

        # 6. Visualize
        print("[Engine] Generating visualization...")
        heatmap = InterpretationVisualizer.generate_heatmap(regions, selected_indices, scores, image.shape)
        overlay = InterpretationVisualizer.overlay_heatmap(image, heatmap)
        
        save_path = "explanation_result.png"
        InterpretationVisualizer.save_comparison(image, overlay, heatmap, save_path)
        
        return {
            "response": response,
            "target_phrase": target_phrase,
            "selected_indices": selected_indices,
            "scores": scores,
            "save_path": save_path
        }

    def explain_video(self, video_path: str, prompt_text: str, target_phrase: str = None, max_steps: int = 32):
        """
        Internal method for video attribution.
        """
        # 1. Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise FileNotFoundError(f"Video not found or empty at {video_path}")
            
        video_np = np.stack(frames) # [T, H, W, 3]
        print(f"[Engine] Loaded video with {len(frames)} frames.")

        # 2. Initial inference to get response
        print(f"[Engine] Initial inference for video prompt: '{prompt_text}'")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        text = self.adaptor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info(messages)
        
        inputs = self.adaptor.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.adaptor.model_device)

        with torch.no_grad():
            output_ids = self.adaptor.model.generate(**inputs, max_new_tokens=128)
            
        response = self.adaptor.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"[Engine] Model Response: '{response}'")

        # 3. Identify target token
        target_token_idx, target_token_id, decoded_tokens = self._find_target_token(output_ids, inputs.input_ids.shape[1], response, target_phrase)
        absolute_target_pos = inputs.input_ids.shape[1] + target_token_idx
        
        print(f"[Engine] Explaining token '{decoded_tokens[target_token_idx]}' at pos {absolute_target_pos}")

        # 4. Setup adaptor
        self.adaptor.set_target(output_ids, absolute_target_pos, target_token_id, prompt_text)

        # 5. Divide video into spatio-temporal regions
        print("[Engine] Segmenting video...")
        self.visual_processor.patch_size = 64 # Use larger patches for video
        regions = self.visual_processor.divide_regions(video_np)
        print(f"[Engine] Created {len(regions)} spatio-temporal regions.")

        # 6. Run EAGLE Greedy Search
        print("[Engine] Starting EAGLE Greedy Search (Video)...")
        selected_indices, scores = self.strategy.run(video_np, regions, max_steps=max_steps)

        # 7. Visualize
        print("[Engine] Generating video visualization...")
        heatmap = InterpretationVisualizer.generate_heatmap(regions, selected_indices, scores, video_np.shape)
        overlay = InterpretationVisualizer.overlay_heatmap(video_np, heatmap)
        
        save_path = "explanation_result_video.mp4"
        InterpretationVisualizer.save_comparison(video_np, overlay, heatmap, save_path)
        
        return {
            "response": response,
            "target_phrase": target_phrase,
            "selected_indices": selected_indices,
            "scores": scores,
            "save_path": save_path
        }

    def _find_target_token(self, output_ids, context_len, response, target_phrase):
        """
        Helper to find the target token index and ID.
        """
        response_tokens = output_ids[0, context_len:]
        decoded_tokens = [self.adaptor.processor.tokenizer.decode([t]) for t in response_tokens]
        
        if target_phrase is None:
            # Default to explaining the last token of the first 5 words
            target_phrase = " ".join(response.split()[:5])
            
        target_token_idx = -1
        phrase_str = ""
        for i, tok in enumerate(decoded_tokens):
            phrase_str += tok
            if target_phrase in phrase_str:
                target_token_idx = i
                break
        
        if target_token_idx == -1:
            target_token_idx = len(response_tokens) - 1
            
        return target_token_idx, response_tokens[target_token_idx].item(), decoded_tokens
