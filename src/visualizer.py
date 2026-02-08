import os
import torch
import numpy as np
import cv2
import matplotlib
import math
from typing import List, Dict

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class AttentionVisualizer:
    def __init__(self, viz_config: Dict, tokenizer):
        self.cfg = viz_config
        self.tokenizer = tokenizer
        self.font_prop = self._check_and_load_font()

        try:
            self.VISION_START_ID = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.VISION_END_ID = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except:
            self.VISION_START_ID = 151652
            self.VISION_END_ID = 151653

    def _check_and_load_font(self):
        font_paths = [
            self.cfg.get('font_path', 'assets/SimHei.ttf'),
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
            'C:/Windows/Fonts/simhei.ttf'
        ]
        for fp in font_paths:
            if os.path.exists(fp): return FontProperties(fname=fp)
        return None

    def _find_keyword_indices(self, input_ids, prompt_start_idx):
        keywords = self.cfg.get('keywords', [])
        if not keywords: return {}

        if isinstance(input_ids, torch.Tensor):
            full_ids = input_ids.tolist()
        else:
            full_ids = list(input_ids)

        prompt_ids = full_ids[prompt_start_idx:]
        kw_map = {}

        for kw in keywords:
            kw_tokens = self.tokenizer.encode(kw, add_special_tokens=False)
            if not kw_tokens: continue
            indices = []
            len_kw = len(kw_tokens)
            for i in range(len(prompt_ids) - len_kw + 1):
                if prompt_ids[i: i + len_kw] == kw_tokens:
                    indices.extend([x + prompt_start_idx + i for x in range(len_kw)])
            if indices: kw_map[kw] = list(set(indices))
        return kw_map

    def _generate_heatmap(self, frame, attn_flat, grid_hw, frame_idx):
        """
        生成热力图 (带详细维测)
        """
        try:
            num_tokens = attn_flat.shape[0]
            if num_tokens == 0: return frame

            h_img, w_img = frame.shape[:2]
            h_grid, w_grid = grid_hw
            expected = h_grid * w_grid

            # === [维测点 4] Heatmap 形状检查 ===
            if num_tokens != expected:
                print(f"[Viz ERR] Frame {frame_idx} Heatmap Mismatch!")
                print(f"   > Attn Flat Shape: {num_tokens}")
                print(f"   > Target Grid:     {h_grid} x {w_grid} = {expected}")
                print(f"   > Ratio:           {expected / (num_tokens + 1e-5):.2f}")

                # 为了防止报错导致程序中断，尝试截断或跳过，但记录错误
                if num_tokens > expected:
                    attn_flat = attn_flat[:expected]
                else:
                    return frame

            attn_2d = attn_flat.reshape(h_grid, w_grid).float().cpu().numpy()
            attn_norm = attn_2d - attn_2d.min()
            attn_norm = attn_norm / (attn_norm.max() + 1e-9)
            attn_uint8 = (attn_norm * 255).astype(np.uint8)
            attn_resized = cv2.resize(attn_uint8, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
            return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        except Exception as e:
            print(f"[Viz ERR] Heatmap generation failed: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def _create_summary_grid(self, image_paths, save_path):
        if not image_paths: return
        try:
            img0 = cv2.imread(image_paths[0])
            h, w, c = img0.shape
            n = len(image_paths)
            cols = 2
            rows = math.ceil(n / cols)
            canvas = np.ones((rows * h, cols * w, c), dtype=np.uint8) * 255
            for i, p in enumerate(image_paths):
                img = cv2.imread(p)
                if img is None: continue
                r, c_idx = i // cols, i % cols
                y_s, y_e = r * h, (r + 1) * h
                x_s, x_e = c_idx * w, (c_idx + 1) * w
                if img.shape != (h, w, c): img = cv2.resize(img, (w, h))
                canvas[y_s:y_e, x_s:x_e] = img
            cv2.imwrite(save_path, canvas)
        except:
            pass

    def process(self, task_id, video_path, engine_inputs, model_outputs):
        if not self.cfg.get('enabled', True): return

        print(f"\n[Viz Debug] ========== START TASK: {task_id} ==========")
        base_dir = self.cfg.get('output_dir', 'outputs/viz_results')
        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # 1. Input IDs 解析
        input_ids = engine_inputs.input_ids[0]
        try:
            v_start_indices = (input_ids == self.VISION_START_ID).nonzero(as_tuple=True)[0]
            v_end_indices = (input_ids == self.VISION_END_ID).nonzero(as_tuple=True)[0]

            if len(v_start_indices) == 0: return

            v_start_idx = v_start_indices[0].item() + 1
            v_end_idx = v_end_indices[-1].item()
            prompt_start_idx = v_end_idx + 1

            total_vis_tokens = v_end_idx - v_start_idx
            print(f"[Viz Debug] Input Structure: Start={v_start_idx}, End={v_end_idx}, Count={total_vis_tokens}")

        except Exception as e:
            print(f"[Viz ERR] Token parsing failed: {e}")
            return

        kw_map = self._find_keyword_indices(input_ids, prompt_start_idx)

        # 2. Grid 信息解析
        if 'video_grid_thw' in engine_inputs:
            grid_thw = engine_inputs.video_grid_thw[0].cpu().numpy()
            if len(grid_thw.shape) == 2: grid_thw = grid_thw[0]

            t_grid = int(grid_thw[0])
            h_grid = int(grid_thw[1])
            w_grid = int(grid_thw[2])

            raw_tokens_per_frame = h_grid * w_grid
            raw_total_tokens = t_grid * raw_tokens_per_frame

            print(f"[Viz Debug] Raw Grid: T={t_grid}, H={h_grid}, W={w_grid} (Total: {raw_total_tokens})")

            # =================================================================
            # [核心修复] 检测 2x2 Pooling 并自动修正 H/W
            # =================================================================
            # 如果 (Grid总数) 约等于 (实际Token数 * 4)，说明发生了 2x2 Pooling
            ratio = raw_total_tokens / (total_vis_tokens + 1e-5)
            print(f"[Viz Debug] Compression Ratio: {ratio:.2f}x")

            if 3.5 < ratio < 4.5:
                print(f"[Viz Auto] 🟢 Detected 2x2 Pooling (Factor ~4x). Halving Grid H/W.")
                h_grid = h_grid // 2
                w_grid = w_grid // 2
            # =================================================================

            # 更新修正后的 Token 数
            tokens_per_frame = h_grid * w_grid
            print(f"[Viz Debug] Corrected Grid: H={h_grid}, W={w_grid}")
            print(f"[Viz Debug] Tokens per Frame: {tokens_per_frame}")

        else:
            print("[Viz ERR] Missing video_grid_thw.")
            return

        # 3. 视频与采样
        cap = cv2.VideoCapture(video_path)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = self.cfg.get('frame_stride', 1)
        model_t_indices = np.arange(0, t_grid, stride)
        raw_frame_indices = np.linspace(0, total_frames_video - 1, t_grid, dtype=int)[model_t_indices]

        # 4. Layer 循环
        target_layers = self.cfg.get('layers', [10, 20])
        attentions = model_outputs.attentions

        for layer_idx in target_layers:
            if layer_idx >= len(attentions): continue

            layer_dir = os.path.join(task_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            attn_full = attentions[layer_idx][0].float().mean(dim=0).cpu()

            # 简单检查
            if attn_full.shape[0] < v_end_idx:
                print(f"[Viz Warn] Attention tensor smaller than expected indices!")

            summary_images = []

            for i, (t_idx, raw_idx) in enumerate(zip(model_t_indices, raw_frame_indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ret, frame_bgr = cap.read()
                if not ret: break

                # === 计算切片 (使用修正后的 tokens_per_frame) ===
                curr_start = v_start_idx + (t_idx * tokens_per_frame)
                curr_end = curr_start + tokens_per_frame

                # 打印切片信息用于验证
                # print(f"   > Frame {i}: Requesting [{curr_start}:{curr_end}]")

                if curr_end > attn_full.shape[1]:
                    print(f"     [Viz ERR] Frame {i} OOB: {curr_end} > {attn_full.shape[1]}")
                    break

                attn_slice = attn_full[prompt_start_idx:, curr_start:curr_end]

                if attn_slice.shape[1] == 0: continue

                spatial_attn_flat = attn_slice.mean(dim=0)

                # 绘图
                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                axes[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"T={t_idx}")

                # 传入修正后的 h_grid, w_grid
                hm = self._generate_heatmap(frame_bgr, spatial_attn_flat, (h_grid, w_grid), i)
                axes[1].imshow(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB))
                axes[1].set_title(f"Attn ({spatial_attn_flat.shape[0]} tokens)")

                save_path = os.path.join(layer_dir, f"frame_{i:03d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)
                summary_images.append(save_path)

            self._create_summary_grid(summary_images, os.path.join(layer_dir, f"SUMMARY.jpg"))

        cap.release()
        print(f"[Viz Debug] ========== END TASK ==========\n")