import os
import torch
import numpy as np
import cv2
import matplotlib
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import List, Dict


class AttentionVisualizer:
    def __init__(self, viz_config: Dict, tokenizer):
        self.cfg = viz_config
        self.tokenizer = tokenizer
        self.font_prop = self._check_and_load_font()

        # 动态获取 Token ID
        try:
            self.VISION_START_ID = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.VISION_END_ID = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        except:
            self.VISION_START_ID = 151652
            self.VISION_END_ID = 151653

    def _check_and_load_font(self):
        font_path = self.cfg.get('font_path', 'assets/SimHei.ttf')
        if os.path.exists(font_path):
            return FontProperties(fname=font_path)
        return None

    def _find_keyword_indices(self, input_ids, prompt_start_idx):
        """
        查找关键词。增加 Debug 信息，确保关键词真的在 Prompt 里。
        """
        keywords = self.cfg.get('keywords', [])
        if not keywords: return {}

        # 转为 list
        if isinstance(input_ids, torch.Tensor):
            full_ids = input_ids.tolist()
        else:
            full_ids = list(input_ids)

        prompt_ids = full_ids[prompt_start_idx:]

        # Debug: 打印解码后的 Prompt，确认里面有关键词
        decoded_prompt = self.tokenizer.decode(prompt_ids)
        # print(f"[Viz Debug] Prompt content for matching: {decoded_prompt[:50]}...")

        kw_map = {}
        for kw in keywords:
            kw_tokens = self.tokenizer.encode(kw, add_special_tokens=False)
            if not kw_tokens: continue

            indices = []
            len_kw = len(kw_tokens)

            # 滑动窗口查找
            for i in range(len(prompt_ids) - len_kw + 1):
                # 模糊匹配：允许 token 序列完全一致
                if prompt_ids[i: i + len_kw] == kw_tokens:
                    indices.extend([x + prompt_start_idx + i for x in range(len_kw)])

            if indices:
                kw_map[kw] = list(set(indices))
                print(f"[Viz] Found keyword '{kw}' at {len(indices)} positions.")
            else:
                pass
                # print(f"[Viz] Keyword '{kw}' not found in prompt.")

        return kw_map

    def _find_best_grid_shape(self, num_tokens, img_h, img_w):
        """
        [核心修复算法] 因子分解法
        寻找 h * w = num_tokens，使得 w/h 最接近 img_w/img_h
        """
        target_ratio = img_w / img_h
        best_h, best_w = 1, num_tokens
        min_diff = float('inf')

        # 遍历因子
        for h in range(1, int(math.sqrt(num_tokens)) + 1):
            if num_tokens % h == 0:
                w = num_tokens // h

                # 检查两种组合: (h, w) 和 (w, h)
                for h_cand, w_cand in [(h, w), (w, h)]:
                    ratio = w_cand / h_cand
                    diff = abs(ratio - target_ratio)

                    if diff < min_diff:
                        min_diff = diff
                        best_h = h_cand
                        best_w = w_cand

        return best_h, best_w

    def _generate_heatmap(self, frame, attn_flat):
        try:
            num_tokens = attn_flat.shape[0]
            if num_tokens == 0: return frame

            h_img, w_img = frame.shape[:2]

            # === 使用因子分解法计算 Grid ===
            h_grid, w_grid = self._find_best_grid_shape(num_tokens, h_img, w_img)
            # ============================

            # Reshape
            attn_2d = attn_flat.reshape(h_grid, w_grid).float().cpu().numpy()

            # 归一化
            attn_norm = attn_2d - attn_2d.min()
            attn_norm = attn_norm / (attn_norm.max() + 1e-9)
            attn_uint8 = (attn_norm * 255).astype(np.uint8)

            # 高质量 Resize (CUBIC 比 LINEAR 更平滑，减少马赛克感)
            attn_resized = cv2.resize(attn_uint8, (w_img, h_img), interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)

            return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

        except Exception as e:
            print(f"[Viz Error] Heatmap calc failed: {e}")
            return frame

    def _create_summary_grid(self, image_paths, save_path):
        """生成超高分辨率汇总图"""
        if not image_paths: return
        try:
            img0 = cv2.imread(image_paths[0])
            h, w, c = img0.shape

            # 布局优化：如果是 20 帧，不要垂直排，做成 4列 x 5行
            n = len(image_paths)
            cols = 2  # 增加列数
            rows = math.ceil(n / cols)

            canvas = np.ones((rows * h, cols * w, c), dtype=np.uint8) * 255

            for i, p in enumerate(image_paths):
                img = cv2.imread(p)
                if img is None: continue

                r = i // cols
                c_idx = i % cols

                y_s, y_e = r * h, (r + 1) * h
                x_s, x_e = c_idx * w, (c_idx + 1) * w

                if img.shape != (h, w, c): img = cv2.resize(img, (w, h))
                canvas[y_s:y_e, x_s:x_e] = img

            cv2.imwrite(save_path, canvas)
        except:
            pass

    def process(self, task_id, video_path, engine_inputs, model_outputs):
        if not self.cfg.get('enabled', True): return

        # 1. 目录准备
        base_dir = self.cfg.get('output_dir', 'outputs/viz_results')
        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # 2. 定位 Token
        input_ids = engine_inputs.input_ids[0]
        try:
            v_start_indices = (input_ids == self.VISION_START_ID).nonzero(as_tuple=True)[0]
            v_end_indices = (input_ids == self.VISION_END_ID).nonzero(as_tuple=True)[0]

            if len(v_start_indices) == 0: return
            v_start_idx = v_start_indices[0].item() + 1
            v_end_idx = v_end_indices[-1].item()
            prompt_start_idx = v_end_idx + 1
        except:
            return

        # 3. 关键词映射 (确保关键词在 Prompt 中存在！)
        kw_map = self._find_keyword_indices(input_ids, prompt_start_idx)

        # 4. 获取时间步 T (只信任 T)
        if 'video_grid_thw' in engine_inputs:
            grid_thw = engine_inputs.video_grid_thw[0].cpu().numpy()
            if len(grid_thw.shape) == 2: grid_thw = grid_thw[0]
            t_grid = grid_thw[0]
        else:
            t_grid = 1

        # 5. 视频信息
        cap = cv2.VideoCapture(video_path)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_video == 0: return

        # === 需求1: 生成更多帧 ===
        # 使用 stride 方式，或者指定生成数量
        # 这里设置为每隔 1 个时间步生成一张 (dense)
        model_t_indices = np.arange(0, t_grid, 1)
        # 映射回原始视频帧
        raw_frame_indices = np.linspace(0, total_frames_video - 1, t_grid, dtype=int)[model_t_indices]

        # 6. 逐层可视化
        target_layers = self.cfg.get('layers', [10, 20])
        attentions = model_outputs.attentions

        for layer_idx in target_layers:
            if layer_idx >= len(attentions): continue
            layer_dir = os.path.join(task_dir, f"layer_{layer_idx:02d}")
            os.makedirs(layer_dir, exist_ok=True)

            # [Seq, Seq]
            layer_attn_full = attentions[layer_idx][0].float().mean(dim=0).cpu()
            summary_images = []

            for i, (t_idx, raw_idx) in enumerate(zip(model_t_indices, raw_frame_indices)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
                ret, frame_bgr = cap.read()
                if not ret: break

                # 计算切片
                total_vis_tokens = v_end_idx - v_start_idx
                tokens_per_frame = total_vis_tokens // t_grid
                curr_start = v_start_idx + (t_idx * tokens_per_frame)
                curr_end = curr_start + tokens_per_frame

                # 切片 Attention
                attn_slice = layer_attn_full[prompt_start_idx:, curr_start:curr_end]
                if attn_slice.shape[1] == 0: continue

                # A. Global
                spatial_attn_flat = attn_slice.mean(dim=0)

                # B. Keywords
                kw_heatmaps = {}
                for kw, indices in kw_map.items():
                    if indices:
                        kw_slice = layer_attn_full[indices, curr_start:curr_end]
                        # 对该关键词的所有 token 取平均
                        kw_heatmaps[kw] = kw_slice.mean(dim=0)

                # === 绘图 (需求2: 高分辨率) ===
                num_plots = 2 + len(kw_map)
                # 增加 figsize, (宽, 高)
                # 假设原图是 1920x720，我们希望输出大一点
                fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
                if num_plots == 1: axes = [axes]

                # 1. 原图
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                axes[0].imshow(frame_rgb)
                # 字体调大
                axes[0].set_title(f"Layer {layer_idx} | T_model={t_idx} | Frame={raw_idx}",
                                  fontproperties=self.font_prop, fontsize=16)
                axes[0].axis('off')

                # 2. Global
                hm_overall = self._generate_heatmap(frame_bgr, spatial_attn_flat)
                axes[1].imshow(cv2.cvtColor(hm_overall, cv2.COLOR_BGR2RGB))
                axes[1].set_title("Global Attention", fontproperties=self.font_prop, fontsize=16)
                axes[1].axis('off')

                # 3. Keywords
                for k_i, (kw, kw_attn_flat) in enumerate(kw_heatmaps.items()):
                    ax = axes[2 + k_i]
                    hm_kw = self._generate_heatmap(frame_bgr, kw_attn_flat)
                    ax.imshow(cv2.cvtColor(hm_kw, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"Focus: {kw}", fontproperties=self.font_prop, fontsize=16)
                    ax.axis('off')

                save_path = os.path.join(layer_dir, f"frame_{i:03d}.jpg")
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)  # 提高 DPI
                plt.close(fig)
                summary_images.append(save_path)

            summary_path = os.path.join(layer_dir, f"SUMMARY_Layer_{layer_idx}.jpg")
            self._create_summary_grid(summary_images, summary_path)

        cap.release()