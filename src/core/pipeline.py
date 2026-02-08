import os
import torch
import numpy as np
import cv2
import math
import gc
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data.base import DataSample

console = Console()


class InferencePipeline:
    def __init__(self, model_wrapper, composer, visualizer=None):
        self.model = model_wrapper
        self.composer = composer
        self.viz = visualizer

        # 定义 Qwen 的 Patch 参数 (通常固定)
        self.patch_size = 14
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)

    def run_sample(self, sample: DataSample):
        """处理单个样本的全流程"""
        result = {
            "id": sample.id,
            "prompt": sample.get_prompt_text(),
            "video_path": None,
            "prediction": None,
            "error": None
        }

        # try:
        # 1. 准备视频
        video_path = self._prepare_video_resource(sample)
        if not video_path:
            result['error'] = "Video generation failed"
            return result

        result['video_path'] = video_path

        # 2. 准备模型输入
        inputs = self.model.prepare_inputs(video_path, result['prompt'])

        # === [DEBUG] 深度检查输入 ===
        self._inspect_inputs(sample.id, inputs, video_path)

        # 3. 推理生成
        output_text = self.model.generate(inputs)
        result['prediction'] = output_text
        # 推理完后，生成的中间状态其实没用了，强制回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        # 4. 可视化 & 输出检查
        if self.viz:
            self._run_visualization(sample.id, video_path, inputs)

        # except Exception as e:
        #     console.print(f"[bold red][Pipeline Error][/bold red] Sample {sample.id}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     result['error'] = str(e)

        return result

    def _prepare_video_resource(self, sample):
        """处理 DataSample 的多态输入"""
        video_input = sample.get_model_input()
        if isinstance(video_input, list):
            return self.composer.process(sample)
        if isinstance(video_input, str) and os.path.exists(video_input):
            return video_input
        return None

    def _inspect_inputs(self, task_id, inputs, origin_path):
        """
        [关键功能] 深度解析模型输入 Tensor
        1. 打印 Grid 形状 (T, H, W)
        2. 将 Tensor 还原为视频并保存，查看模型实际看到的画质
        """
        # 准备 Debug 目录
        debug_dir = os.path.join("outputs", "debug_inspect", task_id)
        os.makedirs(debug_dir, exist_ok=True)

        table = Table(title=f"🔍 Model Input Inspection: {task_id}", show_lines=True)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value / Shape", style="green")

        # 1. Input IDs
        input_ids = inputs['input_ids']
        table.add_row("Input IDs Shape", str(tuple(input_ids.shape)))
        table.add_row("Total Context Length", str(input_ids.shape[1]))

        # 2. Pixel Values (Vision)
        if 'pixel_values_videos' in inputs:
            pixel_vals = inputs['pixel_values_videos']  # [N_patches, C*P*P]
            grid_thw = inputs['video_grid_thw']  # [1, 3] -> (T, H, W)

            # 兼容 batch 维度
            if len(grid_thw.shape) == 2: grid_thw = grid_thw[0]

            t_grid, h_grid, w_grid = grid_thw.tolist()

            table.add_row("Pixel Values Shape", str(tuple(pixel_vals.shape)))
            table.add_row("Grid Geometry (T, H, W)", f"T={t_grid}, H={h_grid}, W={w_grid}")

            # 计算实际分辨率
            res_h = h_grid * self.patch_size
            res_w = w_grid * self.patch_size
            table.add_row("Effective Resolution", f"{res_w} x {res_h}")
            table.add_row("Effective Total Frames", str(t_grid))

            # === [核心] 还原视频 Tensor 为 MP4 ===
            save_path = os.path.join(debug_dir, "model_seen_input.mp4")
            self._reconstruct_video_tensor(pixel_vals, (t_grid, h_grid, w_grid), save_path)
            table.add_row("Debug Video Saved", save_path)
        else:
            table.add_row("Vision Input", "[yellow]None[/yellow]")

        console.print(table)

    def _reconstruct_video_tensor(self, pixel_values, grid_thw, save_path):
        """
        逆向工程：将 Qwen 的 Patch Tensor 还原回人类可读视频
        [修复] 适配 Patch 16 + Channel-First (C, T, H, W) 布局
        """
        try:
            t_grid, h_grid, w_grid = grid_thw
            n_patches, dim = pixel_values.shape

            # === 1. 动态推导 Patch 规格 ===
            # 公式: temporal_patch * patch_size^2 * 3 = dim
            temporal_patch = 1
            patch_size = 14

            # 针对你的 Dim=1536 (16*16*3*2)
            if dim == 1536:
                patch_size = 16
                temporal_patch = 2
            elif dim == 1176:  # Qwen2-VL standard
                patch_size = 14
                temporal_patch = 2
            elif dim == 768:
                patch_size = 16
                temporal_patch = 1

            console.print(f"[Debug] Inferred Patch Config: P={patch_size}, T_p={temporal_patch} (Dim={dim})")

            # === 2. Reshape & Permute ===
            tensor_vals = pixel_values.float().cpu()

            # 还原 Grid: [T_grid, H_grid, W_grid, D]
            video_tensor = tensor_vals.reshape(t_grid, h_grid, w_grid, dim)

            # [核心修复] 拆解 D -> (C, T_patch, P, P)
            # 之前的错误假设是 (T, C, P, P)，导致了颜色和空间错位
            video_tensor = video_tensor.reshape(
                t_grid, h_grid, w_grid,
                3, temporal_patch, patch_size, patch_size
            )

            # 现在的维度: (Tg, Hg, Wg, C, Tp, Ph, Pw)
            # 目标维度:   (Tg, Tp, Hg, Ph, Wg, Pw, C) -> (Total_T, Total_H, Total_W, C)

            # Permute indices:
            # Tg(0), Tp(4), Hg(1), Ph(5), Wg(2), Pw(6), C(3)
            video_tensor = video_tensor.permute(0, 4, 1, 5, 2, 6, 3)

            # Merge dimensions
            video_tensor = video_tensor.reshape(
                t_grid * temporal_patch,  # Total Time
                h_grid * patch_size,  # Total Height
                w_grid * patch_size,  # Total Width
                3  # Channels
            )

            # 转 numpy
            video_tensor = video_tensor.numpy()

            # === 3. 反归一化 & 保存 ===
            # (x * std + mean)
            video_tensor = video_tensor * self.std + self.mean
            video_tensor = np.clip(video_tensor, 0, 1)
            video_tensor = (video_tensor * 255).astype(np.uint8)

            h_real, w_real = video_tensor.shape[1:3]
            console.print(f"[Debug] Reconstructed Video Shape: {video_tensor.shape}")

            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (w_real, h_real))
            for i in range(len(video_tensor)):
                # RGB -> BGR for OpenCV
                frame = cv2.cvtColor(video_tensor[i], cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()

        except Exception as e:
            console.print(f"[bold red]Failed to reconstruct debug video:[/bold red] {e}")
            import traceback
            traceback.print_exc()

    def _run_visualization(self, task_id, video_path, inputs):
        """执行可视化 & 输出检查"""
        try:
            # Forward Pass
            outputs = self.model.forward_for_viz(inputs)

            # === [DEBUG] 检查输出 ===
            self._inspect_outputs(task_id, outputs)

            # Draw Heatmaps
            self.viz.process(
                task_id=task_id,
                video_path=video_path,
                engine_inputs=inputs,
                model_outputs=outputs
            )
        except Exception as e:
            console.print(f"[Viz Warn] Failed to visualize {task_id}: {e}")

    def _inspect_outputs(self, task_id, outputs):
        """解析模型输出规格"""
        table = Table(title=f"🧠 Model Output Analysis", show_header=False, box=None)

        # Logits
        if hasattr(outputs, 'logits'):
            table.add_row("[bold]Logits Shape[/bold]", str(tuple(outputs.logits.shape)))

        # Attention
        if hasattr(outputs, 'attentions'):
            # attentions is tuple of len(layers)
            num_layers = len(outputs.attentions)
            # shape: (Batch, Heads, Seq_Len, Seq_Len)
            first_layer_shape = tuple(outputs.attentions[0].shape)

            table.add_row("[bold]Attention Layers[/bold]", str(num_layers))
            table.add_row("[bold]Attn Tensor Shape[/bold]", str(first_layer_shape))

            # 计算总注意力参数量
            heads = first_layer_shape[1]
            seq_len = first_layer_shape[2]
            table.add_row("[bold]Attention Heads[/bold]", str(heads))
            table.add_row("[bold]Sequence Length[/bold]", str(seq_len))

        console.print(Panel(table, title=f"Output Stats: {task_id}", border_style="magenta"))