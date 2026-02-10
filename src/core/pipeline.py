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
        # # 推理完后，生成的中间状态其实没用了，强制回收
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()
        # gc.collect()

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
        逆向工程：适配 (Tp, C, Ph, Pw) 布局
        """
        try:
            t_grid, h_grid, w_grid = grid_thw
            n_patches, dim = pixel_values.shape

            # 1. 动态推导规格
            patch_size = 14
            temporal_patch = 1
            if dim == 1536:  # 16*16*3*2
                patch_size = 16
                temporal_patch = 2
            elif dim == 1176:
                patch_size = 14
                temporal_patch = 2

            console.print(f"[Debug] Config: P={patch_size}, T_p={temporal_patch}, Dim={dim}")

            # 2. Reshape 恢复 Grid
            # [N, D] -> [Tg, Hg, Wg, D]
            tensor_vals = pixel_values.float().cpu()
            video_tensor = tensor_vals.reshape(t_grid, h_grid, w_grid, dim)

            # 3. 拆解 Patch [核心修复]
            # 之前的错误尝试: (Tp, Ph, Pw, C) -> 导致灰褐色噪点
            # 正确的内存布局: (Tp, C, Ph, Pw)
            video_tensor = video_tensor.reshape(
                t_grid, h_grid, w_grid,
                temporal_patch, 3, patch_size, patch_size
            )

            # 此时维度索引:
            # 0: Tg
            # 1: Hg
            # 2: Wg
            # 3: Tp
            # 4: C
            # 5: Ph
            # 6: Pw

            # 4. Permute (归位)
            # 目标: (Tg, Tp, Hg, Ph, Wg, Pw, C)
            # 也就是: (Total_T, Total_H, Total_W, C)
            # 索引映射: 0, 3, 1, 5, 2, 6, 4
            video_tensor = video_tensor.permute(0, 3, 1, 5, 2, 6, 4)

            # 5. Merge
            video_tensor = video_tensor.reshape(
                t_grid * temporal_patch,
                h_grid * patch_size,
                w_grid * patch_size,
                3
            )

            # ... (后续转 numpy, 反归一化, 保存代码不变) ...
            video_tensor = video_tensor.numpy()
            video_tensor = video_tensor * self.std + self.mean
            video_tensor = np.clip(video_tensor, 0, 1)
            video_tensor = (video_tensor * 255).astype(np.uint8)

            # Save
            h_real, w_real = video_tensor.shape[1:3]
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (w_real, h_real))
            for i in range(len(video_tensor)):
                frame = cv2.cvtColor(video_tensor[i], cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()

        except Exception as e:
            console.print(f"[red]Reconstruct Error: {e}[/red]")

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