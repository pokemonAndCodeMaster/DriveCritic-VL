import argparse
import os
import sys
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box

# 路径 hack，确保能导入 src
sys.path.append(os.getcwd())

from src.data.factory import DatasetFactory
from src.data.base import DataSample

console = Console()


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_file_exists(path):
    """物理检查文件是否存在"""
    if os.path.exists(path):
        return "[green]Found[/green]"
    return "[red]Missing[/red]"


def analyze_image_source(image_paths):
    """
    深入分析 image_paths 结构
    DataSample.image_paths 可能是:
    1. [] (FolderDataset)
    2. [p1, p2, p3] (LingoQA - 单视图时序)
    3. [[v1, v2..], [v1, v2..]] (NuScenes/DriveLM - 多视图时序)
    """
    if not image_paths:
        return "Empty (Video Only Mode?)"

    total_frames = len(image_paths)
    first_frame = image_paths[0]

    if isinstance(first_frame, str):
        # 单视图模式
        status = check_file_exists(first_frame)
        return (f"Single-View Sequence\n"
                f"├── Length: {total_frames} frames\n"
                f"├── Sample[0]: {os.path.basename(first_frame)} ({status})")

    elif isinstance(first_frame, list):
        # 多视图模式
        views_per_frame = len(first_frame)
        # 检查第一帧的第一个视角
        status = check_file_exists(first_frame[0])
        return (f"Multi-View Sequence\n"
                f"├── Time Steps: {total_frames}\n"
                f"├── Views per Step: {views_per_frame}\n"
                f"├── Sample[0][0]: {os.path.basename(first_frame[0])} ({status})")

    return "Unknown Structure"


def inspect_sample(sample: DataSample, index: int):
    """生成单个样本的详细体检报告"""
    table = Table(title=f"Sample #{index} Inspection", box=box.ROUNDED, show_lines=True)
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value / Analysis", style="white")

    # 1. 基础信息
    table.add_row("ID", sample.id)
    table.add_row("Description",
                  sample.description[:100] + "..." if len(sample.description) > 100 else sample.description)

    # 2. 媒体源分析 (Raw Images)
    img_analysis = analyze_image_source(sample.image_paths)
    table.add_row("Raw Images", img_analysis)

    # 3. 视频缓存状态 (Video Cache)
    vid_path = sample.video_path
    if vid_path:
        status = check_file_exists(vid_path)
        table.add_row("Video Cache", f"{os.path.basename(vid_path)}\nStatus: {status}")
    else:
        table.add_row("Video Cache", "[yellow]Not Generated[/yellow] (Will use raw images)")

    # 4. 智能加载模拟
    # 模拟 get_model_input() 的行为
    model_input = sample.get_model_input()
    input_type = "MP4 File" if isinstance(model_input, str) else "Image List"
    table.add_row("Model Input", f"[bold magenta]{input_type}[/bold magenta] (Current Strategy)")

    # 5. QA 对分析
    if sample.qa_pairs:
        qa_tree = Tree(f"QA Pairs (Total: {len(sample.qa_pairs)})")
        # 只展示前2个和最后1个，避免刷屏
        display_indices = list(range(min(2, len(sample.qa_pairs))))
        if len(sample.qa_pairs) > 2:
            display_indices.append(len(sample.qa_pairs) - 1)

        for i in display_indices:
            q = sample.qa_pairs[i]['q']
            a = sample.qa_pairs[i]['a']
            if i == display_indices[-1] and len(sample.qa_pairs) > 3:
                qa_tree.add(f"... (skipped {len(sample.qa_pairs) - 3}) ...")
            qa_tree.add(f"[Q]: {q}\n[A]: {a}")
        table.add_row("QA Content", qa_tree)
    else:
        table.add_row("QA Content", "[yellow]None[/yellow]")

    # 6. Meta Info
    table.add_row("Meta Info", str(sample.meta_info))

    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../../configs/datasets.yaml", help="Path to dataset config")
    parser.add_argument("--dataset_key", default="nuscenes", required=True, help="Key in yaml (e.g., drivelm, nuscenes, lingoqa)")
    parser.add_argument("--num_check", type=int, default=2, help="How many samples to inspect")
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        console.print(f"[bold red]Config file not found: {args.config}[/bold red]")
        return

    full_cfg = load_yaml(args.config)
    if args.dataset_key not in full_cfg:
        console.print(f"[bold red]Key '{args.dataset_key}' not found in {args.config}[/bold red]")
        console.print(f"Available keys: {list(full_cfg.keys())}")
        return

    dataset_cfg = full_cfg[args.dataset_key]

    # 2. 初始化数据集
    console.print(Panel(f"Initializing Dataset: [bold blue]{args.dataset_key}[/bold blue]", style="bold white"))
    try:
        dataset = DatasetFactory.get_dataset(dataset_cfg)
        console.print(f"[green]Successfully loaded dataset object.[/green] Total Samples: [bold]{len(dataset)}[/bold]")
    except Exception as e:
        console.print(f"[bold red]Failed to init dataset:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    if len(dataset) == 0:
        console.print("[yellow]Dataset is empty. Check data_root paths.[/yellow]")
        return

    # 3. 抽样检查 (检查头部和尾部)
    indices = [0]
    if len(dataset) > 1:
        indices.append(len(dataset) - 1)

    # 限制检查数量
    indices = indices[:args.num_check]

    for idx in indices:
        inspect_sample(dataset[idx], idx)

    # 4. 总结建议
    console.rule("[bold]Diagnosis & Next Steps[/bold]")
    sample = dataset[0]
    if sample.video_path and os.path.exists(sample.video_path):
        console.print("✅ [green]Video cache detected.[/green] Pipeline will run fast using pre-generated MP4s.")
    else:
        console.print("⚠️ [yellow]Video cache NOT detected.[/yellow]")
        console.print("   Inference will run in [bold]On-the-fly[/bold] mode (slower).")
        console.print(
            f"   Recommended: Run [cyan]python tools/prepare_videos.py --config {args.config} --dataset_key {args.dataset_key}[/cyan] first.")


if __name__ == "__main__":
    main()