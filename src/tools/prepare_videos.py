import argparse
import os
import sys
import yaml
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel

# Path Hack
sys.path.append(os.getcwd())

from src.data.factory import DatasetFactory
from src.utils.video_composer import get_composer

console = Console()


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_composer_type(dataset_key, dataset_type):
    """
    根据数据集类型自动推断使用哪种视频合成策略。
    """
    # 规则 1: 包含 'nuscenes' 或 'drivelm' 的通常需要 6 视图拼接
    if 'nuscenes' in dataset_type or 'drivelm' in dataset_type:
        return "nusc_grid"

    # 规则 2: LingoQA 或其他通常是单视图序列
    if 'lingoqa' in dataset_type or 'folder' in dataset_type:
        return "single"

    # 默认回退
    return "single"


def main():
    parser = argparse.ArgumentParser(description="Asset Build Pipeline: Convert raw images to videos")
    parser.add_argument("--config", default="../../configs/datasets.yaml", required=True, help="Path to dataset config yaml")
    parser.add_argument("--dataset_key", default="nuscenes", required=True, help="Key in yaml (e.g., drivelm, lingoqa)")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS for generated videos")
    parser.add_argument("--force", action="store_true", help="Force regenerate existing videos")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    # 1. 加载配置
    full_cfg = load_yaml(args.config)
    if args.dataset_key not in full_cfg:
        console.print(f"[bold red]Error: Key '{args.dataset_key}' not found in config.[/bold red]")
        return

    ds_cfg = full_cfg[args.dataset_key]
    console.print(Panel(f"Pipeline Start: [bold blue]{args.dataset_key}[/bold blue]", style="bold white"))

    # 2. 初始化数据集
    # 注意：此时数据集加载的是"Raw Images"，因为 Index 可能还没生成或不完整
    try:
        dataset = DatasetFactory.get_dataset(ds_cfg)
        console.print(f"Loaded Dataset. Total Samples: [green]{len(dataset)}[/green]")
    except Exception as e:
        console.print(f"[bold red]Dataset Init Failed:[/bold red] {e}")
        return

    # 3. 准备 Composer (策略模式)
    # 确定输出目录
    cache_dir_name = ds_cfg.get('video_cache_dir', 'processed_videos')
    output_root = os.path.join(ds_cfg['data_root'], cache_dir_name)

    # 确定合成策略
    composer_type = infer_composer_type(args.dataset_key, ds_cfg.get('type', ''))
    console.print(f"Strategy Selected: [bold magenta]{composer_type}[/bold magenta]")
    console.print(f"Output Directory:  [dim]{output_root}[/dim]")

    composer = get_composer(composer_type, output_root, fps=args.fps)

    # 4. 准备索引管理器
    # 我们直接使用 Dataset 实例中自带的 indexer (他在 BaseDataset init 时已经加载了旧索引)
    indexer = dataset.indexer
    console.print(f"Index File: [dim]{indexer.index_path}[/dim]")

    # 5. 主循环
    success_count = 0
    skip_count = 0
    fail_count = 0

    samples_to_process = dataset
    if args.limit > 0:
        samples_to_process = list(dataset)[:args.limit]
        console.print(f"[yellow]Limiting processing to first {args.limit} samples.[/yellow]")

    # 使用 rich 的进度条或 tqdm
    for sample in tqdm(samples_to_process, desc="Processing"):
        try:
            # 检查是否跳过
            if not args.force:
                existing_video = indexer.get_video_path(sample.id)
                # 检查索引存在且文件物理存在
                if existing_video and os.path.exists(existing_video):
                    skip_count += 1
                    continue

            # 执行合成 (Strategy Pattern)
            # process 内部会自动处理 List[str] 或 List[List[str]]
            video_path = composer.process(sample)

            if video_path:
                # 更新内存索引
                indexer.update(sample.id, video_path)
                success_count += 1
            else:
                fail_count += 1
                # console.print(f"[dim]Skipped/Failed {sample.id} (No images?)[/dim]")

        except KeyboardInterrupt:
            console.print("[bold red]Process Interrupted![/bold red]")
            break
        except Exception as e:
            fail_count += 1
            console.print(f"[red]Error on {sample.id}: {e}[/red]")

    # 6. 保存索引
    # 这一步至关重要，只有保存了，后续 DataLoader 才能懒加载视频
    indexer.save()

    # 7. 总结报告
    console.rule("[bold]Pipeline Summary[/bold]")
    console.print(f"Total Processed: {len(samples_to_process)}")
    console.print(f"✅ Generated:    [green]{success_count}[/green]")
    console.print(f"⏭️  Skipped:      [yellow]{skip_count}[/yellow] (Already exists)")
    console.print(f"❌ Failed:       [red]{fail_count}[/red]")
    console.print(f"💾 Index Saved:  {indexer.index_path}")

    if success_count > 0:
        console.print("\n[bold green]Next Step:[/bold green]")
        console.print(
            f"Run [cyan]python tools/check_dataset.py --config {args.config} --dataset_key {args.dataset_key}[/cyan] to verify video cache.")


if __name__ == "__main__":
    main()