import yaml
import argparse
import os
import json
import logging
from tqdm import tqdm
from typing import List, Optional

# === Core Modules ===
from src.core.model_wrapper import QwenModelWrapper
from src.core.pipeline import InferencePipeline
from src.data.factory import DatasetFactory
from src.data.base import DataSample
from src.utils.video_composer import get_composer
from src.visualizer import AttentionVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_components(cfg: dict):
    """
    Step 1: 初始化核心组件 (模型、可视化器、合成器、流水线)
    """
    logger.info(f"Initializing components in mode: {cfg['mode']}")

    # 1. 模型包装器
    # 注入可视化开关，决定是否开启 eager attention
    cfg['model']['viz_enabled'] = cfg['visualization'].get('enabled', False)
    model_wrapper = QwenModelWrapper(cfg['model'])

    # 2. 可视化器 (可选)
    visualizer = None
    if cfg['visualization'].get('enabled', False):
        viz_cfg = cfg['visualization']
        # 修正相对路径
        save_dir = os.path.join(cfg['output_dir'], viz_cfg.get('save_dir', 'viz_heatmaps'))
        viz_cfg['output_dir'] = save_dir
        visualizer = AttentionVisualizer(viz_cfg, model_wrapper.processor.tokenizer)
        logger.info(f"Visualizer enabled. Saving to {save_dir}")

    # 3. 视频合成器 (Composer)
    # 即使是推理，也需要 Composer 来处理 "Lazy Loading" (图片转视频)
    # 我们需要从 dataset 配置中读取缓存路径
    ds_meta = cfg['dataset']
    all_ds_cfg = load_config(ds_meta['config_path'])
    target_ds_cfg = all_ds_cfg[ds_meta['name']]

    cache_dir = target_ds_cfg.get('video_cache_dir', 'processed_videos')
    full_cache_path = os.path.join(target_ds_cfg['data_root'], cache_dir)

    # 自动判断策略: DriveLM/NuScenes 用 Grid, LingoQA 用 Single
    c_type = "nusc_grid" if ds_meta['name'] in ['drivelm', 'nuscenes'] else "single"
    composer = get_composer(c_type, full_cache_path, fps=2)

    # 4. 组装流水线
    pipeline = InferencePipeline(model_wrapper, composer, visualizer)
    return pipeline, target_ds_cfg


def prepare_workload(cfg: dict, dataset_cfg: dict) -> List[DataSample]:
    """
    Step 2: 准备数据
    策略: 统一通过 DatasetFactory 加载，然后根据 Mode 进行过滤或全量返回。
    """
    mode = cfg['mode']
    logger.info(f"Loading dataset: {cfg['dataset']['name']}...")

    # 始终通过 Factory 加载数据集，确保获得 QA、MetaInfo 和 Index
    full_dataset = DatasetFactory.get_dataset(dataset_cfg)

    samples_to_run = []

    if mode == 'dataset':
        # 全量模式
        samples_to_run = list(full_dataset)
        logger.info(f"Mode [Dataset]: Loaded all {len(samples_to_run)} samples.")

    elif mode == 'single':
        # 单样本模式 (基于索引 ID)
        target_id = cfg['single_input'].get('sample_id')
        if not target_id:
            # Fallback: 如果没提供 ID，尝试读取 raw video path (外部视频)
            raw_path = cfg['single_input'].get('video_path')
            raw_prompt = cfg['single_input'].get('prompt', 'Describe the scene.')
            if raw_path and os.path.exists(raw_path):
                logger.info("Mode [Single]: Using raw external video file.")
                samples_to_run = [DataSample(
                    id=os.path.basename(raw_path),
                    image_paths=[],
                    video_path=raw_path,
                    qa_pairs=[{'q': raw_prompt, 'a': ''}],
                    description="External Video"
                )]
            else:
                raise ValueError("Mode is 'single' but no valid 'sample_id' or 'video_path' provided.")
        else:
            # 核心逻辑: 在数据集中查找该 ID
            # 这样能保留原始的 QA 和 MetaInfo
            raw_prompt = cfg['single_input'].get('prompt', 'Describe the scene.')
            found = False
            for s in full_dataset:
                if str(s.id) == str(target_id):
                    if not s.qa_pairs:
                        s.qa_pairs = [{'q': raw_prompt, 'a': ''}]
                    samples_to_run = [s]
                    found = True
                    break
            if not found:
                logger.warning(f"Sample ID {target_id} not found in dataset! Check your ID.")
            else:
                logger.info(f"Mode [Single]: Found sample {target_id} with full metadata.")

    return samples_to_run


def run_inference_loop(pipeline: InferencePipeline, samples: List[DataSample]):
    """
    Step 3: 执行核心循环
    """
    results = []
    logger.info(f"Starting inference loop for {len(samples)} samples...")

    for sample in tqdm(samples, desc="Inference"):
        # 调用 Pipeline 的标准接口
        res = pipeline.run_sample(sample)
        results.append(res)

        # 如果是单样本调试，直接打印结果
        if len(samples) == 1:
            print("\n" + "=" * 40)
            print(f"Sample: {sample.id}")
            print(f"Prompt: {res['prompt']}")
            print(f"Pred  : {res['prediction']}")
            print("=" * 40 + "\n")

    return results


def save_results(results: list, output_dir: str):
    """
    Step 4: 保存结果
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "results.json")

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved successfully to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml", help="Path to inference config")
    # 允许命令行覆盖 mode，方便调试
    parser.add_argument("--mode", choices=["dataset", "single"], help="Override mode in config")
    parser.add_argument("--id", help="Override sample_id for single mode")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = load_config(args.config)

    # 命令行参数覆盖配置
    if args.mode: cfg['mode'] = args.mode
    if args.id: cfg['single_input']['sample_id'] = args.id

    # 2. 初始化组件
    # 返回 pipeline 对象 和 解析后的数据集配置(用于获取缓存路径等)
    pipeline, ds_cfg = setup_components(cfg)

    # 3. 准备数据
    samples = prepare_workload(cfg, ds_cfg)
    if not samples:
        logger.error("No samples found to process. Exiting.")
        return

    # 4. 执行循环
    results = run_inference_loop(pipeline, samples)

    # 5. 保存结果
    save_results(results, cfg['output_dir'])


if __name__ == "__main__":
    main()