import os
import json
import glob
import pandas as pd
from tqdm import tqdm
from .base import BaseDataset, DataSample, VideoIndexManager


# ==========================================
# 1. NuScenes Dataset
# ==========================================
class NuScenesDataset(BaseDataset):
    def _load_data(self):
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError("Please install nuscenes-devkit!")

        version = self.cfg.get('version', 'v1.0-mini')
        # NuScenes 初始化比较慢，加个打印
        print(f"[Data] Initializing NuScenes SDK ({version})...")
        nusc = NuScenes(version=version, dataroot=self.data_root, verbose=False)

        target_cam = self.cfg.get('camera', 'CAM_FRONT')  # 默认取前视，或者 'ALL'
        scene_limit = self.cfg.get('scene_limit', -1)  # 调试用，限制加载场景数

        scenes = nusc.scene
        if scene_limit > 0:
            scenes = scenes[:scene_limit]

        print(f"[Data] Parsing {len(scenes)} NuScenes scenes...")

        for scene in tqdm(scenes):
            scene_token = scene['token']
            scene_name = scene['name']

            # 逻辑：把一个 Scene 当作一个长视频
            # 或者：把 Scene 切成多个 Clip

            # 这里演示：提取整个 Scene 的所有帧
            first_sample_token = scene['first_sample_token']
            sample_token = first_sample_token

            frame_paths = []  # 存储每一帧的图片路径(或路径列表)

            while sample_token:
                sample = nusc.get('sample', sample_token)

                # 获取当前帧的图片路径
                if target_cam == 'ALL':
                    # 6视图顺时针：Front, FrontRight, BackRight, Back, BackLeft, FrontLeft
                    cams = [
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ]
                    current_frame_imgs = []
                    for c in cams:
                        sd_token = sample['data'][c]
                        sd = nusc.get('sample_data', sd_token)
                        current_frame_imgs.append(os.path.join(self.data_root, sd['filename']))
                    frame_paths.append(current_frame_imgs)
                else:
                    # 单视图
                    if target_cam in sample['data']:
                        sd_token = sample['data'][target_cam]
                        sd = nusc.get('sample_data', sd_token)
                        frame_paths.append(os.path.join(self.data_root, sd['filename']))

                sample_token = sample['next']

            # 检查是否有预生成的视频
            video_path = self.indexer.get_video_path(scene_name)

            self.samples.append(DataSample(
                id=scene_name,
                image_paths=frame_paths,
                video_path=video_path,
                description=scene['description'],
                meta_info={
                    "token": scene_token,
                    "count": len(frame_paths),
                    "camera": target_cam
                }
            ))


# ==========================================
# 2. DriveLM Dataset (适配你的目录结构)
# ==========================================
class DriveLMDataset(BaseDataset):
    def _load_data(self):
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError("DriveLM depends on nuScenes devkit.")

        # 配置参数读取
        anno_file = self.cfg.get('annotation_file', 'v1_1_train_nus.json')
        # 关键：DriveLM 的 data_root 是 DriveLM 文件夹，但它依赖 nuscenes 文件夹
        # 根据你的目录：data/OpenDriveLab/DriveLM 下面有 nuscenes 文件夹
        nusc_root = os.path.join(self.data_root, 'nuscenes')

        # 加载 JSON
        anno_path = os.path.join(self.data_root, anno_file)
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation not found: {anno_path}")

        with open(anno_path, 'r') as f:
            drivelm_data = json.load(f)

        # 初始化 NuScenes 用来查图
        # 注意：这里 dataroot 要指向 nuscenes 子目录
        print(f"[Data] Init NuScenes for DriveLM lookup...")
        nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_root, verbose=False)

        # 遍历数据
        # 你的 JSON 可能是 {scene_id: {key_frame_token: ...}} 结构
        for scene_id, frames in tqdm(drivelm_data.items(), desc="Loading DriveLM"):
            for frame_token, frame_info in frames.items():
                if frame_token == "key_frame_token": continue  # 过滤掉非数据的key

                # 获取图片：DriveLM 关注关键帧
                # 我们通过 nusc SDK 获取该 token 对应的 6 张图
                try:
                    sample = nusc.get('sample', frame_token)
                except KeyError:
                    # 可能 mini 集没有这个 token
                    continue

                # 提取 6 视图路径
                cams = [
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ]
                img_paths = []
                for c in cams:
                    sd_token = sample['data'][c]
                    sd = nusc.get('sample_data', sd_token)
                    # 注意：nusc SDK 返回的 filename 是相对于 nusc_root 的
                    img_paths.append(os.path.join(nusc_root, sd['filename']))

                # 提取 QA
                qa_raw = frame_info.get('QA', frame_info.get('qa', []))
                qa_pairs = [{'q': item['Q'], 'a': item['A']} for item in qa_raw]

                # 检查视频缓存
                # 这里假设 sample id 是 frame_token (或者你可以用 scene_id + frame_token)
                sample_id = frame_token
                video_path = self.indexer.get_video_path(sample_id)

                self.samples.append(DataSample(
                    id=sample_id,
                    image_paths=[img_paths],  # 注意：这里是一个时间步，包含了6张图
                    video_path=video_path,
                    qa_pairs=qa_pairs,
                    description=qa_pairs[0]['a'] if qa_pairs else "DriveLM Scene",
                    meta_info={
                        "scene_id": scene_id,
                        "frame_token": frame_token
                    }
                ))


# ==========================================
# 3. LingoQA Dataset (适配你的目录结构)
# ==========================================
class LingoQADataset(BaseDataset):
    def _load_data(self):
        # subset: 'action' or 'scenery'
        subset = self.cfg.get('subset', 'action')
        # 根据你的目录结构: data/LingoQA/action
        subset_root = os.path.join(self.data_root, subset)

        # 覆盖 indexer 的路径，因为 LingoQA 分 action 和 scenery
        # index 放在 action/video_index.json
        self.indexer = VideoIndexManager(os.path.join(subset_root, 'video_index.json'))

        # 读取 parquet
        parquet_path = os.path.join(subset_root, 'train.parquet')
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        img_root = os.path.join(subset_root, 'images', 'train')

        print(f"[Data] Loading LingoQA ({subset})...")

        # 按 segment_id 分组 (一个视频对应多个问题)
        if 'segment_id' in df.columns:
            grouped = df.groupby('segment_id')
        else:
            # 兼容不同版本，有些列名可能是 id
            grouped = df.groupby('id')

        for vid_id, group in tqdm(grouped):
            # 查找图片目录
            vid_img_dir = os.path.join(img_root, str(vid_id))
            if not os.path.exists(vid_img_dir):
                continue  # 图片未下载或解压

            # 获取所有 jpg，按文件名排序
            images = sorted(glob.glob(os.path.join(vid_img_dir, "*.jpg")))
            if not images: continue

            # 收集该视频下的所有 QA
            qa_pairs = []
            for _, row in group.iterrows():
                qa_pairs.append({
                    'q': row['question'],
                    'a': row['answer']
                })

            # 检查视频缓存
            video_path = self.indexer.get_video_path(str(vid_id))

            self.samples.append(DataSample(
                id=str(vid_id),
                image_paths=images,  # 单视图连续帧
                video_path=video_path,
                qa_pairs=qa_pairs,
                description=f"Driving scene {vid_id}",
                meta_info={"subset": subset}
            ))


# ==========================================
# 4. Folder Dataset (保留备用)
# ==========================================
class FolderDataset(BaseDataset):
    def _load_data(self):
        exts = ['*.mp4', '*.avi', '*.mov']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.data_root, "**", ext), recursive=True))

        for f in sorted(files):
            vid_id = os.path.splitext(os.path.basename(f))[0]
            # 文件夹模式默认是“已生成的视频”
            # 所以 image_paths 留空，video_path 填值
            self.samples.append(DataSample(
                id=vid_id,
                image_paths=[],
                video_path=f,
                description="Raw video file"
            ))