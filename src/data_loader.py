import os
import glob
import yaml
from typing import List, Dict


class ConfigLoader:
    @staticmethod
    def load_yaml(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


class DataHandler:
    def __init__(self, infer_config):
        self.cfg = infer_config

    def get_tasks(self) -> List[Dict]:
        tasks = []
        mode = self.cfg['mode']
        path = self.cfg['input_path']

        if mode == "single":
            tasks.append({"id": "single_task", "path": path})
        elif mode == "folder":
            exts = ['*.mp4', '*.avi', '*.mov']
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(path, ext)))
            for f in files:
                tasks.append({"id": os.path.splitext(os.path.basename(f))[0], "path": f})
        # Dataset logic ignored for brevity

        print(f"[Data] Loaded {len(tasks)} tasks from {path}")
        return tasks