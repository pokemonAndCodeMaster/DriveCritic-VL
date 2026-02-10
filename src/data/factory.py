from .datasets import FolderDataset, LingoQADataset, NuScenesDataset, DriveLMDataset

class DatasetFactory:
    _REGISTRY = {
        "folder": FolderDataset,
        "lingoqa": LingoQADataset,
        "nuscenes": NuScenesDataset,
        "drivelm": DriveLMDataset,  # 新增
    }

    @staticmethod
    def get_dataset(cfg: dict):
        d_type = cfg.get('type', 'folder').lower()
        if d_type not in DatasetFactory._REGISTRY:
            raise ValueError(f"Unknown dataset type: {d_type}")
        return DatasetFactory._REGISTRY[d_type](cfg)