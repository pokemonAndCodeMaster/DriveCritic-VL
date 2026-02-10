from .single_view import SingleViewComposer
from .multi_view import NuScenesGridComposer, NuScenesBEVComposer

def get_composer(composer_type, output_dir, fps=2):
    mapping = {
        "single": SingleViewComposer,
        "nusc_grid": NuScenesGridComposer,
        "nusc_bev": NuScenesBEVComposer
    }
    cls = mapping.get(composer_type, SingleViewComposer)
    return cls(output_dir, fps)