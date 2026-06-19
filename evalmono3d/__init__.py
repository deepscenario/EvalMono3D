"""EvalMono3D -- monocular 3D object detection evaluation with full SO(3) rotation.

A small, standalone toolkit that computes 2D and 3D average precision for
monocular 3D detections. Unlike KITTI / Waymo / nuScenes evaluation, the 3D IoU
accounts for the complete ``SO(3)`` rotation group, which matters for the drone
viewpoints introduced by the CDrone benchmark (GCPR 2024, oral).

Typical use::

    from evalmono3d import evaluate
    results = evaluate("my_split", "gt.json", "pred.pth", log_dir="out")
    print(results["AP2D"], results["AP3D"])
"""

from .dataset import DEFAULT_FILTER_SETTINGS
from .evaluator import Omni3DEvaluator, evaluate
from .iou3d import box3d_overlap

__version__ = "1.0.0"
__all__ = ["evaluate", "Omni3DEvaluator", "box3d_overlap", "DEFAULT_FILTER_SETTINGS", "__version__"]
