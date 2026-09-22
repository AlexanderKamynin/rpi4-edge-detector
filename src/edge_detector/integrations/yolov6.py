import sys
from pathlib import Path

YOLOV6_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "YOLOv6"

if not YOLOV6_ROOT.exists():
    raise ImportError("YOLOv6 submodule is missing. " "Run: git submodule update --init --recursive")

if str(YOLOV6_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV6_ROOT))


from yolov6.layers.common import RepVGGBlock
from yolov6.models.efficientrep import EfficientRep
from yolov6.utils.torch_utils import fuse_model

__all__ = [
    "RepVGGBlock",
    "EfficientRep",
    "fuse_model",
]
