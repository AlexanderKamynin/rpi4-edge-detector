from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

from edge_detector.backbones.baselines import create_baseline_backbone
from edge_detector.backbones.custom import (
    C2fBackbone,
    FasterNetBackbone,
    RepVGGBackbone,
    ShuffleNetBackbone,
)
from edge_detector.models.detector import Detector


@dataclass(frozen=True)
class DetectionMetrics:
    ap50_95: float
    ap50: float
    ar100: float

    @classmethod
    def from_coco_stats(cls, stats):
        return cls(
            ap50_95=float(stats[0]),
            ap50=float(stats[1]),
            ar100=float(stats[8]),
        )


CUSTOM_BACKBONES = {
    "custom_shufflenet": ShuffleNetBackbone,
    "custom_fasternet": FasterNetBackbone,
    "custom_repvgg": RepVGGBackbone,
    "custom_c2f": C2fBackbone,
}

BASELINE_BACKBONES = {
    "baseline_shufflenetv2_x0_5": {
        "model_name": "shufflenet_v2_x0_5",
        "out_channels": (48, 96, 192),
    },
    "baseline_fasternet_t0": {
        "model_name": "fasternet_t0",
        "out_channels": (80, 160, 320),
    },
    "baseline_yolov8n": {
        "model_name": "yolov8n",
        "out_channels": (64, 128, 256),
    },
    "baseline_yolov6n": {
        "model_name": "yolov6n_efficientrep",
        "out_channels": (64, 128, 256),
    },
}


def create_detector(model_name: str, num_classes: int) -> Detector:
    if model_name in CUSTOM_BACKBONES:
        backbone = CUSTOM_BACKBONES[model_name]()

    elif model_name in BASELINE_BACKBONES:
        config = BASELINE_BACKBONES[model_name]

        backbone = create_baseline_backbone(
            model_name=config["model_name"],
            out_channels=config["out_channels"],
            deploy=False,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Detector(
        backbone=backbone,
        num_classes=num_classes,
    )


def load_checkpoint_model(path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )

    model = create_detector(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["state_dict"])

    return model.float().to(device).eval()


def prepare_for_inference(model):
    backbone = model.model[0]
    switch_to_deploy = getattr(backbone, "switch_to_deploy", None)
    if callable(switch_to_deploy):
        switch_to_deploy()
    model.eval()
    return model


def preprocess(image: np.ndarray, imgsz: int = 640) -> torch.Tensor:
    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
    image_resized = letterbox(image=image)
    tensor = image_resized[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor)
    return torch.from_numpy(tensor).float().div_(255.0).unsqueeze(0)


@torch.inference_mode()
def predict_coco(
    checkpoint_path: str | Path,
    gt_json: str | Path,
    images_dir: str | Path,
    save_path: str | Path,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    device: str = "cpu",
    category_map: dict[int, int] | None = None,
):
    model = prepare_for_inference(load_checkpoint_model(checkpoint_path, device))

    with open(gt_json, "r", encoding="utf-8") as file:
        gt = json.load(file)

    images_dir = Path(images_dir)
    predictions = []

    for image_info in tqdm(gt["images"], desc="Predict"):
        image_path = images_dir / Path(image_info["file_name"]).name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        original_shape = image.shape[:2]
        x = preprocess(image, imgsz).to(device)
        output = model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]

        detections = non_max_suppression(
            output,
            conf_thres=conf,
            iou_thres=iou,
            max_det=max_det,
            nc=model.nc,
        )[0]
        if len(detections) == 0:
            continue

        detections[:, :4] = scale_boxes(x.shape[2:], detections[:, :4], original_shape)

        for x1, y1, x2, y2, score, class_id in detections.tolist():
            class_id = int(class_id)
            category_id = category_map[class_id] if category_map is not None else class_id
            predictions.append(
                {
                    "image_id": int(image_info["id"]),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(predictions), encoding="utf-8")
    return predictions


def _evaluate(coco_gt: COCO, coco_dt: COCO, category_ids=None, use_categories=True):
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    if category_ids is not None:
        evaluator.params.catIds = category_ids
    evaluator.params.useCats = int(use_categories)

    with redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return DetectionMetrics.from_coco_stats(evaluator.stats)


def evaluate_predictions(gt_json: str | Path, pred_json: str | Path):
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(pred_json))

    result = {
        "binary": _evaluate(coco_gt, coco_dt, use_categories=False),
        "multi": _evaluate(coco_gt, coco_dt, use_categories=True),
        "per_class": {},
    }

    for category in coco_gt.loadCats(coco_gt.getCatIds()):
        result["per_class"][category["name"]] = _evaluate(
            coco_gt,
            coco_dt,
            category_ids=[category["id"]],
            use_categories=True,
        )

    return result


def print_results(results) -> None:
    for name in ("binary", "multi"):
        metrics = results[name]
        print(name.capitalize())
        print(f"  AP50-95: {metrics.ap50_95:.4f}")
        print(f"  AP50:    {metrics.ap50:.4f}")
        print(f"  AR100:   {metrics.ar100:.4f}")

    print("Per class")
    for class_name, metrics in results["per_class"].items():
        print(
            f"  {class_name:12s} "
            f"AP50-95={metrics.ap50_95:.4f} "
            f"AP50={metrics.ap50:.4f} "
            f"AR100={metrics.ar100:.4f}"
        )
