from ultralytics.models.yolo.detect import DetectionTrainer

from edge_detector.models.detector import Detector


class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, backbone_cls, *args, **kwargs):
        self.backbone_cls = backbone_cls
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = Detector(
            backbone=self.backbone_cls(),
            num_classes=self.data["nc"],
        )
        model.args = self.args
        model.names = self.data["names"]
        return model
