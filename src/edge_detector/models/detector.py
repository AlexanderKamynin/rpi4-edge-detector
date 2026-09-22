import torch
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils.loss import v8DetectionLoss


class Detector(BaseModel):

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()

        Detect.legacy = False
        detect = Detect(nc=num_classes, ch=backbone.out_channels)
        detect.stride = torch.tensor([8.0, 16.0, 32.0])
        detect.bias_init()

        self.model = nn.ModuleList([backbone, detect])
        self.stride = detect.stride
        self.nc = num_classes
        self.names = {i: str(i) for i in range(num_classes)}
        self.task = "detect"
        self.yaml = {"nc": num_classes}
        self.args = get_cfg()

    def predict(self, x, *args, **kwargs):
        c3, c4, c5 = self.model[0](x)
        return self.model[-1]([c3, c4, c5])

    def init_criterion(self):
        return v8DetectionLoss(self)
