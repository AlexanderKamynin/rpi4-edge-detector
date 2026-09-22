from dataclasses import dataclass

import timm
import torchvision.models as torchvision_models
from timm.utils.model import reparameterize_model
from torch import nn
from ultralytics import YOLO

from edge_detector.integrations.yolov6 import (
    EfficientRep,
    RepVGGBlock,
    fuse_model,
)


@dataclass(frozen=True)
class BackboneConfig:
    model_name: str
    output_indices: tuple[int, int, int]
    num_blocks: int


TIMM_BLOCK_CONFIGS = {
    "mobilenetv3_large_100": BackboneConfig(
        model_name="mobilenetv3_large_100",
        output_indices=(2, 4, 5),
        num_blocks=6,
    ),
    "fbnetv3_d": BackboneConfig(
        model_name="fbnetv3_d",
        output_indices=(2, 4, 5),
        num_blocks=6,
    ),
    "mobilenetv4_conv_small_035": BackboneConfig(
        model_name="mobilenetv4_conv_small_035",
        output_indices=(1, 2, 3),
        num_blocks=4,
    ),
    "mobilenetv4_conv_small_050": BackboneConfig(
        model_name="mobilenetv4_conv_small_050",
        output_indices=(1, 2, 3),
        num_blocks=4,
    ),
    "mobilenetv4_conv_small": BackboneConfig(
        model_name="mobilenetv4_conv_small",
        output_indices=(1, 2, 3),
        num_blocks=4,
    ),
}


class TimmBlockBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()

        model = timm.create_model(
            config.model_name,
            pretrained=False,
        )

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.blocks = model.blocks[: config.num_blocks]
        self.output_indices = config.output_indices

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)

        features = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.output_indices:
                features.append(x)

        return tuple(features)


class ShuffleNetNativeBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = getattr(
            torchvision_models,
            model_name,
        )(weights=None)

        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return c3, c4, c5


class FasterNetNativeBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,
        )

        self.patch_embed = model.patch_embed
        self.stages = model.stages

    def forward(self, x):
        x = self.patch_embed(x)

        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i in (1, 2, 3):
                features.append(x)

        return tuple(features)


class ReparameterizedTimmBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,
        ).eval()

        model = reparameterize_model(model)

        self.stem = model.stem
        self.stages = model.stages

    def forward(self, x):
        x = self.stem(x)

        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i in (1, 2, 3):
                features.append(x)

        return tuple(features)


class GhostNetNativeBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,
        )

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks[:9]

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        features = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in (4, 6, 8):
                features.append(x)

        return tuple(features)


class StarNetNativeBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = timm.create_model(
            model_name,
            pretrained=False,
        )

        self.stem = model.stem
        self.stages = model.stages

    def forward(self, x):
        x = self.stem(x)

        features = []

        for i, stage in enumerate(self.stages):
            x = stage(x)

            if i in (1, 2, 3):
                features.append(x)

        return tuple(features)


class YOLOv8NativeBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        model = YOLO(f"{model_name}.yaml").model
        self.layers = model.model[:10]

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i in (4, 6, 9):
                features.append(x)

        return tuple(features)


class EfficientRepNativeBackbone(nn.Module):
    def __init__(self, deploy: bool = False):
        super().__init__()

        self.model = EfficientRep(
            in_channels=3,
            channels_list=[16, 32, 64, 128, 256],
            num_repeats=[1, 2, 4, 6, 2],
            block=RepVGGBlock,
            fuse_P2=True,
            cspsppf=True,
        )

        if deploy:
            self.switch_to_deploy()

    def forward(self, x):
        outputs = self.model(x)
        return tuple(outputs[1:])

    def switch_to_deploy(self):
        self.eval()
        self.model = fuse_model(self.model)

        for module in self.model.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

        return self


def create_baseline_backbone(
    model_name: str,
    out_channels: tuple[int, int, int] | None = None,
    deploy: bool = False,
) -> nn.Module:
    if model_name in TIMM_BLOCK_CONFIGS:
        backbone = TimmBlockBackbone(
            TIMM_BLOCK_CONFIGS[model_name],
        )
    elif model_name.startswith("shufflenet_v2_"):
        backbone = ShuffleNetNativeBackbone(model_name)
    elif model_name.startswith("fasternet_"):
        backbone = FasterNetNativeBackbone(model_name)
    elif model_name.startswith("ghostnet_"):
        backbone = GhostNetNativeBackbone(model_name)
    elif model_name in ("mobileone_s0", "repvit_m0_9"):
        backbone = ReparameterizedTimmBackbone(model_name)
    elif model_name == "starnet_s1":
        backbone = StarNetNativeBackbone(model_name)
    elif model_name in ("yolov8n", "yolov8s"):
        backbone = YOLOv8NativeBackbone(model_name)
    elif model_name == "yolov6n_efficientrep":
        backbone = EfficientRepNativeBackbone(
            deploy=deploy,
        )
    else:
        raise ValueError(f"Unsupported backbone: {model_name}")

    if out_channels is not None:
        backbone.out_channels = out_channels

    return backbone
