from torch import nn

from edge_detector.integrations.yolov6 import RepVGGBlock, fuse_model
from edge_detector.models.blocks import (
    Conv,
    c2f_block,
    fasternet_block,
    repvgg_block,
    shufflenet_block,
)


class ShuffleNetBackbone(nn.Module):
    out_channels = (64, 128, 256)

    def __init__(self):
        super().__init__()

        self.stem = Conv(3, 16, kernel_size=3, stride=2)

        self.stage2 = nn.Sequential(
            Conv(16, 32, kernel_size=3, stride=2),
            shufflenet_block(32),
        )
        self.stage3 = nn.Sequential(
            Conv(32, 64, kernel_size=3, stride=2),
            shufflenet_block(64),
            shufflenet_block(64),
        )
        self.stage4 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=2),
            shufflenet_block(128),
            shufflenet_block(128),
        )
        self.stage5 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            shufflenet_block(256),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5


class FasterNetBackbone(nn.Module):
    out_channels = (64, 128, 256)

    def __init__(self):
        super().__init__()

        self.stem = Conv(3, 16, kernel_size=3, stride=2)

        self.stage2 = nn.Sequential(
            Conv(16, 32, kernel_size=3, stride=2),
            fasternet_block(32),
        )
        self.stage3 = nn.Sequential(
            Conv(32, 64, kernel_size=3, stride=2),
            fasternet_block(64),
            fasternet_block(64),
        )
        self.stage4 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=2),
            fasternet_block(128),
            fasternet_block(128),
        )
        self.stage5 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            fasternet_block(256),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5


class RepVGGBackbone(nn.Module):
    out_channels = (64, 128, 256)

    def __init__(self):
        super().__init__()

        self.stem = Conv(3, 16, kernel_size=3, stride=2)

        self.stage2 = nn.Sequential(
            Conv(16, 32, kernel_size=3, stride=2),
            repvgg_block(32),
        )
        self.stage3 = nn.Sequential(
            Conv(32, 64, kernel_size=3, stride=2),
            repvgg_block(64),
            repvgg_block(64),
        )
        self.stage4 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=2),
            repvgg_block(128),
            repvgg_block(128),
        )
        self.stage5 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            repvgg_block(256),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5

    def switch_to_deploy(self):
        fuse_model(self)

        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

        self.eval()
        return self


class C2fBackbone(nn.Module):
    out_channels = (64, 128, 256)

    def __init__(self):
        super().__init__()

        self.stem = Conv(3, 16, kernel_size=3, stride=2)

        self.stage2 = nn.Sequential(
            Conv(16, 32, kernel_size=3, stride=2),
            c2f_block(32),
        )
        self.stage3 = nn.Sequential(
            Conv(32, 64, kernel_size=3, stride=2),
            c2f_block(64, repeats=2),
        )
        self.stage4 = nn.Sequential(
            Conv(64, 128, kernel_size=3, stride=2),
            c2f_block(128, repeats=2),
        )
        self.stage5 = nn.Sequential(
            Conv(128, 256, kernel_size=3, stride=2),
            c2f_block(256),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)

        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5


CUSTOM_BACKBONES = {
    "shufflenet": ShuffleNetBackbone,
    "fasternet": FasterNetBackbone,
    "repvgg": RepVGGBackbone,
    "c2f": C2fBackbone,
}
