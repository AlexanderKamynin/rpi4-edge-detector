from torch import Tensor, nn
from timm.models.fasternet import MLPBlock
from torchvision.models.shufflenetv2 import InvertedResidual
from ultralytics.nn.modules import C2f

from edge_detector.integrations.yolov6 import RepVGGBlock


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


def shufflenet_block(channels: int) -> nn.Module:
    return InvertedResidual(
        inp=channels,
        oup=channels,
        stride=1,
    )


def fasternet_block(channels: int) -> nn.Module:
    return MLPBlock(
        dim=channels,
        n_div=4,
        mlp_ratio=2,
        drop_path=0,
        layer_scale_init_value=0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        pconv_fw_type="split_cat",
    )


def c2f_block(
    channels: int,
    repeats: int = 1,
) -> nn.Module:
    return C2f(
        c1=channels,
        c2=channels,
        n=repeats,
        shortcut=True,
        e=0.5,
    )


def repvgg_block(
    channels: int,
    deploy: bool = False,
) -> nn.Module:
    return RepVGGBlock(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        deploy=deploy,
        use_se=False,
    )