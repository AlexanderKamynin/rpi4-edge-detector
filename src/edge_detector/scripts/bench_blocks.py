from torch import nn
from timm.models._efficientnet_blocks import UniversalInvertedResidual
from timm.models.fasternet import MLPBlock, PatchMerging
from timm.models.ghostnet import GhostBottleneck
from ultralytics.nn.modules import C2f, Conv as UltralyticsConv
from torchvision.models.shufflenetv2 import InvertedResidual

from edge_detector.integrations.yolov6 import RepVGGBlock
from edge_detector.models.blocks import (
    c2f_block,
    fasternet_block,
    repvgg_block,
    shufflenet_block,
)

BLOCK_NAMES = {
    "shufflenet",
    "fasternet",
    "c2f",
    "ghost",
    "uib",
    "repvgg",
}


GHOST_BLOCK_CONFIGS = {
    64: {
        "expand_ratio": 3,
        "kernel_size": 5,
    },
    128: {
        "expand_ratio": 6,
        "kernel_size": 3,
    },
    256: {
        "expand_ratio": 6,
        "kernel_size": 5,
    },
}
GHOST_TRANSITION_CONFIGS = {
    64: {
        "kernel_size": 3,
        "se_ratio": 0.0,
    },
    128: {
        "kernel_size": 5,
        "se_ratio": 0.25,
    },
}
UIB_TRANSITION_CONFIGS = {
    64: {
        "kernel_size": 5,
        "expand_ratio": 3.0,
    },
    128: {
        "kernel_size": 3,
        "expand_ratio": 6.0,
    },
}


def build_benchmark_block(
    block_name: str,
    channels: int,
) -> nn.Module:
    if block_name not in BLOCK_NAMES:
        raise ValueError(f"Unsupported block: {block_name}")

    if block_name == "shufflenet":
        return shufflenet_block(channels)

    if block_name == "fasternet":
        return fasternet_block(channels)

    if block_name == "c2f":
        return c2f_block(channels)

    if block_name == "repvgg":
        return repvgg_block(
            channels=channels,
            deploy=True,
        )

    if block_name == "ghost":
        if channels not in GHOST_BLOCK_CONFIGS:
            raise ValueError(f"Unsupported Ghost block channels: {channels}")

        return build_ghost_block(channels)

    return build_uib_block(channels)


def build_benchmark_transition(
    block_name: str,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    if block_name not in BLOCK_NAMES:
        raise ValueError(f"Unsupported transition: {block_name}")

    if block_name == "shufflenet":
        return InvertedResidual(
            inp=in_channels,
            oup=out_channels,
            stride=2,
        )

    if block_name == "fasternet":
        return nn.Sequential(
            PatchMerging(
                dim=in_channels,
                patch_size=2,
                norm_layer=nn.BatchNorm2d,
            ),
            MLPBlock(
                dim=out_channels,
                n_div=4,
                mlp_ratio=2,
                drop_path=0,
                layer_scale_init_value=0,
                act_layer=nn.GELU,
                norm_layer=nn.BatchNorm2d,
                pconv_fw_type="split_cat",
            ),
        )

    if block_name == "c2f":
        return nn.Sequential(
            UltralyticsConv(
                c1=in_channels,
                c2=out_channels,
                k=3,
                s=2,
            ),
            C2f(
                c1=out_channels,
                c2=out_channels,
                n=1,
                shortcut=True,
                e=0.5,
            ),
        )

    if block_name == "repvgg":
        return nn.Sequential(
            RepVGGBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                deploy=True,
                use_se=False,
            ),
            RepVGGBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                deploy=True,
                use_se=False,
            ),
        )

    if block_name == "ghost":
        if in_channels not in GHOST_TRANSITION_CONFIGS:
            raise ValueError(f"Unsupported Ghost transition: " f"{in_channels} -> {out_channels}")

        return build_ghost_transition(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    if in_channels not in UIB_TRANSITION_CONFIGS:
        raise ValueError(f"Unsupported UIB transition: " f"{in_channels} -> {out_channels}")

    return build_uib_transition(
        in_channels=in_channels,
        out_channels=out_channels,
    )


def build_ghost_block(channels: int) -> nn.Module:
    config = GHOST_BLOCK_CONFIGS[channels]

    return GhostBottleneck(
        in_chs=channels,
        mid_chs=channels * config["expand_ratio"],
        out_chs=channels,
        dw_kernel_size=config["kernel_size"],
        stride=1,
        act_layer=nn.ReLU,
        se_ratio=0.25,
    )


def build_uib_block(channels: int) -> nn.Module:
    return UniversalInvertedResidual(
        in_chs=channels,
        out_chs=channels,
        dw_kernel_size_start=0,
        dw_kernel_size_mid=3,
        dw_kernel_size_end=0,
        stride=1,
        exp_ratio=2.0,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        se_layer=None,
        layer_scale_init_value=None,
    )


def build_ghost_transition(
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    config = GHOST_TRANSITION_CONFIGS[in_channels]

    return GhostBottleneck(
        in_chs=in_channels,
        mid_chs=in_channels * 6,
        out_chs=out_channels,
        dw_kernel_size=config["kernel_size"],
        stride=2,
        act_layer=nn.ReLU,
        se_ratio=config["se_ratio"],
    )


def build_uib_transition(
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    config = UIB_TRANSITION_CONFIGS[in_channels]

    return UniversalInvertedResidual(
        in_chs=in_channels,
        out_chs=out_channels,
        dw_kernel_size_start=config["kernel_size"],
        dw_kernel_size_mid=config["kernel_size"],
        dw_kernel_size_end=0,
        stride=2,
        exp_ratio=config["expand_ratio"],
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        se_layer=None,
        layer_scale_init_value=None,
    )
