from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hfg_backbone import ConvBnReLU, HFGNet_W18, HFGNet_W18_Small, HFGNet_W48


HFGNET_VARIANTS = {
    "HFGNet_W18_Small": (HFGNet_W18_Small, 279),
    "HFGNet_W18": (HFGNet_W18, 279),
    "HFGNet_W48": (HFGNet_W48, 744),
}

LEGACY_VARIANT_ALIASES = {
    "UHRNet_W18_Small": "HFGNet_W18_Small",
    "UHRNet_W18": "HFGNet_W18",
    "UHRNet_W48": "HFGNet_W48",
}


def normalize_hfgnet_variant(backbone):
    return LEGACY_VARIANT_ALIASES.get(backbone, backbone)


def remap_legacy_hfgnet_state_dict(state_dict):
    """Map historical checkpoint keys onto the HFGNet module namespace."""
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())
    strip_module = bool(keys) and all(str(key).startswith("module.") for key in keys)

    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = str(key)
        if strip_module:
            new_key = new_key[len("module."):]
        if new_key.startswith("backbone."):
            new_key = "encoder." + new_key[len("backbone."):]
        new_key = new_key.replace(".unilk.", ".large_kernel.")
        new_key = new_key.replace(".HP.", ".adaptive_filter.")
        remapped[new_key] = value
    return remapped


class HFGNet(nn.Module):
    def __init__(self, num_classes=2, backbone="HFGNet_W18_Small"):
        super(HFGNet, self).__init__()
        variant = normalize_hfgnet_variant(backbone)
        if variant not in HFGNET_VARIANTS:
            raise ValueError(
                "Unsupported HFGNet variant '{}'. Available variants: {}".format(
                    backbone, ", ".join(HFGNET_VARIANTS.keys())
                )
            )

        encoder_fn, last_inp_channels = HFGNET_VARIANTS[variant]
        self.backbone_name = variant
        self.encoder = encoder_fn()

        self.head = nn.Sequential()
        self.head.add_module(
            "conv_1",
            ConvBnReLU(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        self.head.add_module(
            "cls",
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def load_state_dict(self, state_dict, strict=True):
        state_dict = remap_legacy_hfgnet_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, inputs):
        h, w = inputs.size(2), inputs.size(3)
        x = self.encoder(inputs)
        x = self.head(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    net = HFGNet()
    x = net(x)
