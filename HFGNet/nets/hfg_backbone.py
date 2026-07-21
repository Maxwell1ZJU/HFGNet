import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath, to_2tuple
import torch.utils.checkpoint as checkpoint
try:
    from .hfg_frequency import AdaptiveHighPassFilter
except ImportError:
    try:
        from hfg_frequency import AdaptiveHighPassFilter
    except ImportError:
        AdaptiveHighPassFilter = None
__all__ = ["HFGNet_W18_Small", "HFGNet_W18", "HFGNet_W48"]


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))
        self.adaptive_filter = AdaptiveHighPassFilter(hr_channels=nc, compressed_channels=nc // 2) if AdaptiveHighPassFilter is not None else nn.Identity()

    def forward(self, x):
        _, C, H, W = x.shape
        x = self.adaptive_filter(x)
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Frequency_Domain(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(nc=channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        A = inp
        x_step2 = self.norm(inp)  # 尺寸 [B, 2*C, H, W]
        x_freq = self.freq(x_step2)  # 尺寸 [B, C, H, W]
        x = A * x_freq
        x = A + x * self.gamma
        return x


def _valid_groups(channels, requested_groups):
    groups = min(channels, max(1, requested_groups))
    while channels % groups != 0:
        groups -= 1
    return groups


class SpatialAttentionModule(nn.Module):
    """Spatial-aware branch used inside the SHF high-frequency refinement."""

    def __init__(self, channels, ffn_ratio=2):
        super().__init__()
        hidden_channels = int(channels * ffn_ratio)
        self.dwconv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03),
            nn.GELU()
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=0.001, momentum=0.03),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        )
        self.se = SEBlock(channels, max(channels // 4, 4))
        self.gate = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        y = self.dwconv(x)
        y = self.ffn(y)
        y = self.se(y)
        return x + y * torch.sigmoid(self.gate(y))


class DynamicKaiserHighPass(nn.Module):
    """Adaptive 3x3 high-pass enhancement following the HFGNet HFM idea."""

    def __init__(self, channels, kernel_size=3, beta=8.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_area = kernel_size * kernel_size
        self.kernel_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03),
            nn.GELU(),
            nn.Conv2d(channels, self.kernel_area, 1, bias=True)
        )
        window_1d = torch.kaiser_window(kernel_size, periodic=False, beta=beta)
        window_2d = torch.outer(window_1d, window_1d)
        window_2d = window_2d / window_2d.sum().clamp_min(1e-6)
        identity = torch.zeros(self.kernel_area)
        identity[self.kernel_area // 2] = 1.0
        self.register_buffer("kaiser_window", window_2d.reshape(1, 1, self.kernel_area, 1, 1))
        self.register_buffer("identity_kernel", identity.reshape(1, 1, self.kernel_area, 1, 1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        b, c, h, w = x.shape
        pad = self.kernel_size // 2
        low_kernel = self.kernel_encoder(x).view(b, 1, self.kernel_area, h, w)
        low_kernel = F.softmax(low_kernel, dim=2) * self.kaiser_window
        low_kernel = low_kernel / low_kernel.sum(dim=2, keepdim=True).clamp_min(1e-6)
        high_kernel = self.identity_kernel - low_kernel

        patches = F.unfold(
            F.pad(x, [pad] * 4, mode='reflect'),
            kernel_size=self.kernel_size
        ).view(b, c, self.kernel_area, h, w)
        high_response = (patches * high_kernel).sum(dim=2)
        return x + self.gamma * high_response


class SpectrumRefinement(nn.Module):
    """Refines amplitude with channel MLP and phase with depthwise spatial mixing."""

    def __init__(self, channels, expand=2):
        super().__init__()
        hidden_channels = channels * expand
        self.amp_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=True)
        )
        self.phase_dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True)
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        dtype = x.dtype
        b, c, h, w = x.shape
        x_float = x.float()
        spectrum = torch.fft.rfft2(x_float, norm='ortho')
        amp = torch.abs(spectrum).clamp_min(1e-6)
        phase = torch.angle(spectrum)

        amp_delta = torch.tanh(self.amp_mlp(torch.log1p(amp)))
        phase_delta = torch.tanh(self.phase_dw(phase))
        amp = amp * (1.0 + 0.1 * amp_delta)
        phase = phase + 0.1 * phase_delta

        refined = torch.polar(amp, phase)
        refined = torch.fft.irfft2(refined, s=(h, w), norm='ortho').to(dtype)
        return x + self.gamma * refined


class HFMModule(nn.Module):
    """Spatial-aware high-frequency mask plus amplitude/phase refinement."""

    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.spatial_attention = SpatialAttentionModule(channels)
        self.high_pass = DynamicKaiserHighPass(channels)
        self.spectrum_refine = SpectrumRefinement(channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm(x)
        y = self.spatial_attention(y)
        y = self.high_pass(y)
        y = self.spectrum_refine(y)
        y = self.out_proj(y)
        return x + self.gamma * y


class SHFBlock(nn.Module):
    """Maintains the high-resolution branch and exports it as a deep prior."""

    def __init__(self, high_channels, low_channels=None, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.high_refine = HFMModule(high_channels) if enabled else nn.Identity()
        self.low_refine = HFMModule(low_channels) if enabled and low_channels is not None else nn.Identity()
        self.prior_head = nn.Sequential(
            nn.Conv2d(high_channels, high_channels, 1, bias=False),
            nn.BatchNorm2d(high_channels, eps=0.001, momentum=0.03),
            nn.GELU()
        )

    def forward(self, features):
        outs = list(features)
        outs[0] = self.high_refine(outs[0])
        if len(outs) > 1:
            outs[-1] = self.low_refine(outs[-1])
        prior = self.prior_head(outs[0])
        return outs, prior


class PriorGuidedPFConv(nn.Module):
    """Prior-guided poly-frequency convolution for Mix-Net stages."""

    def __init__(self, channels, prior_channels, group3=1, group5=2, drop_path=0.0):
        super().__init__()
        self.prior_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, channels, 1, bias=False),
                nn.BatchNorm2d(channels, eps=0.001, momentum=0.03),
                nn.GELU()
            )
            for ch in prior_channels
        ])
        context_channels = channels * 2
        hidden_channels = max(channels // 4, 8)
        self.kernel_gates = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_channels, hidden_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 1, bias=True)
        )
        self.channel_mod = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(context_channels, hidden_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, 1, bias=True)
        )
        self.conv3 = ConvBn(channels, channels, 3, padding='same', groups=_valid_groups(channels, group3))
        self.conv5 = ConvBn(channels, channels, 5, padding='same', groups=_valid_groups(channels, group5))
        self.conv7 = ConvBn(channels, channels, 7, padding='same', groups=channels)
        self.conv9 = ConvBn(channels, channels, 9, padding='same', groups=channels)
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, eps=0.001, momentum=0.03)
        )
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _fuse_prior(self, x, priors):
        if not priors:
            return torch.zeros_like(x)
        fused = 0
        count = 0
        for prior, proj in zip(priors, self.prior_projs):
            if prior is None:
                continue
            prior = proj(prior)
            if prior.shape[-2:] != x.shape[-2:]:
                prior = F.interpolate(prior, size=x.shape[-2:], mode='bilinear', align_corners=False)
            fused = fused + prior
            count += 1
        if count == 0:
            return torch.zeros_like(x)
        return fused / count

    def forward(self, x, priors=None):
        prior = self._fuse_prior(x, priors)
        context = torch.cat([x, prior], dim=1)
        kernel_gates = torch.sigmoid(self.kernel_gates(context))
        channel_mod = torch.sigmoid(self.channel_mod(context))
        in_mod, out_mod = torch.chunk(channel_mod, 2, dim=1)

        guided = x * (1.0 + in_mod) + prior
        branches = [
            self.conv3(guided),
            self.conv5(guided),
            self.conv7(x),
            self.conv9(x)
        ]
        y = 0
        for idx, branch in enumerate(branches):
            y = y + branch * kernel_gates[:, idx:idx + 1]
        y = self.out_proj(y * (1.0 + out_mod))
        return x + self.drop_path(self.gamma * y)

class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


class DilatedLargeKernelBranch(nn.Module):
    """
    Dilated reparameterized large-kernel branch derived from UniRepLKNet
    (https://github.com/AILab-CVC/UniRepLKNet).
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

class ReparamLargeKernelBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedLargeKernelBranch(dim, kernel_size, deploy=deploy,
                                                   use_sync_bn=use_sync_bn,
                                                   attempt_use_lk_impl=attempt_use_lk_impl)

        else:
            assert kernel_size in [3, 5]
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):

        def _f(x):
            return x + self.compute_residual(x)

        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top18', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot18', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low18', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=8, freq_sel_method='top18'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter




class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.ReLU(self.bn(self.conv(x)))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)

    def forward(self, x):
        return self.bn(self.conv(x))


class HFGBackbone(nn.Module):

    def __init__(self,
                 in_channels=3,
                 pretrained=None,
                 stage1_num_modules=1,
                 stage1_num_blocks=(4,),
                 stage1_num_channels=(64,),
                 stage2_num_modules=1,
                 stage2_num_blocks=(4, 4),
                 stage2_num_channels=(18, 36),
                 stage3_num_modules=5,
                 stage3_num_blocks=(4, 4),
                 stage3_num_channels=(36, 72),
                 stage4_num_modules=2,
                 stage4_num_blocks=(4, 4),
                 stage4_num_channels=(72, 144),
                 stage5_num_modules=2,
                 stage5_num_blocks=(4, 4),
                 stage5_num_channels=(144, 288),
                 stage6_num_modules=1,
                 stage6_num_blocks=(4, 4),
                 stage6_num_channels=(72, 144),
                 stage7_num_modules=1,
                 stage7_num_blocks=(4, 4),
                 stage7_num_channels=(36, 72),
                 stage8_num_modules=1,
                 stage8_num_blocks=(4, 4),
                 stage8_num_channels=(18, 36),
                 stage9_num_modules=1,
                 stage9_num_blocks=(4,),
                 stage9_num_channels=(18,),
                 has_se=False,
                 align_corners=False,
                 hfg_enable_refinement=True,
                 hfg_enable_prior=True,
                 hfg_pfconv_group3=1,
                 hfg_pfconv_group5=2,
                 hfg_drop_path=0.0):
        super(HFGBackbone, self).__init__()
        self.has_se = has_se
        self.align_corners = align_corners
        self.hfg_enable_prior = hfg_enable_prior
        hfg_prior_channels = (stage2_num_channels[0], stage3_num_channels[0], stage4_num_channels[0])
        hfg_stage_kwargs = dict(
            hfg_hf_refinement=hfg_enable_refinement,
            hfg_prior_injection=hfg_enable_prior,
            hfg_prior_channels=hfg_prior_channels,
            hfg_pfconv_group3=hfg_pfconv_group3,
            hfg_pfconv_group5=hfg_pfconv_group5,
            hfg_drop_path=hfg_drop_path
        )
        self.feat_channels = [
            sum([
                stage5_num_channels[-1], stage6_num_channels[-1],
                stage7_num_channels[-1], stage8_num_channels[-1],
                stage9_num_channels[-1]
            ]) // 2
        ]

        cur_stride = 1
        # stem net
        self.conv_layer1_1 = ConvBnReLU(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same')
        cur_stride *= 2

        self.conv_layer1_2 = ConvBnReLU(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding='same')
        cur_stride *= 2

        self.la1 = Layer1(
            num_channels=64,
            num_blocks=stage1_num_blocks[0],
            num_filters=stage1_num_channels[0],
            has_se=has_se,
            name="layer2")

        self.tr1 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage1_num_channels[0] * 4,
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage2_num_channels))
            ],
            out_channels=stage2_num_channels,
            align_corners=self.align_corners,
            name="tr1")
        self.st2 = Stage(
            num_channels=stage2_num_channels,
            num_modules=stage2_num_modules,
            num_blocks=stage2_num_blocks,
            num_filters=stage2_num_channels,
            has_se=self.has_se,
            name="st2",
            stage_index = 2,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        self.shf2 = SHFBlock(stage2_num_channels[0], stage2_num_channels[-1], enabled=hfg_enable_refinement)
        cur_stride *= 2

        self.tr2 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage2_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage3_num_channels))
            ],
            out_channels=stage3_num_channels,
            align_corners=self.align_corners,
            name="tr2")
        self.st3 = Stage(
            num_channels=stage3_num_channels,
            num_modules=stage3_num_modules,
            num_blocks=stage3_num_blocks,
            num_filters=stage3_num_channels,
            has_se=self.has_se,
            name="st3",
            stage_index=3,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        self.shf3 = SHFBlock(stage3_num_channels[0], stage3_num_channels[-1], enabled=hfg_enable_refinement)
        cur_stride *= 2

        self.tr3 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage3_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage4_num_channels))
            ],
            out_channels=stage4_num_channels,
            align_corners=self.align_corners,
            name="tr3")
        self.st4 = Stage(
            num_channels=stage4_num_channels,
            num_modules=stage4_num_modules,
            num_blocks=stage4_num_blocks,
            num_filters=stage4_num_channels,
            has_se=self.has_se,
            name="st4",
            stage_index=4,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        self.shf4 = SHFBlock(stage4_num_channels[0], stage4_num_channels[-1], enabled=hfg_enable_refinement)
        cur_stride *= 2

        self.tr4 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage4_num_channels[-1],
            stride_cur=[
                cur_stride * (2 ** i) for i in range(len(stage5_num_channels))
            ],
            out_channels=stage5_num_channels,
            align_corners=self.align_corners,
            name="tr4")
        self.st5 = Stage(
            num_channels=stage5_num_channels,
            num_modules=stage5_num_modules,
            num_blocks=stage5_num_blocks,
            num_filters=stage5_num_channels,
            has_se=self.has_se,
            name="st5",
            stage_index=5,
            align_corners=align_corners,
            **hfg_stage_kwargs)

        self.tr5 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage5_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage6_num_channels) - i - 1))
                for i in range(len(stage6_num_channels))
            ],
            out_channels=stage6_num_channels,
            align_corners=self.align_corners,
            name="tr5")
        self.st6 = Stage(
            num_channels=stage6_num_channels,
            num_modules=stage6_num_modules,
            num_blocks=stage6_num_blocks,
            num_filters=stage6_num_channels,
            has_se=self.has_se,
            name="st6",
            stage_index=6,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        cur_stride = cur_stride // 2

        self.tr6 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage6_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage7_num_channels) - i - 1))
                for i in range(len(stage7_num_channels))
            ],
            out_channels=stage7_num_channels,
            align_corners=self.align_corners,
            name="tr6")
        self.st7 = Stage(
            num_channels=stage7_num_channels,
            num_modules=stage7_num_modules,
            num_blocks=stage7_num_blocks,
            num_filters=stage7_num_channels,
            has_se=self.has_se,
            name="st7",
            stage_index=7,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        cur_stride = cur_stride // 2

        self.tr7 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage7_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage8_num_channels) - i - 1))
                for i in range(len(stage8_num_channels))
            ],
            out_channels=stage8_num_channels,
            align_corners=self.align_corners,
            name="tr7")
        self.st8 = Stage(
            num_channels=stage8_num_channels,
            num_modules=stage8_num_modules,
            num_blocks=stage8_num_blocks,
            num_filters=stage8_num_channels,
            has_se=self.has_se,
            name="st8",
            stage_index=8,
            align_corners=align_corners,
            **hfg_stage_kwargs)
        cur_stride = cur_stride // 2

        self.tr8 = TransitionLayer(
            stride_pre=cur_stride,
            in_channel=stage8_num_channels[0],
            stride_cur=[
                cur_stride // (2 ** (len(stage9_num_channels) - i - 1))
                for i in range(len(stage9_num_channels))
            ],
            out_channels=stage9_num_channels,
            align_corners=self.align_corners,
            name="tr8")
        self.st9 = Stage(
            num_channels=stage9_num_channels,
            num_modules=stage9_num_modules,
            num_blocks=stage9_num_blocks,
            num_filters=stage9_num_channels,
            has_se=self.has_se,
            name="st9",
            stage_index=9,
            align_corners=align_corners,
            **hfg_stage_kwargs)

        self.last_layer = nn.Sequential(
            ConvBnReLU(
                in_channels=self.feat_channels[0],
                out_channels=self.feat_channels[0],
                kernel_size=1,
                padding='same',
                stride=1,
                bias=True),
            nn.Conv2d(
                in_channels=self.feat_channels[0],
                out_channels=19,
                kernel_size=1,
                stride=1,
                padding=0))
        self.hfg_output_refine = nn.Sequential(
            ConvBnReLU(
                in_channels=self.feat_channels[0],
                out_channels=self.feat_channels[0],
                kernel_size=1,
                padding='same',
                stride=1,
                bias=False),
            ConvBn(
                in_channels=self.feat_channels[0],
                out_channels=self.feat_channels[0],
                kernel_size=1,
                padding='same',
                stride=1,
                bias=False)
        )
        self.hfg_output_gamma = nn.Parameter(torch.zeros(1, self.feat_channels[0], 1, 1))

    def _concat(self, x1, x2):
        x1 = F.avg_pool3d(
            x1.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        x2 = F.avg_pool3d(
            x2.unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1, 1)).squeeze(1)
        return torch.concat([x1, x2], axis=1)

    def show_feat(self, input):
        feat = input[0][0]
        plt.imshow(feat.detach().cpu(), cmap='gray')
        plt.show()

    def forward(self, x):
        conv1 = self.conv_layer1_1(x)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1(la1)
        st2 = self.st2(tr1)
        st2, prior2 = self.shf2(st2)
        skip21 = st2[0]

        tr2 = self.tr2(st2[-1])
        st3 = self.st3(tr2)
        st3, prior3 = self.shf3(st3)
        skip31 = st3[0]

        tr3 = self.tr3(st3[-1])
        st4 = self.st4(tr3)
        st4, prior4 = self.shf4(st4)
        skip41 = st4[0]
        hfg_priors = [prior2, prior3, prior4] if self.hfg_enable_prior else None

        tr4 = self.tr4(st4[-1])
        st5 = self.st5(tr4, priors=hfg_priors)
        x5 = st5[-1]

        tr5 = self.tr5(st5[0], shape=skip41.shape[-2:])
        tr5[0] = self._concat(tr5[0], skip41)
        st6 = self.st6(tr5, priors=hfg_priors)
        x4 = st6[-1]

        tr6 = self.tr6(st6[0], shape=skip31.shape[-2:])
        tr6[0] = self._concat(tr6[0], skip31)
        st7 = self.st7(tr6, priors=hfg_priors)
        x3 = st7[-1]

        tr7 = self.tr7(st7[0], shape=skip21.shape[-2:])
        tr7[0] = self._concat(tr7[0], skip21)
        st8 = self.st8(tr7, priors=hfg_priors)
        x2 = st8[-1]

        tr8 = self.tr8(st8[0])
        st9 = self.st9(tr8, priors=hfg_priors)
        x1 = st9[-1]

        x = [x1, x2, x3, x4, x5]
        for i in range(len(x)):
            x[i] = F.avg_pool3d(
                x[i].unsqueeze(1), kernel_size=(2, 1, 1), stride=(2, 1,
                                                                  1)).squeeze(1)

        # upsampling
        x0_h, x0_w = (x[0]).shape[-2:]
        for i in range(1, len(x)):
            x[i] = F.interpolate(
                x[i],
                size=[x0_h, x0_w],
                mode='bilinear',
                align_corners=self.align_corners)
        x = torch.concat(x, axis=1)
        x = x + self.hfg_output_gamma * self.hfg_output_refine(x)

        return x


class Layer1(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 num_blocks,
                 has_se=False,
                 name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = nn.Sequential()

        for i in range(num_blocks):
            self.bottleneck_block_list.add_module(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else num_filters * 4,
                    num_filters=num_filters,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1)))

    def forward(self, x):
        conv = x
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Module):
    def __init__(self,
                 stride_pre,
                 in_channel,
                 stride_cur,
                 out_channels,
                 align_corners=False,
                 name=None):
        super(TransitionLayer, self).__init__()
        self.align_corners = align_corners
        num_out = len(out_channels)
        if num_out != len(stride_cur):
            raise ValueError(
                'The length of `out_channels` does not equal to the length of `stride_cur`'
                    .format(num_out, len(stride_cur)))
        self.conv_bn_func_list = nn.ModuleList()
        for i in range(num_out):
            residual = nn.Sequential()
            if stride_cur[i] == stride_pre:
                if in_channel != out_channels[i]:
                    residual.add_module(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBnReLU(
                            in_channels=in_channel,
                            out_channels=out_channels[i],
                            kernel_size=3,
                            padding='same',
                        ))
                else:
                    residual = None
            elif stride_cur[i] > stride_pre:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=3,
                        stride=2,
                        padding='same',
                    ))
            else:
                residual.add_module(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBnReLU(
                        in_channels=in_channel,
                        out_channels=out_channels[i],
                        kernel_size=1,
                        stride=1,
                        padding='same',
                    ))
            self.conv_bn_func_list.append(residual)

    def forward(self, x, shape=None):
        outs = []
        for conv_bn_func in self.conv_bn_func_list:
            if conv_bn_func is None:
                outs.append(x)
            else:
                out = conv_bn_func(x)
                if shape is not None:
                    out = F.interpolate(
                        out,
                        shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                outs.append(out)
        return outs


class Branches(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels,
                 has_se=False, name=None, stage_index=None,
                 hfg_hf_refinement=True,
                 hfg_prior_injection=True,
                 hfg_prior_channels=(18, 36, 72),
                 hfg_pfconv_group3=1,
                 hfg_pfconv_group5=2,
                 hfg_drop_path=0.0):  # 新增stage_index参数
        super(Branches, self).__init__()
        self.basic_block_list = nn.ModuleList()
        self.stage_index = stage_index
        basic_att = False
        for i in range(len(out_channels)):
            basic_block_func = nn.ModuleList()

            if stage_index in [1, 2, 3, 4]:  
                k = 5
                # basic_att = True
            elif stage_index in [5]: 
                k = 3
                # basic_att = True
            elif stage_index == 6:  
                k = [7, 5][i]  
            elif stage_index == 7:  
                k = [9, 7][i]  
            elif stage_index == 8:  
                k = [11, 9][i]  
            elif stage_index == 9:  
                k = 13  

            for j in range(num_blocks[i]):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func.add_module(
                    f"bb_{name}_branch_{i + 1}_{j + 1}",
                    BasicBlock(num_channels=in_ch,
                               num_filters=out_channels[i],
                               stage_lk_size=k,
                               has_att = basic_att,
                               stage_index = stage_index,
                               hfg_hf_refinement=hfg_hf_refinement,
                               hfg_prior_injection=hfg_prior_injection,
                               hfg_prior_channels=hfg_prior_channels,
                               hfg_pfconv_group3=hfg_pfconv_group3,
                               hfg_pfconv_group5=hfg_pfconv_group5,
                               hfg_drop_path=hfg_drop_path)  
                )
            self.basic_block_list.append(basic_block_func)

    def forward(self, x, priors=None):
        outs = []
        for idx, input in enumerate(x):
            conv = input
            for basic_block_func in self.basic_block_list[idx]:
                conv = basic_block_func(conv, priors=priors)
            outs.append(conv)
        return outs


class BottleneckBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBnReLU(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            padding='same',
        )

        self.conv2 = ConvBnReLU(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            padding='same',
        )

        self.conv3 = ConvBn(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            padding='same',
        )

        if self.downsample:
            self.conv_down = ConvBn(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                padding='same',
            )

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name=name + '_fc')

    def forward(self, x):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = conv3 + residual
        y = F.relu(y)
        return y


# def get_bn(channels):
#     return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2   
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result



class BasicBlock(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stage_lk_size,
                 has_att,
                 stage_index,
                 hfg_hf_refinement=True,
                 hfg_prior_injection=True,
                 hfg_prior_channels=(18, 36, 72),
                 hfg_pfconv_group3=1,
                 hfg_pfconv_group5=2,
                 hfg_drop_path=0.0
                 ):
        super(BasicBlock, self).__init__()
        self.large_kernel = ReparamLargeKernelBlock(dim=num_channels, kernel_size=stage_lk_size)
        # self.has_att = has_att
        # self.att = MultiSpectralAttentionLayer(num_filters, c2wh[num_filters], c2wh[num_filters],  reduction=2, freq_sel_method = 'top18')
        self.hfm = HFMModule(num_channels) if hfg_hf_refinement and stage_index in [2, 3, 4] else nn.Identity()
        self.mix = PriorGuidedPFConv(
            channels=num_channels,
            prior_channels=hfg_prior_channels,
            group3=hfg_pfconv_group3,
            group5=hfg_pfconv_group5,
            drop_path=hfg_drop_path
        ) if hfg_prior_injection and stage_index in [5, 6, 7, 8, 9] else None
        self.kernelsize = stage_lk_size
        self.stage_index = stage_index

    def forward(self, x, priors=None):
        # residual = x
        large_kernel_out = self.large_kernel(x)
        large_kernel_out = self.hfm(large_kernel_out)
        if self.mix is not None:
            large_kernel_out = self.mix(large_kernel_out, priors=priors)

        # if self.has_att:
        #     large_kernel_out = self.att(large_kernel_out)

        return large_kernel_out


class SELayer(nn.Module):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = nn.AdaptiveAvgPool2d(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            act="relu",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            act="sigmoid",
            param_attr=torch.ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, x):
        pool = self.pool2d_gap(x)
        pool = torch.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = torch.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = x * excitation
        return out


class Stage(nn.Module):
    def __init__(
        self,
        num_channels,
        num_modules,
        num_blocks,
        num_filters,
        has_se=False,
        multi_scale_output=True,
        name=None,
        align_corners=False,
        stage_index=None,
        hfg_hf_refinement=True,
        hfg_prior_injection=True,
        hfg_prior_channels=(18, 36, 72),
        hfg_pfconv_group3=1,
        hfg_pfconv_group5=2,
        hfg_drop_path=0.0
    ):
        super(Stage, self).__init__()
        self._num_modules = num_modules
        self.hfg_prior_injection = hfg_prior_injection
        self.stage_func_list = nn.Sequential()
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                self.stage_func_list.add_module(
                    f"stage_{name}_{i+1}",
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=f"{name}_{i+1}",
                        align_corners=align_corners,
                        stage_index=stage_index,
                        hfg_hf_refinement=hfg_hf_refinement,
                        hfg_prior_injection=hfg_prior_injection,
                        hfg_prior_channels=hfg_prior_channels,
                        hfg_pfconv_group3=hfg_pfconv_group3,
                        hfg_pfconv_group5=hfg_pfconv_group5,
                        hfg_drop_path=hfg_drop_path
                    )
                )
            else:
                self.stage_func_list.add_module(
                    f"stage_{name}_{i+1}",
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_blocks=num_blocks,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=f"{name}_{i+1}",
                        align_corners=align_corners,
                        stage_index=stage_index,
                        hfg_hf_refinement=hfg_hf_refinement,
                        hfg_prior_injection=hfg_prior_injection,
                        hfg_prior_channels=hfg_prior_channels,
                        hfg_pfconv_group3=hfg_pfconv_group3,
                        hfg_pfconv_group5=hfg_pfconv_group5,
                        hfg_drop_path=hfg_drop_path
                    )
                )

    def forward(self, x, priors=None):
        out = x
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out, priors=priors)
        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_channels,
        num_blocks,
        num_filters,
        has_se=False,
        multi_scale_output=True,
        name=None,
        align_corners=False,
        stage_index=None,
        hfg_hf_refinement=True,
        hfg_prior_injection=True,
        hfg_prior_channels=(18, 36, 72),
        hfg_pfconv_group3=1,
        hfg_pfconv_group5=2,
        hfg_drop_path=0.0
    ):
        super(HighResolutionModule, self).__init__()
        self.branches_func = Branches(
            num_blocks=num_blocks,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name,
            stage_index=stage_index,
            hfg_hf_refinement=hfg_hf_refinement,
            hfg_prior_injection=hfg_prior_injection,
            hfg_prior_channels=hfg_prior_channels,
            hfg_pfconv_group3=hfg_pfconv_group3,
            hfg_pfconv_group5=hfg_pfconv_group5,
            hfg_drop_path=hfg_drop_path
        )

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name,
            align_corners=align_corners)

    def forward(self, x, priors=None):
        out = self.branches_func(x, priors=priors)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None,
                 align_corners=False):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels
        self.align_corners = align_corners

        self.residual_func_list = nn.Sequential()
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                if j > i:
                    self.residual_func_list.add_module(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBn(
                            in_channels=in_channels[j],
                            out_channels=out_channels[i],
                            kernel_size=1,
                            padding='same',
                        ))
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBn(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[i],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                ))
                            pre_num_filters = out_channels[i]
                        else:
                            self.residual_func_list.add_module(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBnReLU(
                                    in_channels=pre_num_filters,
                                    out_channels=out_channels[j],
                                    kernel_size=3,
                                    stride=2,
                                    padding='same',
                                ))
                            pre_num_filters = out_channels[j]

        if len(self.residual_func_list) == 0:
            self.residual_func_list.add_module("identity",
                                               nn.Identity())  # for flops calculation

    def forward(self, x):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = x[i]
            residual_shape = residual.shape[-2:]

            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    y = F.interpolate(
                        y,
                        residual_shape,
                        mode='bilinear',
                        align_corners=self.align_corners)
                    residual = residual + y
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = residual + y

            residual = F.relu(residual)
            outs.append(residual)

        return outs


def HFGNet_W18_Small(**kwargs):
    model = HFGBackbone(
        stage1_num_modules=1,
        stage1_num_blocks=[2],
        stage1_num_channels=[64],
        stage2_num_modules=1,
        stage2_num_blocks=[2, 2],
        stage2_num_channels=[18, 36],
        stage3_num_modules=2,
        stage3_num_blocks=[2, 2],
        stage3_num_channels=[36, 72],
        stage4_num_modules=2,
        stage4_num_blocks=[2, 2],
        stage4_num_channels=[72, 144],
        stage5_num_modules=2,
        stage5_num_blocks=[2, 2],
        stage5_num_channels=[144, 288],
        stage6_num_modules=1,
        stage6_num_blocks=[2, 2],
        stage6_num_channels=[72, 144],
        stage7_num_modules=1,
        stage7_num_blocks=[2, 2],
        stage7_num_channels=[36, 72],
        stage8_num_modules=1,
        stage8_num_blocks=[2, 2],
        stage8_num_channels=[18, 36],
        stage9_num_modules=1,
        stage9_num_blocks=[2],
        stage9_num_channels=[18],
        **kwargs)
    return model


def HFGNet_W18(**kwargs):
    model = HFGBackbone(
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(18, 36),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(36, 72),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(72, 144),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(144, 288),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(72, 144),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(36, 72),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(18, 36),
        stage9_num_modules=1,
        stage9_num_blocks=(4,),
        stage9_num_channels=(18,),
        **kwargs)
    return model


def HFGNet_W48(**kwargs):
    kwargs.setdefault('hfg_pfconv_group3', 8)
    kwargs.setdefault('hfg_pfconv_group5', 16)
    model = HFGBackbone(
        stage1_num_modules=1,
        stage1_num_blocks=(4,),
        stage1_num_channels=(64,),
        stage2_num_modules=1,
        stage2_num_blocks=(4, 4),
        stage2_num_channels=(48, 96),
        stage3_num_modules=5,
        stage3_num_blocks=(4, 4),
        stage3_num_channels=(96, 192),
        stage4_num_modules=2,
        stage4_num_blocks=(4, 4),
        stage4_num_channels=(192, 384),
        stage5_num_modules=2,
        stage5_num_blocks=(4, 4),
        stage5_num_channels=(384, 768),
        stage6_num_modules=1,
        stage6_num_blocks=(4, 4),
        stage6_num_channels=(192, 384),
        stage7_num_modules=1,
        stage7_num_blocks=(4, 4),
        stage7_num_channels=(96, 192),
        stage8_num_modules=1,
        stage8_num_blocks=(4, 4),
        stage8_num_channels=(48, 96),
        stage9_num_modules=1,
        stage9_num_blocks=(4,),
        stage9_num_channels=(48,),
        **kwargs)
    return model
