# TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np

def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape 
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)




def hamming2D(M, N):
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class AdaptiveHighPassFilter(nn.Module):
    def __init__(self,
                 hr_channels,
                 # lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False,  # use offset generator or not
                 feature_resample_group=4,
                 comp_feat_upsample=True,  # use adaptive filters for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,  # for regularization, do not matter really
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        # self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(  # ALPF generator
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(  # adaptive high-pass generator
                self.compressed_channels,  # 输入通道=64
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,  # 输出通道=9×1×1×1=9
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(
                hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(
                hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        # group，在自适应高通滤波中，由于mask_c=k*k*scale*scale，一定会被k**2整除，mask_channel一定会是1或者scale的平方


        mask = mask.view(n, mask_channel, -1, h, w)    #[B,1,9,H,W]
        # 分成了 mask_channel 组，每组有9个通道，对于特征图上的每一个空间位置 ，Softmax 会将对应的 9 个通道值归一化，使它们的和为 1。这样，这 9 个值就构成了一个针对该位置的、独一无二的 3x3 卷积核的权重。
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)  #[B,1,9,H,W]
        # softmax 沿 dim=2（即第3维度）计算,softmax操作会作用在该位置对应的9个通道值上（h*w中的一个）,结果是每个位置的9个通道值被归一化为概率分布（和为1）

        # 例子：以空间位置 (0,0) 为例：
        # 输入：组0在 (0,0) 的9个通道值：[v0, v1, ..., v8]
        # softmax 后：[exp(v0)/sum, exp(v1)/sum, ..., exp(v8)/sum]（sum 是9个值的指数和）

        mask = mask.view(n, mask_channel, kernel, kernel, h, w) #  [B,1,3,3,H,W]
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel) # [B,H*W,3,3]，按照H*W展开，每个通道对应着特征图上每个像素的3*3卷积核权重
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming # [B,H*W,3,3]，对每个核乘上汉明窗，进行低通滤波，也就是先根据空间特征图生成，再低通滤波，于是最后一定是低通滤波的
        mask /= mask.sum(dim=(-1, -2), keepdims=True) #归一化
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)  #  [B,1,H,W,9]
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        #  [B,9,H,W] 把卷积核的参数按照原特征图尺寸返回，虽然这个时候他是特征图的尺寸，但里面已经是卷积核的参数了
        return mask

    def forward(self, hr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    # 从hr_feat得到初始高通滤波特征，这里得到3*3*1*1=9个通道，也说明这一步使用了9个卷积核, mask_hr_hr_feat在这一步是[B,9,h,w]
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel,hamming=self.hamming_highpass)
                    # kernel归一化得到初始高通滤波，[B,9,H,W]，把卷积核的参数按照原特征图尺寸返回，虽然这个时候他是特征图的尺寸，但里面已经是卷积核的参数了
                    hr_feat = hr_feat + hr_feat - carafe(hr_feat,mask_hr_init,self.highpass_kernel,self.up_group,1)  # 利用初始高通滤波对压缩hr_feat的高频增强 （x-x的低通结果=x的高通结果）
                    # CARAFE（Content-Aware ReAssembly of FEatures）是一种内容感知的上采样/滤波操作，对输入特征进行局部邻域加权平均操作，天然具有低通滤波特性
        return hr_feat

