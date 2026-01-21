"""
AXONNet (Axial Context Network): U-Net style semantic segmentation network with lightweight transformer-like bottleneck.

Main components
- Encoder: Stem `ConvBlock` followed by 4 Down stages. Each Down stage uses MaxPool2d + `ConvBlock`
  and a CASAB block (depthwise-separable conv + channel/spatial attention) to enhance local features.
- Bottleneck: `LightPCT` splits channels, applies lightweight conv + channel/spatial attention on a subset
  and projects back via 1x1 conv, providing long-range/context modeling at low cost.
- Decoder: 4 Up stages. Each stage upsamples, aligns channels with 1x1 conv on the upsampled tensor, then
  fuses with the corresponding skip via `DPCF` (AdaptiveCombiner on 4 channel chunks) and refines with `ConvBlock`.
- Head: `OutConv` 1x1 producing `num_classes` logits.

I/O
- Input: Tensor of shape (N, in_channels, H, W)
- Output: dict with key "out" mapping to raw logits of shape (N, num_classes, H, W)

Notes
- Default in_channels=1, base_c=64. Works with non-square inputs.
- CASAB attention modules are named `CASABChannelAttention` / `CASABSpatialAttention` to avoid name clashes
  with the attention modules used in `LightPCT`.

Road-specific tweaks
- For non-structured road segmentation (thin, elongated structures), an orientation-aware thin-structure
  enhancer (TSE) based on strip pooling is added in the decoder stages to improve long-range continuity
  along height/width while keeping computation light.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from math import sqrt
import torch.fft as fft

# LightPCT 模块及其依赖，直接集成到同一个文件中
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_prelu(output)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.SELU()

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)
        return output

def Split(x, p):
    c = int(x.size()[1])
    c1 = round(c * (1 - p))
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()
    return x1, x2

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.SELU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_out))

class LightPCT(nn.Module):
    def __init__(self, nIn, p=0.5):
        super().__init__()
        self.p = p
        c = int(nIn) - round(int(nIn) * (1 - p))
        self.conv_block = nn.Sequential(
            Conv(c, c, kSize=3, stride=1, padding=1, bn_acti=True),
            Conv(c, c, kSize=1, stride=1, padding=0, bn_acti=True)
        )
        self.channel_attention = ChannelAttention(c)
        self.spatial_attention = SpatialAttention()
        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=True)

    def forward(self, input):
        output1, output2 = Split(input, self.p)
        output2 = self.conv_block(output2)
        output2 = output2 * self.channel_attention(output2)
        output2 = output2 * self.spatial_attention(output2)
        output = torch.cat([output1, output2], dim=1)
        output = self.conv1x1(output)
        return output

# --- 核心模块 ---

class AdaptiveCombiner(nn.Module):
    def __init__(self):
        super(AdaptiveCombiner, self).__init__()
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        # p: low-level feature from skip connection (detail)
        # i: high-level feature from upsampling (semantic)
        batch_size, channel, w, h = p.shape
        d = self.d.expand(batch_size, channel, w, h)
        edge_att = torch.sigmoid(d)
        # Learnable weighted sum
        return edge_att * p + (1 - edge_att) * i

class conv_block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), norm_type='bn', activation=True, use_bias=True, groups = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=use_bias, groups = groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class DPCF(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.ac = AdaptiveCombiner()
        self.tail_conv = nn.Sequential(
            conv_block(in_features=in_features,
                       out_features=out_features,
                       kernel_size=(1, 1),
                       padding=(0, 0))
        )

    def forward(self, x_low, x_high ):
        # ** BUG FIX **: Removed the incorrect F.interpolate call.
        # Size alignment is already handled correctly by F.pad in the parent `Up` module.
        if x_low is not None:
            x_low_chunks = torch.chunk(x_low, 4, dim=1)
        if x_high is not None:
            x_high_chunks = torch.chunk(x_high, 4, dim=1)
        
        x0 = self.ac(x_low_chunks[0], x_high_chunks[0])
        x1 = self.ac(x_low_chunks[1], x_high_chunks[1])
        x2 = self.ac(x_low_chunks[2], x_high_chunks[2])
        x3 = self.ac(x_low_chunks[3], x_high_chunks[3])
        
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x

# Thin-structure Enhancer (TSE)
class StripPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.w_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.shape
        h_pool = x.mean(dim=3, keepdim=True) # (N,C,H,1)
        h_feat = self.h_proj(h_pool).expand(-1, -1, h, w)
        w_pool = x.mean(dim=2, keepdim=True) # (N,C,1,W)
        w_feat = self.w_proj(w_pool).expand(-1, -1, h, w)
        att = self.act(h_feat + w_feat)
        return att

class ThinStructureEnhancer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.strip = StripPooling(channels)

    def forward(self, x):
        y = self.dw1(x) + self.dw2(x)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        att = self.strip(y)
        y = y * att
        return x + y

# CASAB Module
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class CASABChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CASABChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = avg_out + max_out
        return x * scale

class CASABSpatialAttention(nn.Module):
    def __init__(self):
        super(CASABSpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, groups=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        min_out, _ = torch.min(x, dim=1, keepdim=True)
        sum_out = torch.sum(x, dim=1, keepdim=True)
        pool = torch.cat([mean_out, max_out, min_out, sum_out], dim=1)
        attention = self.conv(pool)
        return x * attention

class CASAB(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CASAB, self).__init__()
        self.convblock = ConvBlock(in_channels,in_channels)
        self.channel_attention = CASABChannelAttention(in_channels, reduction)
        self.spatial_attention = CASABSpatialAttention()

    def forward(self, x):
        x = self.convblock(x)
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        return ca + sa

# --- U-Net 主体结构 ---

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_casab = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            CASAB(out_channels)
        )

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_casab(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv_x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dpcf = DPCF(out_channels, out_channels)
        self.tse = ThinStructureEnhancer(out_channels)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the deeper layer (needs upsampling), x2 is the skip connection
        x1 = self.up(x1)
        
        # Pad x1 to match the size of x2, handling non-even dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Now x1 and x2 have the same spatial dimensions
        x1 = self.conv_x1(x1) # Align channels
        
        # Fuse using DPCF, enhance with TSE, then refine with ConvBlock
        x = self.dpcf(x_low=x2, x_high=x1)
        x = self.tse(x)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# 最终模型: AXONNet
class AXONNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, bilinear=True, base_c=64):
        super(AXONNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Encoder
        self.in_conv = ConvBlock(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)
        
        # Bottleneck
        self.bottleneck = LightPCT(nIn=base_c * 16, p=0.5)
        
        # Decoder
        self.up1 = Up(base_c * 16, base_c * 8, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        
        # Head
        self.out_conv = OutConv(base_c, num_classes)
        
    def forward(self, x):
        # Encoder path
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Bottleneck
        x5 = self.bottleneck(x5)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.out_conv(x)
        return {"out": logits}

if __name__ == "__main__":
    # Test with a non-square input to verify the fix
    input_tensor = torch.randn(2, 1, 125, 70) 
    model = AXONNet(in_channels=1, num_classes=2)
    output = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output['out'].shape) # Should match input H, W
