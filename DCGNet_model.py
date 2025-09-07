import torch
import torch.nn as nn
from torch.nn import functional as F


class EnhancedCBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernels=[3, 5, 7]):
        super(EnhancedCBAMLayer, self).__init__()

        # 通道注意力：Max + Avg + Variance 池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 新增方差池化
        self.var_pool = lambda x: torch.var(x, dim=[2, 3], keepdim=True)

        # 共享MLP，使用Mish激活函数
        self.mlp = nn.Sequential(
            nn.Conv2d(channel * 3, channel // reduction, 1, bias=False),  # 3倍通道数
            nn.Mish(inplace=True),  # 替换ReLU为Mish
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # 多尺度空间注意力
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False) 
            for k in spatial_kernels
        ])
        self.spatial_fusion = nn.Conv2d(len(spatial_kernels), 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 像素注意力
        # self.pa = PixelAttention(channel)

    def forward(self, x):
        # 通道注意力：Max + Avg + Variance
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        var_out = self.var_pool(x)
        
        # 拼接三种池化结果
        channel_feat = torch.cat([max_out, avg_out, var_out], dim=1)
        channel_out = self.sigmoid(self.mlp(channel_feat))
        x = channel_out * x

        # 多尺度空间注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([max_out, avg_out], dim=1)
        
        # 多尺度卷积
        spatial_outs = []
        for conv in self.spatial_convs:
            spatial_outs.append(conv(spatial_feat))
        
        # 融合多尺度结果
        spatial_out = self.spatial_fusion(torch.cat(spatial_outs, dim=1))
        spatial_out = self.sigmoid(spatial_out)
        
        x = spatial_out * x
        # x = self.pa(x)
        return x

## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor) ## ffn_expansion_factor可调，原文为2.5

        self.project_in =nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_5 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
        groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
        groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


class DCGAB(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernels=[3, 5, 7], ffn_expansion_factor=2.5, bias=False):
        super(DCGAB, self).__init__()
        # 批量归一化层
        self.bn1 = nn.BatchNorm2d(channel)
        # 增强版CBAM 层
        self.cbam = EnhancedCBAMLayer(channel, reduction, spatial_kernels)
        # 第二个批量归一化层
        self.bn2 = nn.BatchNorm2d(channel)
        # DGFF 层
        self.dgff = FeedForward(channel, ffn_expansion_factor, bias)
    def forward(self, x):
        # 先经过 BN 和 CBAM
        residual = x
        x = self.bn1(x)
        x = self.cbam(x)

        # 第一次残差相加
        x1 = x + residual

        # 第二次经过 BN 和 DGFF
        residual = x1
        x1 = self.bn2(x1)
        x1 = self.dgff(x1)

        # 第二次残差相加得到最终输出
        x2 = x1 + residual
        return x2


class SkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)  # 对跳跃特征归一化
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 降维

    def forward(self, x_deep, x_shallow):
        x_shallow = self.norm(x_shallow)
        x_shallow = self.conv(x_shallow)  # 对齐通道数
        return torch.cat([x_deep, x_shallow], dim=1)


# 网络主体
class DCGNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(DCGNet, self).__init__()

        # 浅层特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 定义 DCGAB 块和下采样操作
        self.block1 = DCGAB(base_channels)
        # 下采样操作（平均池化+1x1卷积）
        self.downsample1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)
        )

        self.block2 = DCGAB(base_channels * 2)
        # 下采样操作（平均池化+1x1卷积）
        self.downsample2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)
        )

        self.block3 = DCGAB(base_channels * 4)
        # 下采样操作（平均池化+1x1卷积）
        self.downsample3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)
        )

        # 中间的 DCGAB 块
        self.block4 = DCGAB(base_channels * 8)

        self.block5 = DCGAB(base_channels * 8)

        # 跳跃连接
        self.skip1 = SkipConnection(base_channels*8, base_channels*8)
        self.skip2 = SkipConnection(base_channels*4, base_channels*4)

        self.conv1x1_1 = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=1)  # 调整通道数
        # 上采样操作（插值+卷积，避免棋盘效应）
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        )
        self.block6 = DCGAB(base_channels * 4)

        self.conv1x1_2 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)  # 调整通道数
        # 上采样操作（插值+卷积，避免棋盘效应）
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        )
        self.block7 = DCGAB(base_channels * 2)

        # 上采样操作（插值+卷积，避免棋盘效应）
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        )
        self.block8 = DCGAB(base_channels)

        # 最后的 3x3 卷积用于输出
        self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 浅层特征提取
        x = self.initial_conv(x)
        print("input:",x.shape)
        # 下采样部分
        x1 = self.downsample1(self.block1(x))
        print("block1:", x1.shape)


        x2 = self.downsample2(self.block2(x1))
        print("block2:", x2.shape)

        x3 = self.downsample3(self.block3(x2))
        print("block3:", x3.shape)

        # 中间的 DCGAB 块
        x4 = self.block4(x3)
        print("block4:", x4.shape)


        x5 = self.block5(x4)
        print("block5:", x5.shape)
        x5 = self.skip1(x5, x3)
        print("block5_cat_block3:", x5.shape)

        x6 = self.conv1x1_1(x5)
        x6 = self.block6(self.upsample1(x6))
        print("block6:", x6.shape)
        x6 = self.skip2(x6, x2)
        print("block6_cat_block2:", x6.shape)


        x7 = self.conv1x1_2(x6)
        x7 = self.block7(self.upsample2(x7))
        print("block7:", x7.shape)

        x8 = self.block8(self.upsample3(x7))
        print("block8:",x8.shape)

        # 最后的 3x3 卷积用于输出重建
        output = self.output_conv(x8)
        print("output:", output.shape)

        return output

# # 使用示例
input_image = torch.randn(1, 3, 256, 256)  # 示例输入图像，假设大小为 256x256
model = DCGNet()
output_image = model(input_image)
print(output_image.shape)
