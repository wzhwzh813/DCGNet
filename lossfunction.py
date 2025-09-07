import torch
import math  # 引入math库
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        # 生成高斯核
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # 计算方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # 增加数值稳定性：确保方差不为负
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        # SSIM 公式中的常数C1, C2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM 公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class AdaptiveMixLoss(nn.Module):
    """
    自适应混合损失：根据当前SSIM动态调整L1和SSIM的权重。
    SSIM低时更关注结构，SSIM高时更关注像素精度。
    """
    def __init__(self, window_size=11, size_average=True, min_alpha=0.3, max_alpha=0.7):
        super(AdaptiveMixLoss, self).__init__()
        self.ssim_loss = SSIMLoss(window_size, size_average)
        self.l1_loss = nn.L1Loss()
        self.min_alpha = min_alpha  # L1最小权重
        self.max_alpha = max_alpha  # L1最大权重

    def forward(self, pred, target):
        # 这里的ssim_val实际相当于SSIM值，SSIM值越大，结构越相似，因此训练初期ssim_val值越小，训练后期ssim_val值越大
        ssim_val = 1 - self.ssim_loss(pred, target)  
        # 自适应权重：ssim_val越大，L1权重越大。训练初期注重结构（SSIM），训练后期注重像素精度（L1）
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * ssim_val.item()
        alpha = max(self.min_alpha, min(self.max_alpha, alpha))
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        loss = alpha * l1 + (1 - alpha) * ssim
        return loss
