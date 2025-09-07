import torch
import numpy as np
import torch.nn.functional as F

def PSNR(img1, img2):
    # 将模型输出裁剪到[0, 1]范围
    img1 = torch.clamp(img1, min=0, max=1)
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    PSNR = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return PSNR.item()


class SSIM:
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def ssim(self, img1, img2):
        (_, channel, _, _) = img1.size()
        self.create_window(self.window_size, channel)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def __call__(self, img1, img2):
        return self.ssim(img1, img2)

