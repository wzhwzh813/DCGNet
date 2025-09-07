import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from glob import glob
import re
from PIL import Image
from tqdm import tqdm
import time
from lossfunction import AdaptiveMixLoss
from DCGNet_model import DCGNet
# from ablation.original_CBAM.origin_CBAM_model import DCGNet

from metric import PSNR, SSIM
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiModalDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.transform = transform

        # 获取所有gt文件名并排序
        clean_filenames = [os.path.basename(f) for f in glob(os.path.join(clean_dir, '*.png'))]
        clean_filenames_sorted = sorted(clean_filenames, key=lambda x: int(x.split('.')[0]))

        # 获取所有haze文件名，并通过正则表达式提取第一个数字并排序
        noisy_filenames = [os.path.basename(f) for f in glob(os.path.join(noisy_dir, '*.png'))]

        def extract_scene_number(filename):
            match = re.match(r'(\d+)_', filename)
            if match:
                return int(match.group(1))  # 提取haze图像的场景编号
            return None

        # 按照场景编号排序 haze 图像
        noisy_filenames_sorted = sorted(noisy_filenames, key=extract_scene_number)

        # 检查 gt 和 haze 是否能够按场景编号进行一一对应
        if len(clean_filenames_sorted) != len(noisy_filenames_sorted):
            raise ValueError(f"Number of clean images ({len(clean_filenames_sorted)}) and noisy images ({len(noisy_filenames_sorted)}) must be the same.")

        # 保存完整路径
        self.clean_paths = [os.path.join(clean_dir, f) for f in clean_filenames_sorted]
        self.noisy_paths = [os.path.join(noisy_dir, f) for f in noisy_filenames_sorted]
        # print("Clean Paths:", self.clean_paths)
        # print("Noisy Paths:", self.noisy_paths)

        print(f"Total {len(self.clean_paths)} image pairs found.")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        noisy_img = Image.open(self.noisy_paths[idx]).convert('RGB')
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        return noisy_img, clean_img
    
# 定义超参数
batch_size = 4
learning_rate = 1e-4
num_epochs = 1  
continue_train = True
start_epoch = 0

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),
])

# 数据加载
test_clean_dir = r"E:\pycharm\DCGNet\datasets\Haze4K\test\gt"
test_noisy_dir = r"E:\pycharm\DCGNet\datasets\Haze4K\test\haze"

# 创建数据集
test_dataset = MultiModalDataset(test_clean_dir, test_noisy_dir, transform=transform)

# 创建 DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 模型、损失函数和优化器
model = DCGNet().to(device)

from thop import profile
# 创建一个随机的输入张量用于FLOPs计算
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 使用 thop 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(dummy_input, ))

print(f"Total number of parameters: {params}")
print(f"Total number of FLOPs: {flops}")


# **删除 FLOPs 和参数计数器属性**
for module in model.modules():
    if hasattr(module, 'total_ops'):
        del module.total_ops
    if hasattr(module, 'total_params'):
        del module.total_params
        

model = nn.DataParallel(model)

criterion = AdaptiveMixLoss(window_size=11, size_average=True).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

best_model_path='DCGNet_best_model.pth'

# 加载之前保存的模型参数（如果需要）
if continue_train and os.path.exists(best_model_path):
    print(f"Loading model parameters from {best_model_path}")
    model.module.load_state_dict(torch.load(best_model_path))

# 初始化SSIM计算器
ssim_calculator = SSIM()

psnr_list = []
ssim_list = []
valid_count = 0

# 实时性统计变量
num_images = 0
total_infer_time = 0
warmup_iters = 5  # 前5个batch不计入统计

# 去噪示例并保存图像
with torch.no_grad():
    for batch_idx, (noisy_imgs, clean_imgs) in enumerate(tqdm(test_loader, desc="Processing Batches")):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # warmup: 前warmup_iters个batch不计入统计
        if batch_idx < warmup_iters:
            _ = model(noisy_imgs)
            num_images += noisy_imgs.size(0)
            continue

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        outputs = model(noisy_imgs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        total_infer_time += (end - start)
        num_images += noisy_imgs.size(0)

        # 将结果移回CPU
        outputs = outputs.cpu()
        noisy_imgs = noisy_imgs.cpu()
        clean_imgs = clean_imgs.cpu()

        for i in range(outputs.size(0)):
            out_img = outputs[i]
            gt_img = clean_imgs[i]
            # 检查NaN/inf
            if torch.isnan(out_img).any() or torch.isinf(out_img).any() or \
               torch.isnan(gt_img).any() or torch.isinf(gt_img).any():
                continue
            mse = torch.mean((out_img - gt_img) ** 2)
            if mse == 0 or mse < 0 or torch.isnan(mse) or torch.isinf(mse):
                continue
            psnr_val = PSNR(out_img, gt_img)
            ssim_val = ssim_calculator(out_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
            if not (np.isnan(psnr_val) or np.isinf(psnr_val) or np.isnan(ssim_val) or np.isinf(ssim_val)):
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                valid_count += 1

# 输出实时性指标
# 有效图片数 = 总图片数 - warmup阶段图片数
batch_size = test_loader.batch_size if hasattr(test_loader, 'batch_size') else 1
warmup_images = warmup_iters * batch_size
# 由于num_images在warmup阶段也累加了，所以要减去warmup_images
# 但如果最后num_images < warmup_images，说明数据量太小

if num_images > warmup_images and total_infer_time > 0:
    effective_images = num_images - warmup_images
    fps = effective_images / total_infer_time
    latency = (total_infer_time / effective_images) * 1000  # ms
    print(f"[Real-time] Total images (excluding warmup): {effective_images}")
    print(f"[Real-time] Total inference time: {total_infer_time:.4f} s")
    print(f"[Real-time] Average FPS: {fps:.2f}")
    print(f"[Real-time] Average latency per image: {latency:.2f} ms")
else:
    print("[Real-time] Not enough images for timing (check batch size and warmup_iters)")

# 计算平均PSNR和SSIM（只用有效值）
if valid_count > 0:
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    min_psnr = np.min(psnr_list)
    max_psnr = np.max(psnr_list)
    min_ssim = np.min(ssim_list)
    max_ssim = np.max(ssim_list)
else:
    avg_psnr = avg_ssim = min_psnr = max_psnr = min_ssim = max_ssim = float('nan')

print(f'Average PSNR: {avg_psnr:.4f} dB')
print(f'Average SSIM: {avg_ssim:.4f}')
print(f'Minimum PSNR: {min_psnr:.4f} dB')
print(f'Maximum PSNR: {max_psnr:.4f} dB')
print(f'Minimum SSIM: {min_ssim:.4f}')
print(f'Maximum SSIM: {max_ssim:.4f}')
print(f'Number of valid image pairs: {valid_count} / 1000')

with torch.no_grad():
    dataset = MultiModalDataset(test_clean_dir, test_noisy_dir, transform=transform)
    noisy_imgs, clean_imgs = [], []
    indices = [33, 66, 99, 132, 165, 198] 
    for i in indices:
        noisy_img, clean_img = dataset[i]
        noisy_imgs.append(noisy_img)
        clean_imgs.append(clean_img)
    noisy_imgs = torch.stack(noisy_imgs).to(device)
    clean_imgs = torch.stack(clean_imgs).to(device)

    outputs = model(noisy_imgs)
    outputs = outputs.cpu()
    noisy_imgs = noisy_imgs.cpu()
    clean_imgs = clean_imgs.cpu()

    # 计算PSNR值
    psnr_values = [PSNR(outputs[i], clean_imgs[i]) for i in range(6)]
    psnr_values_noisy = [PSNR(noisy_imgs[i], clean_imgs[i]) for i in range(6)]

    # 初始化SSIM计算器
    ssim_calculator = SSIM()

    # 计算SSIM值
    ssim_values = [ssim_calculator(outputs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item() for i in range(6)]
    ssim_values_noisy = [ssim_calculator(noisy_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item() for i in range(6)]

    fig, axs = plt.subplots(3, 6, figsize=(18, 9))  # 设置3行6列的子图布局
    for i in range(6):
        axs[0, i].imshow(transforms.ToPILImage()(noisy_imgs[i]))
        axs[0, i].set_title(f'Noisy\nPSNR: {psnr_values_noisy[i]:.2f} dB')
        axs[0, i].axis('off')
        axs[1, i].imshow(transforms.ToPILImage()(outputs[i]))
        axs[1, i].set_title(f'Denoised\nPSNR: {psnr_values[i]:.2f} dB,SSIM: {ssim_values[i]:.4f}')
        axs[1, i].axis('off')
        axs[2, i].imshow(transforms.ToPILImage()(clean_imgs[i]))
        axs[2, i].set_title('Clean')
        axs[2, i].axis('off')

# 保存并关闭图像
plt.tight_layout()
plt.savefig('model/1_haze4k_test.png')
# plt.savefig('model/haze4k_no_residual_connnection_test.png')
plt.close()

