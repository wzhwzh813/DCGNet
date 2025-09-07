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
# from model import DCGNet
from DCGNet_model import DCGNet

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
num_epochs = 150  
continue_train = True
start_epoch = 0
loss_file_path = 'DCGNet_loss.txt'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),
])

# 数据加载
train_clean_dir = r"E:\pycharm\DCGNet\datasets\Haze4K\train\gt"
train_noisy_dir = r"E:\pycharm\DCGNet\datasets\Haze4K\train\haze"

# 创建数据集
train_dataset = MultiModalDataset(train_clean_dir, train_noisy_dir, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

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

# 训练状态变量
best_train_loss = float('inf')
# best_model_path = 'model/haze4k_best_model.pth'
best_model_path = 'model/DCGNet_best_model.pth'

# 加载模型
if continue_train and os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    model.module.load_state_dict(torch.load(best_model_path))
    print("Model loaded.")

# 训练过程
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_train_loss = 0
    start_time = time.time()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', unit='batch') as pbar:
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
            pbar.set_postfix({'loss': epoch_train_loss / (pbar.n + 1)})
            pbar.update(1)
    avg_train_loss = epoch_train_loss / len(train_loader)
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s')

    # 只将训练集损失值写入文件
    with open(loss_file_path, 'a') as f:
        f.write(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}\n')

    # 保存最佳训练模型
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save(model.module.state_dict(), best_model_path)
        print(f'Train loss improved. Best model saved with Train Loss: {best_train_loss:.4f}')
    else:
        print(f'Train loss did not improve. Best Train Loss remains: {best_train_loss:.4f}')

    scheduler.step(avg_train_loss)




