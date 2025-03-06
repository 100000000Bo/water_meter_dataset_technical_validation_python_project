import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp

# 自定义数据集类
class WaterMeterDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 二值标签，单通道
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# 定义训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    start_time = time.time()  # 记录训练开始时间
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算总耗时
    print(f"训练总耗时: {total_training_time:.2f}秒")

# 设置路径和参数
images_dir = "dataset/train_img3k"  # 替换为你的图像文件夹路径
masks_dir = "dataset/train_label3k"    # 替换为你的标签文件夹路径
batch_size = 4
num_epochs = 20
learning_rate = 1e-4

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_dataset = WaterMeterDataset(images_dir, masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化PSPNet模型、损失函数和优化器
model = smp.PSPNet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)  # 二分类输出为1通道
criterion = nn.BCEWithLogitsLoss()  # 使用二分类损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# 保存模型
torch.save(model.state_dict(), "pspnet_water_meter.pth")
print("模型已保存为 pspnet_water_meter.pth")
