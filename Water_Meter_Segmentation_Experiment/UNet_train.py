import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision import models


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


# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=True)  # 使用ResNet34作为UNet的编码器
        self.encoder.fc = nn.Identity()  # 去掉ResNet34的分类头

        # 定义解码器
        self.upconv1 = self._upconv_block(512, 256)
        self.upconv2 = self._upconv_block(256, 128)
        self.upconv3 = self._upconv_block(128, 64)
        self.upconv4 = self._upconv_block(64, 32)

        # 最后的卷积层
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        x1 = self.encoder.conv1(x)
        x2 = self.encoder.bn1(x1)
        x3 = self.encoder.relu(x2)
        x4 = self.encoder.maxpool(x3)

        # 编码层的卷积
        x5 = self.encoder.layer1(x4)
        x6 = self.encoder.layer2(x5)
        x7 = self.encoder.layer3(x6)
        x8 = self.encoder.layer4(x7)

        # 解码器部分
        x = self.upconv1(x8)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)

        # 进行最终的上采样，确保输出尺寸为原始输入尺寸
        x = nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

        # 最后的卷积
        x = self.final_conv(x)
        return x


# 定义训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    start_time = time.time()  # 记录训练开始时间
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算总耗时
    print(f"训练总耗时: {total_training_time:.2f}秒")


# 设置路径和参数
images_dir = "dataset/train_img3k"  # 替换为你的图像文件夹路径
masks_dir = "dataset/train_label3k"  # 替换为你的标签文件夹路径
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

# 初始化模型、损失函数和优化器
model = UNet(in_channels=3, out_channels=1)  # UNet的输入通道为3（RGB），输出通道为1（二分类）
criterion = nn.BCEWithLogitsLoss()  # 使用二分类损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# 保存模型
torch.save(model.state_dict(), "unet_water_meter.pth")
print("模型已保存为 unet_water_meter.pth")
