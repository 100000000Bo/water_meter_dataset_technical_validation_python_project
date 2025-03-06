import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# 定义测试数据集类
class MeterDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = [int(char) for char in self.image_files[idx][:5]]  # 前五位作为标签

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label), self.image_files[idx]  # 返回文件名以便保存

# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((60, 200)),  # 修改图像尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

test_dataset = MeterDatasetTest('test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型结构
class DenseNet_Meter(nn.Module):
    def __init__(self):
        super(DenseNet_Meter, self).__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)  # 使用预训练权重
        self.features = densenet.features

        # 计算特征提取部分输出的特征维度
        self.feature_dim = 1024  # DenseNet121 的特征图通道数
        self.pool_size = (2, 7)  # 全局平均池化后特征图的空间维度 (height, width)

        # 动态构造分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * self.pool_size[0] * self.pool_size[1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 50)  # 输出 5 个数字，每个数字有 10 个类别
        )

    def forward(self, x):
        x = self.features(x)  # 提取特征
        x = nn.AdaptiveAvgPool2d(self.pool_size)(x)  # 自适应池化到固定大小
        x = torch.flatten(x, 1)  # 展平，输出形状 (batch_size, feature_dim)
        x = self.classifier(x)
        return x.view(-1, 5, 10)  # 输出形状为 (batch_size, 5, 10)


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet_Meter().to(device)
model.load_state_dict(torch.load('water_meter_densenet_model.pth', map_location=device))
model.eval()

# 定义评估和保存预测函数
def evaluate_and_save_predictions(model, test_loader, allowed_error=1, output_file='predictions.csv'):
    model.eval()
    correct_full = 0
    correct_allowed = 0
    total = 0
    total_inference_time = 0  # 累积推理时间

    # 打开文件用于保存预测结果
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'true_labels', 'predicted_labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for images, labels, filenames in test_loader:
                images, labels = images.to(device), labels.to(device)

                # 记录推理开始时间
                start_time = time.time()
                outputs = model(images)
                # 记录推理结束时间
                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # 计算每个数字的预测值
                predicted = torch.argmax(outputs, dim=2)  # 输出是一个 (batch_size, 5) 的数组

                # 计算完全没有出错的图片
                correct_full += (predicted == labels).all(dim=1).sum().item()

                # 计算允许一个数字出错的精度
                for i in range(labels.size(0)):
                    errors = (predicted[i] != labels[i]).sum().item()
                    if errors <= allowed_error:
                        correct_allowed += 1

                    # 保存每个样本的预测结果
                    true_labels = ''.join(map(str, labels[i].cpu().numpy()))  # 将标签转为字符串
                    predicted_labels = ''.join(map(str, predicted[i].cpu().numpy()))  # 将预测结果转为字符串

                    writer.writerow(
                        {'filename': filenames[i], 'true_labels': true_labels, 'predicted_labels': predicted_labels})

                total += labels.size(0)

    # 计算精度
    accuracy_full = correct_full / total
    accuracy_allowed = correct_allowed / total

    # 打印推理时间信息
    avg_inference_time_per_image = total_inference_time / total
    print(f"Test Accuracy (No error): {accuracy_full*100:.2f}%")
    print(f"Test Accuracy (Allow {allowed_error} error): {accuracy_allowed*100:.2f}%")
    print(f"Total Inference Time: {total_inference_time:.4f} s")
    print(f"Average Inference Time per Image: {avg_inference_time_per_image*1000:.4f} ms")


# 执行评估并保存结果
evaluate_and_save_predictions(model, test_loader)
