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


class MeterDataset(Dataset):
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

        return image, torch.tensor(label)


# 数据增强与预处理
transform = transforms.Compose([
    transforms.Resize((60, 200)),  # 修改图像尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
])

train_dataset = MeterDataset('train', transform=transform)
test_dataset = MeterDataset('test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class VGG16_Meter(nn.Module):
    def __init__(self):
        super(VGG16_Meter, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)  # 使用预训练权重
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 50)  # 输出 5 个数字，每个数字有 10 个类（0-9）
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1, 5, 10)  # 将输出重塑为(batch_size, 5, 10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16_Meter().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 20

# 修改训练过程中的标签处理
start_time = time.time()  # 记录训练开始时间
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_full = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # 输出的形状为 (batch_size, 5, 10)
        outputs = model(images)

        # 计算每个数字的损失
        loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(5))  # 对每个数字的预测计算交叉熵损失
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算精度
        predicted = torch.argmax(outputs, dim=2)  # 获取每个数字的预测值
        correct_full += (predicted == labels).all(dim=1).sum().item()  # 判断完全正确的样本
        total_samples += labels.size(0)  # 累积样本总数

    # 计算每个 epoch 的平均损失和精度
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_full / total_samples

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'water_meter_vgg16_model.pth')

end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算总耗时
print(f"训练总耗时: {total_training_time:.2f}秒")



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
