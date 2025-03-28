import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os


# 数据集定义
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


# DenseNet 模型定义
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



# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet_Meter().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 20
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_full = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        # 计算每个数字的损失
        loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(5))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算精度
        predicted = torch.argmax(outputs, dim=2)
        correct_full += (predicted == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_full / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'water_meter_densenet_model.pth')

end_time = time.time()
total_training_time = end_time - start_time
print(f"训练总耗时: {total_training_time:.2f}秒")


# 测试与结果保存
def evaluate_and_save_predictions(model, test_loader, allowed_error=1, output_file='predictions.csv'):
    model.eval()
    correct_full = 0
    correct_allowed = 0
    total = 0
    total_inference_time = 0

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'true_labels', 'predicted_labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                start_time = time.time()
                outputs = model(images)
                inference_time = time.time() - start_time
                total_inference_time += inference_time

                predicted = torch.argmax(outputs, dim=2)
                correct_full += (predicted == labels).all(dim=1).sum().item()

                for i in range(labels.size(0)):
                    errors = (predicted[i] != labels[i]).sum().item()
                    if errors <= allowed_error:
                        correct_allowed += 1

                    true_labels = ''.join(map(str, labels[i].cpu().numpy()))
                    predicted_labels = ''.join(map(str, predicted[i].cpu().numpy()))

                    writer.writerow({'filename': 'unknown', 'true_labels': true_labels, 'predicted_labels': predicted_labels})

                total += labels.size(0)

    accuracy_full = correct_full / total
    accuracy_allowed = correct_allowed / total
    avg_inference_time_per_image = total_inference_time / total

    print(f"Test Accuracy (No error): {accuracy_full*100:.2f}%")
    print(f"Test Accuracy (Allow {allowed_error} error): {accuracy_allowed*100:.2f}%")
    print(f"Total Inference Time: {total_inference_time:.4f} s")
    print(f"Average Inference Time per Image: {avg_inference_time_per_image*1000:.4f} ms")


# 加载测试集进行评估
evaluate_and_save_predictions(model, test_loader)
