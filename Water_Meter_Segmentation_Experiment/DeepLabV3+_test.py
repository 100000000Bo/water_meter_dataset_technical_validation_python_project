import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import os
import numpy as np

# 使用与训练集相同的自定义数据集类
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
        return image, mask, self.images[idx]  # 返回文件名

# 评估模型并计算精度、召回率和F1 Score
def evaluate_model(model, dataloader, device, save_dir="dataset/predictions"):
    model.to(device)
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []
    total_time = 0

    # 创建保存预测结果的文件夹
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            start_time = time.time()
            outputs = model(images)["out"]
            total_time += time.time() - start_time

            preds = torch.sigmoid(outputs) > 0.5  # 二值化预测结果
            preds = preds.cpu().numpy().astype(int)
            masks = masks.cpu().numpy().astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(masks.flatten())

            # 保存每张图像的预测结果
            for i, filename in enumerate(filenames):
                pred_image = (preds[i][0] * 255).astype(np.uint8)  # 转换为0-255的灰度图
                pred_pil = Image.fromarray(pred_image)
                pred_pil.save(os.path.join(save_dir, filename))

    # 计算精度、召回率和F1 Score
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, recall, precision, f1, total_time

# 设置路径和参数
test_images_dir = "dataset/fenge-test"  # 替换为你的测试图像文件夹路径
test_masks_dir = "dataset/fenge-test-label"  # 替换为你的测试标签文件夹路径
batch_size = 4

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = WaterMeterDataset(test_images_dir, test_masks_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1))  # 确保输出通道数与训练时一致
#model.load_state_dict(torch.load("deeplabv3_water_meter.pth", map_location=device), strict=False)

# 评估模型
accuracy, recall, precision, f1, total_time = evaluate_model(model, test_loader, device)
print(f"测试集上的预测精度: {accuracy * 100:.2f}%")
print(f"测试集上的召回率: {recall * 100:.2f}%")
print(f"测试集上的精确率: {precision * 100:.2f}%")
print(f"测试集上的F1分数: {f1 * 100:.2f}%")
print(f"测试集上的总耗时: {total_time:.2f}秒")
