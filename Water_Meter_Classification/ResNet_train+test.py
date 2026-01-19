import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os

from read_csv import select_images_for_training

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset configuration
num_1 = 1000
num_0 = 1000
which_class = 'dark'

image_dir = 'C:/Users/yibo/Desktop/detection/train/train_img'
csv_file = 'C:/Users/yibo/Desktop/detection/train/train_class_label_CSV.csv'
list_1, list_0 = select_images_for_training(csv_file, which_class, num_1, num_0)

# Test dataset configuration
test_num_1 = 300
test_num_0 = 10000
test_image_dir = 'C:/Users/yibo/Desktop/detection/test/test_img'
test_csv_file = 'C:/Users/yibo/Desktop/detection/test/test_class_label_CSV.csv'
test_list_1, test_list_0 = select_images_for_training(
    test_csv_file, which_class, test_num_1, test_num_0
)

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Custom dataset definition
class WaterMeterDataset(Dataset):
    def __init__(self, image_dir, image_list, label, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.label
        if self.transform:
            image = self.transform(image)
        return image, label


# Construct training dataset
dirty_dataset = WaterMeterDataset(image_dir, list_1, label=1, transform=transform)
clean_dataset = WaterMeterDataset(image_dir, list_0, label=0, transform=transform)

dataset = torch.utils.data.ConcatDataset([dirty_dataset, clean_dataset])
train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# Construct test dataset
test_dirty_dataset = WaterMeterDataset(
    test_image_dir, test_list_1, label=1, transform=transform
)
test_clean_dataset = WaterMeterDataset(
    test_image_dir, test_list_0, label=0, transform=transform
)

test_dataset = torch.utils.data.ConcatDataset(
    [test_dirty_dataset, test_clean_dataset]
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Load a pretrained ResNet18 model for binary classification
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

max_test_acc = 0

# Model training
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    train_accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {total_loss / len(train_loader):.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    # Evaluation on the test set
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            test_loss += loss.item()

    test_accuracy = 100 * test_correct / test_total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Test Loss: {test_loss / len(test_loader):.4f}, "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    if test_accuracy > max_test_acc:
        max_test_acc = test_accuracy
        torch.save(model.state_dict(), "water_meter_model.pth")

print(f"Best test accuracy: {max_test_acc:.2f}%")
print("Model training completed. Weights have been saved.")
