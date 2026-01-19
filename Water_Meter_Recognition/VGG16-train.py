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
    """
    Custom dataset for multi-digit water meter recognition.
    The first five characters of the filename are used as digit labels.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path)
        label = [int(char) for char in self.image_files[idx][:5]]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


# Image preprocessing and normalization
transform = transforms.Compose([
    transforms.Resize((60, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

train_dataset = MeterDataset('train', transform=transform)
test_dataset = MeterDataset('test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class VGG16_Meter(nn.Module):
    """
    VGG16-based network for five-digit recognition.
    The output is reshaped to (batch_size, 5, 10),
    corresponding to five digits with ten classes each.
    """
    def __init__(self):
        super(VGG16_Meter, self).__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        )
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 50)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1, 5, 10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16_Meter().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

        loss = sum(
            criterion(outputs[:, i, :], labels[:, i])
            for i in range(5)
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.argmax(outputs, dim=2)
        correct_full += (predicted == labels).all(dim=1).sum().item()
        total_samples += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_full / total_samples

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
    )

torch.save(model.state_dict(), 'water_meter_vgg16_model.pth')

end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")


def evaluate_and_save_predictions(
    model,
    test_loader,
    allowed_error=1,
    output_file='predictions.csv'
):
    """
    Evaluate the model on the test set and save predictions.
    Accuracy is reported for exact match and for a relaxed
    criterion allowing limited digit errors.
    """
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
            for images, labels, filenames in test_loader:
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

                    true_labels = ''.join(
                        map(str, labels[i].cpu().numpy())
                    )
                    predicted_labels = ''.join(
                        map(str, predicted[i].cpu().numpy())
                    )

                    writer.writerow({
                        'filename': filenames[i],
                        'true_labels': true_labels,
                        'predicted_labels': predicted_labels
                    })

                total += labels.size(0)

    accuracy_full = correct_full / total
    accuracy_allowed = correct_allowed / total
    avg_inference_time = total_inference_time / total

    print(f"Test Accuracy (Exact Match): {accuracy_full * 100:.2f}%")
    print(
        f"Test Accuracy (≤ {allowed_error} digit error): "
        f"{accuracy_allowed * 100:.2f}%"
    )
    print(f"Total inference time: {total_inference_time:.4f} s")
    print(
        f"Average inference time per image: "
        f"{avg_inference_time * 1000:.4f} ms"
    )
