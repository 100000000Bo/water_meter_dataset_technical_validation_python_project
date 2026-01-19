import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp


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
        # Binary ground truth mask (single channel)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    """
    Train the segmentation model and report the average loss per epoch
    as well as the total training time.
    """
    model.to(device)
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")


images_dir = "dataset/train_img3k"
masks_dir = "dataset/train_label3k"
batch_size = 4
num_epochs = 20
learning_rate = 1e-4

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_dataset = WaterMeterDataset(
    images_dir,
    masks_dir,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

model = smp.PSPNet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

torch.save(model.state_dict(), "pspnet_water_meter.pth")
print("Model weights saved to pspnet_water_meter.pth")
