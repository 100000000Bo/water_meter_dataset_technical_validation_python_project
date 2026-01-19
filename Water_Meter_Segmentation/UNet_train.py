import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


class WaterMeterDataset(Dataset):
    """
    Custom dataset for water meter segmentation.
    Images and masks are paired by filename.
    """
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
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


class UNet(nn.Module):
    """
    UNet-style architecture with a ResNet34 encoder.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.encoder.fc = nn.Identity()

        self.upconv1 = self._upconv_block(512, 256)
        self.upconv2 = self._upconv_block(256, 128)
        self.upconv3 = self._upconv_block(128, 64)
        self.upconv4 = self._upconv_block(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)

        x = nn.functional.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=False
        )

        return self.final_conv(x)


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.6f}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")


# Paths and hyperparameters
images_dir = "dataset/train_img3k"
masks_dir = "dataset/train_label3k"
batch_size = 4
num_epochs = 20
learning_rate = 1e-4

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

train_dataset = WaterMeterDataset(images_dir, masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = UNet(in_channels=3, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs, device)

torch.save(model.state_dict(), "unet_water_meter.pth")
print("Model saved as unet_water_meter.pth")
