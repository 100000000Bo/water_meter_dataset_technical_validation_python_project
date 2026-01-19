import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import os
import numpy as np


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

        # Return filename for saving prediction results
        return image, mask, self.images[idx]


class UNet(nn.Module):
    """
    U-Net architecture with a ResNet-34 encoder backbone.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encoder = models.resnet34(pretrained=True)
        self.encoder.fc = nn.Identity()

        self.upconv1 = self._upconv_block(512, 256)
        self.upconv2 = self._upconv_block(256, 128)
        self.upconv3 = self._upconv_block(128, 64)
        self.upconv4 = self._upconv_block(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            ),
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

        # Upsample to match the input resolution
        x = nn.functional.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=False
        )

        return self.final_conv(x)


def evaluate_model(model, dataloader, device, save_dir="dataset/predictions"):
    """
    Evaluate the segmentation model and compute accuracy, recall,
    precision, F1-score, and total inference time.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_time = 0.0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            start_time = time.time()
            outputs = model(images)
            total_time += time.time() - start_time

            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy().astype(int)
            masks = masks.cpu().numpy().astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(masks.flatten())

            for i, filename in enumerate(filenames):
                pred_image = (preds[i][0] * 255).astype(np.uint8)
                Image.fromarray(pred_image).save(
                    os.path.join(save_dir, filename)
                )

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, recall, precision, f1, total_time


test_images_dir = "dataset/fenge-test"
test_masks_dir = "dataset/fenge-test-label"
batch_size = 4

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = WaterMeterDataset(
    test_images_dir,
    test_masks_dir,
    transform=transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(
    torch.load("unet_water_meter.pth", map_location=device)
)

accuracy, recall, precision, f1, total_time = evaluate_model(
    model, test_loader, device
)

print(f"Test set accuracy: {accuracy * 100:.2f}%")
print(f"Test set recall: {recall * 100:.2f}%")
print(f"Test set precision: {precision * 100:.2f}%")
print(f"Test set F1-score: {f1 * 100:.2f}%")
print(f"Total inference time on test set: {total_time:.2f} seconds")
