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


def evaluate_model(model, dataloader, device, save_dir="dataset/predictions"):
    """
    Evaluate the segmentation model and compute accuracy, recall, precision,
    F1-score, and total inference time.
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
            outputs = model(images)["out"]
            total_time += time.time() - start_time

            # Apply sigmoid and threshold to obtain binary predictions
            preds = torch.sigmoid(outputs) > 0.5

            preds = preds.cpu().numpy().astype(int)
            masks = masks.cpu().numpy().astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(masks.flatten())

            # Save prediction masks
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


# Paths and parameters
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

model = deeplabv3_resnet50(pretrained=False)
# Ensure the output channel matches the training configuration
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# model.load_state_dict(
#     torch.load("deeplabv3_water_meter.pth", map_location=device),
#     strict=False
# )

accuracy, recall, precision, f1, total_time = evaluate_model(
    model, test_loader, device
)

print(f"Test set accuracy: {accuracy * 100:.2f}%")
print(f"Test set recall: {recall * 100:.2f}%")
print(f"Test set precision: {precision * 100:.2f}%")
print(f"Test set F1-score: {f1 * 100:.2f}%")
print(f"Total inference time on test set: {total_time:.2f} seconds")
