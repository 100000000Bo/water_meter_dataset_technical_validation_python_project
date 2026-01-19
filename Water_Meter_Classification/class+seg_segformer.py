import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F

from read_csv import select_images_for_seg_testing


# Custom dataset class consistent with the training set
class WaterMeterDataset(Dataset):
    def __init__(self, image_filenames, images_dir, masks_dir, transform=None):
        self.image_filenames = image_filenames
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Binary mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, filename


# Evaluate model and compute metrics
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0

    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            start_time = time.time()
            outputs = model(images).logits
            outputs = F.interpolate(outputs, size=(512, 512), mode="bilinear", align_corners=False)
            total_time += time.time() - start_time

            preds = torch.sigmoid(outputs) > 0.5  # Binarize predictions
            preds = preds.cpu().numpy().astype(int)
            masks = masks.cpu().numpy().astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(masks.flatten())

            # Optional: save predicted images
            for i, filename in enumerate(filenames):
                pred_image = (preds[i][0] * 255).astype(np.uint8)
                pred_pil = Image.fromarray(pred_image)

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, recall, precision, f1, total_time


# Paths and parameters
test_images_dir = 'C:/Users/yibo/Desktop/Word-Wheel_Water_Meter_Dataset/detection/test/test_img'
test_masks_dir = 'C:/Users/yibo/Desktop/Word-Wheel_Water_Meter_Dataset/detection/test/test_seg_label'
test_csv_file = 'C:/Users/yibo/Desktop/Word-Wheel_Water_Meter_Dataset/detection/test/test_class_label_CSV.csv'
batch_size = 4

# Select images for testing
which_class = 'six-digit'
image_filenames = select_images_for_seg_testing(test_csv_file, which_class, 900)
print(which_class)
print(len(image_filenames))

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = WaterMeterDataset(image_filenames, test_images_dir, test_masks_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load SegFormer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.decode_head.classifier = nn.Conv2d(256, 1, kernel_size=(1, 1))  # Match output channels
model.load_state_dict(torch.load("seg_model/segformer_water_meter.pth", map_location=device), strict=False)

# Evaluate model
accuracy, recall, precision, f1, total_time = evaluate_model(model, test_loader, device)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print(f"Recall on test set: {recall * 100:.2f}%")
print(f"Precision on test set: {precision * 100:.2f}%")
print(f"F1 score on test set: {f1 * 100:.2f}%")
print(f"Total evaluation time: {total_time:.2f} seconds")
