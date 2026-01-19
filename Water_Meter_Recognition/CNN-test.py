import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class MeterDatasetTest(Dataset):
    """
    Test dataset for water meter digit recognition.
    The ground-truth labels are encoded in the first five characters of the filename.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        label = [int(char) for char in self.image_files[idx][:5]]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label), self.image_files[idx]


transform = transforms.Compose([
    transforms.Resize((60, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = MeterDatasetTest("test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNN_Meter(nn.Module):
    """
    CNN-based digit recognition model for five-digit water meter readings.
    """
    def __init__(self):
        super(CNN_Meter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 25, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 50)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.view(-1, 5, 10)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Meter().to(device)
model.load_state_dict(
    torch.load("model/water_meter_cnn_model.pth", map_location=device)
)
model.eval()


def evaluate_and_save_predictions(
    model,
    dataloader,
    allowed_error=1,
    output_file="predictions.csv"
):
    """
    Evaluate the model and save per-sample predictions.
    Accuracy is reported under strict matching and allowed-error criteria.
    """
    correct_full = 0
    correct_allowed = 0
    total = 0
    total_inference_time = 0.0

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["filename", "true_labels", "predicted_labels"]
        )
        writer.writeheader()

        with torch.no_grad():
            for images, labels, filenames in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                start_time = time.time()
                outputs = model(images)
                total_inference_time += time.time() - start_time

                predicted = torch.argmax(outputs, dim=2)

                correct_full += (predicted == labels).all(dim=1).sum().item()

                for i in range(labels.size(0)):
                    num_errors = (predicted[i] != labels[i]).sum().item()
                    if num_errors <= allowed_error:
                        correct_allowed += 1

                    writer.writerow({
                        "filename": filenames[i],
                        "true_labels": "".join(map(str, labels[i].cpu().numpy())),
                        "predicted_labels": "".join(map(str, predicted[i].cpu().numpy()))
                    })

                total += labels.size(0)

    accuracy_full = correct_full / total
    accuracy_allowed = correct_allowed / total
    avg_time_per_image = total_inference_time / total

    print(f"Test Accuracy (Exact Match): {accuracy_full * 100:.2f}%")
    print(f"Test Accuracy (≤{allowed_error} digit error): {accuracy_allowed * 100:.2f}%")
    print(f"Total Inference Time: {total_inference_time:.4f} s")
    print(f"Average Inference Time per Image: {avg_time_per_image * 1000:.4f} ms")


evaluate_and_save_predictions(model, test_loader)
