import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 384
BATCH_SIZE = 8

test_csv = (
    "data/idrid/B. Disease Grading/"
    "2. Groundtruths/"
    "b. IDRiD_Disease Grading_Testing Labels.csv"
)

test_image_dir = (
    "data/idrid/B. Disease Grading/"
    "1. Original Images/"
    "b. Testing Set"
)

model_path = "best_model.pth"


# ============================================================
# CROP BLACK BACKGROUND
# Same logic used in baseline training
# ============================================================

def crop_retina(image):
    array = np.array(image)

    gray = array.mean(axis=2)

    mask = gray > 10

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return image

    top = rows[0]
    bottom = rows[-1]
    left = cols[0]
    right = cols[-1]

    return image.crop(
        (left, top, right + 1, bottom + 1)
    )


# ============================================================
# DATASET
# ============================================================

class IDRiDRobustnessDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
        blur_radius=0
    ):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.blur_radius = blur_radius

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_name = row["Image name"]
        label = int(row["Retinopathy grade"])

        image_path = os.path.join(
            self.image_dir,
            image_name + ".jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # 1. Same retinal crop as baseline
        image = crop_retina(image)

        # 2. Resize to exact model input size
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # 3. Apply corruption AFTER resizing
        if self.blur_radius > 0:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=self.blur_radius
                )
            )

        # 4. Convert to tensor + normalize
        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# Resize already happened above
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD SAVED BASELINE MODEL
# ============================================================

model = resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    5
)

model.load_state_dict(
    torch.load(
        model_path,
        weights_only=True
    )
)

model.eval()


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(blur_radius):
    dataset = IDRiDRobustnessDataset(
        test_csv,
        test_image_dir,
        transform=transform,
        blur_radius=blur_radius
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    correct = 0
    total = 0

    confusion = torch.zeros(
        5,
        5,
        dtype=torch.int64
    )

    with torch.no_grad():

        for X, y in dataloader:

            pred = model(X)

            predicted = pred.argmax(
                dim=1
            )

            correct += (
                predicted == y
            ).sum().item()

            total += y.size(0)

            for true_label, predicted_label in zip(
                y,
                predicted
            ):
                confusion[
                    true_label.item(),
                    predicted_label.item()
                ] += 1

    accuracy = correct / total

    return accuracy, confusion


# ============================================================
# BLUR EXPERIMENT
# ============================================================

blur_levels = [
    0,
    0.5,
    1,
    2,
    4,
    8
]


print("\nGaussian Blur Robustness")
print("------------------------")


clean_accuracy = None

results = []


for blur in blur_levels:

    accuracy, confusion = evaluate(
        blur
    )

    if blur == 0:
        clean_accuracy = accuracy

    drop = (
        clean_accuracy
        - accuracy
    )

    results.append(
        (
            blur,
            accuracy,
            drop
        )
    )

    print(
        f"Blur radius {blur}: "
        f"{100 * accuracy:.1f}% "
        f"(drop: {100 * drop:.1f} points)"
    )


# ============================================================
# CLEAN CONFUSION MATRIX
# ============================================================

clean_accuracy, clean_confusion = evaluate(
    0
)

print("\nClean confusion matrix:")
print(clean_confusion)


# ============================================================
# STRONG BLUR CONFUSION MATRIX
# ============================================================

strong_blur_accuracy, strong_blur_confusion = evaluate(
    8
)

print("\nBlur radius 8 confusion matrix:")
print(strong_blur_confusion)