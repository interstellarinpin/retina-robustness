import io
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 384
BATCH_SIZE = 8
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

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
# Same preprocessing used for baseline
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
# JPEG COMPRESSION
# ============================================================

def apply_jpeg_compression(image, quality):
    """
    Re-encode an image as JPEG in memory.

    Lower quality = stronger compression.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    compressed = Image.open(
        buffer
    ).convert("RGB")

    # Copy image so it no longer depends on the buffer
    compressed = compressed.copy()

    buffer.close()

    return compressed


# ============================================================
# GAUSSIAN NOISE
# ============================================================

def apply_gaussian_noise(
    image,
    noise_std,
    seed
):
    """
    Add zero-mean Gaussian noise.

    noise_std is measured on the usual 0-255 pixel scale.
    """

    if noise_std <= 0:
        return image

    array = np.array(
        image,
        dtype=np.float32
    )

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=array.shape
    )

    noisy = array + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        noisy,
        mode="RGB"
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
        blur_radius=0,
        brightness=1.0,
        contrast=1.0,
        noise_std=0.0,
        jpeg_quality=100
    ):
        self.labels = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.blur_radius = blur_radius
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]

        image_name = row["Image name"]
        label = int(
            row["Retinopathy grade"]
        )

        image_path = os.path.join(
            self.image_dir,
            image_name + ".jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------
        # Baseline preprocessing
        # ----------------------------------------

        image = crop_retina(image)

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # ----------------------------------------
        # Gaussian blur
        # ----------------------------------------

        if self.blur_radius > 0:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=self.blur_radius
                )
            )

        # ----------------------------------------
        # Brightness
        # ----------------------------------------

        if self.brightness != 1.0:
            image = ImageEnhance.Brightness(
                image
            ).enhance(
                self.brightness
            )

        # ----------------------------------------
        # Contrast
        # ----------------------------------------

        if self.contrast != 1.0:
            image = ImageEnhance.Contrast(
                image
            ).enhance(
                self.contrast
            )

        # ----------------------------------------
        # Gaussian noise
        # ----------------------------------------

        if self.noise_std > 0:
            image = apply_gaussian_noise(
                image,
                self.noise_std,

                # Fixed per-image noise so repeated
                # evaluations are reproducible
                seed=SEED + idx
            )

        # ----------------------------------------
        # JPEG compression
        # ----------------------------------------

        if self.jpeg_quality < 100:
            image = apply_jpeg_compression(
                image,
                self.jpeg_quality
            )

        # ----------------------------------------
        # Tensor + normalization
        # ----------------------------------------

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD BASELINE MODEL
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
# EVALUATION
# ============================================================

def evaluate(
    blur_radius=0,
    brightness=1.0,
    contrast=1.0,
    noise_std=0.0,
    jpeg_quality=100
):
    dataset = IDRiDRobustnessDataset(
        test_csv,
        test_image_dir,
        transform=transform,

        blur_radius=blur_radius,
        brightness=brightness,
        contrast=contrast,
        noise_std=noise_std,
        jpeg_quality=jpeg_quality
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
# CLEAN BASELINE
# ============================================================

clean_accuracy, clean_confusion = evaluate()

print("\n========================================")
print("CLEAN BASELINE")
print("========================================")

print(
    f"Accuracy: "
    f"{100 * clean_accuracy:.1f}%"
)

print("\nConfusion matrix:")
print(clean_confusion)


# ============================================================
# HELPER FOR PRINTING RESULTS
# ============================================================

def print_result(
    name,
    level,
    accuracy
):
    drop = (
        clean_accuracy
        - accuracy
    )

    print(
        f"{name} {level}: "
        f"{100 * accuracy:.1f}% "
        f"(drop: {100 * drop:.1f} points)"
    )


# ============================================================
# 1. GAUSSIAN BLUR
# ============================================================

blur_levels = [
    0,
    0.5,
    1,
    2,
    4,
    8
]

print("\n========================================")
print("GAUSSIAN BLUR")
print("========================================")

for level in blur_levels:

    accuracy, _ = evaluate(
        blur_radius=level
    )

    print_result(
        "Radius",
        level,
        accuracy
    )


# ============================================================
# 2. BRIGHTNESS
# ============================================================

brightness_levels = [
    0.25,
    0.50,
    0.75,
    1.0,
    1.25,
    1.50,
    1.75
]

print("\n========================================")
print("BRIGHTNESS")
print("========================================")

for level in brightness_levels:

    accuracy, _ = evaluate(
        brightness=level
    )

    print_result(
        "Factor",
        level,
        accuracy
    )


# ============================================================
# 3. CONTRAST
# ============================================================

contrast_levels = [
    0.25,
    0.50,
    0.75,
    1.0,
    1.25,
    1.50,
    2.0
]

print("\n========================================")
print("CONTRAST")
print("========================================")

for level in contrast_levels:

    accuracy, _ = evaluate(
        contrast=level
    )

    print_result(
        "Factor",
        level,
        accuracy
    )


# ============================================================
# 4. GAUSSIAN NOISE
# ============================================================

noise_levels = [
    0,
    5,
    10,
    20,
    30,
    50
]

print("\n========================================")
print("GAUSSIAN NOISE")
print("========================================")

print(
    "Noise std is measured on the "
    "0-255 pixel-value scale."
)

for level in noise_levels:

    accuracy, _ = evaluate(
        noise_std=level
    )

    print_result(
        "Std",
        level,
        accuracy
    )


# ============================================================
# 5. JPEG COMPRESSION
# ============================================================

jpeg_levels = [
    100,
    90,
    70,
    50,
    30,
    10
]

print("\n========================================")
print("JPEG COMPRESSION")
print("========================================")

print(
    "Lower JPEG quality means "
    "stronger compression."
)

for level in jpeg_levels:

    accuracy, _ = evaluate(
        jpeg_quality=level
    )

    print_result(
        "Quality",
        level,
        accuracy
    )


# ============================================================
# EXTREME FAILURE CONFUSION MATRICES
# ============================================================

print("\n========================================")
print("SELECTED EXTREME CONFUSION MATRICES")
print("========================================")


# Strong blur
_, blur_confusion = evaluate(
    blur_radius=8
)

print(
    "\nGaussian blur radius 8:"
)

print(
    blur_confusion
)


# Very dark
_, dark_confusion = evaluate(
    brightness=0.25
)

print(
    "\nBrightness factor 0.25:"
)

print(
    dark_confusion
)


# Very low contrast
_, contrast_confusion = evaluate(
    contrast=0.25
)

print(
    "\nContrast factor 0.25:"
)

print(
    contrast_confusion
)


# Strong Gaussian noise
_, noise_confusion = evaluate(
    noise_std=50
)

print(
    "\nGaussian noise std 50:"
)

print(
    noise_confusion
)


# Heavy JPEG compression
_, jpeg_confusion = evaluate(
    jpeg_quality=10
)

print(
    "\nJPEG quality 10:"
)

print(
    jpeg_confusion
)