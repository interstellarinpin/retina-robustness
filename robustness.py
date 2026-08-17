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

BASELINE_MODEL_PATH = "best_model.pth"
ROBUST_MODEL_PATH = "best_robust_model.pth"


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


np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# RETINA CROP
# ============================================================

def crop_retina(image):

    array = np.array(image)

    gray = array.mean(axis=2)

    mask = gray > 10

    rows = np.where(
        mask.any(axis=1)
    )[0]

    cols = np.where(
        mask.any(axis=0)
    )[0]

    if (
        len(rows) == 0
        or len(cols) == 0
    ):
        return image

    return image.crop(
        (
            cols[0],
            rows[0],
            cols[-1] + 1,
            rows[-1] + 1
        )
    )


# ============================================================
# GAUSSIAN NOISE
# ============================================================

def apply_gaussian_noise(
    image,
    noise_std,
    seed
):

    if noise_std <= 0:
        return image

    array = np.array(
        image,
        dtype=np.float32
    )

    rng = np.random.default_rng(
        seed
    )

    noise = rng.normal(
        0,
        noise_std,
        size=array.shape
    )

    array = np.clip(
        array + noise,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="RGB"
    )


# ============================================================
# JPEG COMPRESSION
# ============================================================

def apply_jpeg_compression(
    image,
    quality
):

    if quality >= 100:
        return image

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    image = Image.open(
        buffer
    ).convert(
        "RGB"
    ).copy()

    buffer.close()

    return image


# ============================================================
# ROBUSTNESS DATASET
# ============================================================

class RobustnessDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_dir,
        blur_radius=0,
        brightness=1.0,
        contrast=1.0,
        noise_std=0.0,
        jpeg_quality=100
    ):

        self.labels = pd.read_csv(
            csv_file
        )

        self.image_dir = image_dir

        self.blur_radius = blur_radius
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality

        self.transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

    def __len__(self):
        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.labels.iloc[
            idx
        ]

        image_name = row[
            "Image name"
        ]

        label = int(
            row[
                "Retinopathy grade"
            ]
        )

        image_path = os.path.join(
            self.image_dir,
            image_name + ".jpg"
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        # ----------------------------------------
        # SAME BASIC PREPROCESSING
        # ----------------------------------------

        image = crop_retina(
            image
        )

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        # ----------------------------------------
        # BLUR
        # ----------------------------------------

        if self.blur_radius > 0:

            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=self.blur_radius
                )
            )

        # ----------------------------------------
        # BRIGHTNESS
        # ----------------------------------------

        if self.brightness != 1.0:

            image = ImageEnhance.Brightness(
                image
            ).enhance(
                self.brightness
            )

        # ----------------------------------------
        # CONTRAST
        # ----------------------------------------

        if self.contrast != 1.0:

            image = ImageEnhance.Contrast(
                image
            ).enhance(
                self.contrast
            )

        # ----------------------------------------
        # GAUSSIAN NOISE
        # ----------------------------------------

        if self.noise_std > 0:

            image = apply_gaussian_noise(
                image,
                self.noise_std,

                # Same corrupted image is used
                # for both models.
                seed=SEED + idx
            )

        # ----------------------------------------
        # JPEG
        # ----------------------------------------

        if self.jpeg_quality < 100:

            image = apply_jpeg_compression(
                image,
                self.jpeg_quality
            )

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path):

    model = resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        5
    )

    model.load_state_dict(
        torch.load(
            path,
            weights_only=True
        )
    )

    model.eval()

    return model


baseline_model = load_model(
    BASELINE_MODEL_PATH
)

robust_model = load_model(
    ROBUST_MODEL_PATH
)


# ============================================================
# EVALUATE BOTH MODELS ON SAME IMAGES
# ============================================================

def evaluate_both(
    blur_radius=0,
    brightness=1.0,
    contrast=1.0,
    noise_std=0.0,
    jpeg_quality=100
):

    dataset = RobustnessDataset(
        test_csv,
        test_image_dir,

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

    baseline_correct = 0
    robust_correct = 0
    total = 0

    with torch.no_grad():

        for X, y in dataloader:

            baseline_pred = baseline_model(
                X
            ).argmax(
                dim=1
            )

            robust_pred = robust_model(
                X
            ).argmax(
                dim=1
            )

            baseline_correct += (
                baseline_pred == y
            ).sum().item()

            robust_correct += (
                robust_pred == y
            ).sum().item()

            total += y.size(0)

    baseline_accuracy = (
        baseline_correct
        / total
    )

    robust_accuracy = (
        robust_correct
        / total
    )

    return (
        baseline_accuracy,
        robust_accuracy
    )


# ============================================================
# PRINT TABLE
# ============================================================

def print_table(
    title,
    levels,
    evaluator
):

    print(
        "\n" + "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    print(
        f"{'Severity':<16}"
        f"{'Baseline':>12}"
        f"{'Robust':>12}"
        f"{'Δ Robust-BL':>16}"
    )

    print(
        "-" * 72
    )

    for level in levels:

        (
            baseline_acc,
            robust_acc
        ) = evaluator(
            level
        )

        change = (
            robust_acc
            - baseline_acc
        )

        print(
            f"{str(level):<16}"
            f"{100 * baseline_acc:>11.1f}%"
            f"{100 * robust_acc:>11.1f}%"
            f"{100 * change:>+15.1f}"
        )


# ============================================================
# CLEAN
# ============================================================

baseline_clean, robust_clean = (
    evaluate_both()
)


print(
    "\n" + "=" * 72
)

print(
    "CLEAN TEST"
)

print(
    "=" * 72
)

print(
    f"Baseline: "
    f"{100 * baseline_clean:.1f}%"
)

print(
    f"Robust:   "
    f"{100 * robust_clean:.1f}%"
)

print(
    f"Change:   "
    f"{100 * (robust_clean - baseline_clean):+.1f} points"
)


# ============================================================
# GAUSSIAN BLUR
# ============================================================

print_table(

    "GAUSSIAN BLUR",

    [
        0,
        0.5,
        1,
        2,
        4,
        8
    ],

    lambda level: evaluate_both(
        blur_radius=level
    )
)


# ============================================================
# BRIGHTNESS
# ============================================================

print_table(

    "BRIGHTNESS",

    [
        0.25,
        0.50,
        0.75,
        1.0,
        1.25,
        1.50,
        1.75
    ],

    lambda level: evaluate_both(
        brightness=level
    )
)


# ============================================================
# CONTRAST
# ============================================================

print_table(

    "CONTRAST",

    [
        0.25,
        0.50,
        0.75,
        1.0,
        1.25,
        1.50,
        2.0
    ],

    lambda level: evaluate_both(
        contrast=level
    )
)


# ============================================================
# GAUSSIAN NOISE
# ============================================================

print_table(

    "GAUSSIAN NOISE",

    [
        0,
        5,
        10,
        20,
        30,
        50
    ],

    lambda level: evaluate_both(
        noise_std=level
    )
)


# ============================================================
# JPEG COMPRESSION
# ============================================================

print_table(

    "JPEG COMPRESSION",

    [
        100,
        90,
        70,
        50,
        30,
        10
    ],

    lambda level: evaluate_both(
        jpeg_quality=level
    )
)