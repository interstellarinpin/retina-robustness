import io
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 8
IMAGE_SIZE = 384
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.20

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


train_csv = (
    "data/idrid/B. Disease Grading/"
    "2. Groundtruths/"
    "a. IDRiD_Disease Grading_Training Labels.csv"
)

train_image_dir = (
    "data/idrid/B. Disease Grading/"
    "1. Original Images/"
    "a. Training Set"
)

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


# ============================================================
# CROP BLACK BACKGROUND
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
# RANDOM GAUSSIAN NOISE
# ============================================================

class RandomGaussianNoise:
    def __init__(
        self,
        p=0.30,
        max_std=15.0
    ):
        self.p = p
        self.max_std = max_std

    def __call__(self, image):
        if random.random() > self.p:
            return image

        array = np.array(
            image,
            dtype=np.float32
        )

        noise_std = random.uniform(
            0,
            self.max_std
        )

        noise = np.random.normal(
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
# RANDOM JPEG COMPRESSION
# ============================================================

class RandomJPEGCompression:
    def __init__(
        self,
        p=0.30,
        min_quality=40,
        max_quality=100
    ):
        self.p = p
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, image):
        if random.random() > self.p:
            return image

        quality = random.randint(
            self.min_quality,
            self.max_quality
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        compressed = Image.open(
            buffer
        ).convert("RGB").copy()

        buffer.close()

        return compressed


# ============================================================
# RANDOM GAUSSIAN BLUR
# ============================================================

class RandomGaussianBlur:
    def __init__(
        self,
        p=0.30,
        max_radius=2.0
    ):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, image):
        if random.random() > self.p:
            return image

        radius = random.uniform(
            0,
            self.max_radius
        )

        return image.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )


# ============================================================
# DATASET
# ============================================================

class IDRiDDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):
        self.labels = dataframe.reset_index(
            drop=True
        )

        self.image_dir = image_dir
        self.transform = transform

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

        image = crop_retina(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    # Ordinary mild appearance augmentation
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.05
    ),

    # Corruption-aware augmentation
    RandomGaussianBlur(
        p=0.30,
        max_radius=2.0
    ),

    RandomGaussianNoise(
        p=0.30,
        max_std=15.0
    ),

    RandomJPEGCompression(
        p=0.30,
        min_quality=40,
        max_quality=100
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


evaluation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

all_train_labels = pd.read_csv(
    train_csv
)

print("\nFull training class counts:")

print(
    all_train_labels[
        "Retinopathy grade"
    ].value_counts().sort_index()
)


train_parts = []
validation_parts = []


for grade, group in all_train_labels.groupby(
    "Retinopathy grade"
):

    group = group.sample(
        frac=1,
        random_state=SEED
    )

    validation_size = max(
        1,
        round(
            len(group)
            * VALIDATION_FRACTION
        )
    )

    validation_group = group.iloc[
        :validation_size
    ]

    training_group = group.iloc[
        validation_size:
    ]

    train_parts.append(
        training_group
    )

    validation_parts.append(
        validation_group
    )


train_dataframe = pd.concat(
    train_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(
    drop=True
)


validation_dataframe = pd.concat(
    validation_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(
    drop=True
)


print("\nTraining split:")

print(
    train_dataframe[
        "Retinopathy grade"
    ].value_counts().sort_index()
)


print("\nValidation split:")

print(
    validation_dataframe[
        "Retinopathy grade"
    ].value_counts().sort_index()
)


test_dataframe = pd.read_csv(
    test_csv
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = IDRiDDataset(
    train_dataframe,
    train_image_dir,
    transform=train_transform
)

validation_dataset = IDRiDDataset(
    validation_dataframe,
    train_image_dir,
    transform=evaluation_transform
)

test_dataset = IDRiDDataset(
    test_dataframe,
    test_image_dir,
    transform=evaluation_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

model.fc = nn.Linear(
    model.fc.in_features,
    5
)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAIN
# ============================================================

def train(
    dataloader,
    model,
    loss_fn,
    optimizer
):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:

        pred = model(X)

        loss = loss_fn(
            pred,
            y
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predicted_classes = pred.argmax(
            dim=1
        )

        correct += (
            predicted_classes == y
        ).sum().item()

        total += y.size(0)

    average_loss = (
        total_loss
        / len(dataloader)
    )

    accuracy = (
        correct / total
    )

    return average_loss, accuracy


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    dataloader,
    model,
    loss_fn
):
    model.eval()

    total_loss = 0
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

            loss = loss_fn(
                pred,
                y
            )

            total_loss += loss.item()

            predicted_classes = pred.argmax(
                dim=1
            )

            correct += (
                predicted_classes == y
            ).sum().item()

            total += y.size(0)

            for true_label, predicted_label in zip(
                y,
                predicted_classes
            ):

                confusion[
                    true_label.item(),
                    predicted_label.item()
                ] += 1

    average_loss = (
        total_loss
        / len(dataloader)
    )

    accuracy = (
        correct / total
    )

    return (
        average_loss,
        accuracy,
        confusion
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_validation_accuracy = 0

best_model_path = (
    "best_robust_model.pth"
)


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train(
        train_dataloader,
        model,
        loss_fn,
        optimizer
    )

    (
        validation_loss,
        validation_accuracy,
        validation_confusion
    ) = evaluate(
        validation_dataloader,
        model,
        loss_fn
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train accuracy: "
        f"{100 * train_accuracy:.1f}%"
    )

    print(
        f"Validation loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{100 * validation_accuracy:.1f}%"
    )

    if (
        validation_accuracy
        > best_validation_accuracy
    ):

        best_validation_accuracy = (
            validation_accuracy
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "Saved new best robust model."
        )


# ============================================================
# LOAD BEST ROBUST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        weights_only=True
    )
)


print(
    "\n========================================"
)

print(
    "BEST ROBUST VALIDATION ACCURACY"
)

print(
    "========================================"
)

print(
    f"{100 * best_validation_accuracy:.1f}%"
)


# ============================================================
# CLEAN TEST EVALUATION
# ============================================================

(
    test_loss,
    test_accuracy,
    test_confusion
) = evaluate(
    test_dataloader,
    model,
    loss_fn
)


print(
    "\n========================================"
)

print(
    "ROBUST MODEL — CLEAN TEST RESULTS"
)

print(
    "========================================"
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{100 * test_accuracy:.1f}%"
)

print(
    "\nConfusion matrix:"
)

print(
    test_confusion
)


print(
    "\nPer-class recall:"
)


for grade in range(5):

    total_for_grade = (
        test_confusion[
            grade
        ].sum().item()
    )

    correct_for_grade = (
        test_confusion[
            grade,
            grade
        ].item()
    )

    if total_for_grade > 0:

        recall = (
            correct_for_grade
            / total_for_grade
        )

    else:
        recall = 0

    print(
        f"Grade {grade}: "
        f"{100 * recall:.1f}%"
    )