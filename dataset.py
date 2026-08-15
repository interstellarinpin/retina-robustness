import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BATCH_SIZE = 4
IMAGE_SIZE = 320
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_FRACTION = 0.20

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# FILE PATHS
# ============================================================

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
    """
    Remove most of the black background surrounding the retina.

    Input:
        PIL RGB image

    Output:
        cropped PIL RGB image
    """

    array = np.array(image)

    # Convert RGB image to approximate grayscale brightness
    gray = array.mean(axis=2)

    # Retina pixels should be brighter than the black background
    mask = gray > 10

    # Find rows and columns containing retinal pixels
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    # Safety fallback
    if len(rows) == 0 or len(cols) == 0:
        return image

    top = rows[0]
    bottom = rows[-1]
    left = cols[0]
    right = cols[-1]

    return image.crop((left, top, right + 1, bottom + 1))


# ============================================================
# CUSTOM DATASET
# ============================================================

class IDRiDDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None
    ):
        self.labels = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

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

        image = Image.open(image_path).convert("RGB")

        # Remove most of surrounding black background
        image = crop_retina(image)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

# Training images get mild augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation/test images should NOT be randomly augmented
evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# READ TRAINING LABELS
# ============================================================

all_train_labels = pd.read_csv(train_csv)

print("Full training class counts:")
print(
    all_train_labels[
        "Retinopathy grade"
    ].value_counts().sort_index()
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

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

    train_parts.append(training_group)
    validation_parts.append(validation_group)


train_dataframe = pd.concat(
    train_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


validation_dataframe = pd.concat(
    validation_parts
).sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


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


# ============================================================
# OFFICIAL TEST DATA
# ============================================================

test_dataframe = pd.read_csv(test_csv)


# ============================================================
# CREATE DATASETS
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
# RESNET-18
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

# Replace ImageNet's 1000-class output
# with our 5 DR grades
model.fc = nn.Linear(
    model.fc.in_features,
    5
)

# We are fine-tuning EVERYTHING.
# Nothing is frozen.


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

# Ordinary cross entropy because your current goal
# is overall classification accuracy.
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
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

    for batch, (X, y) in enumerate(
        dataloader
    ):
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
# EVALUATION
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

    num_classes = 5

    confusion = torch.zeros(
        num_classes,
        num_classes,
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
# TRAIN AND SAVE BEST VALIDATION MODEL
# ============================================================

best_validation_accuracy = 0

best_model_path = "best_model.pth"


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


    # Save the best model based ONLY
    # on validation accuracy
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
            "Saved new best model."
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        best_model_path,
        weights_only=True
    )
)


print(
    "\n=================================="
)

print(
    "BEST VALIDATION ACCURACY:"
)

print(
    f"{100 * best_validation_accuracy:.1f}%"
)

print(
    "=================================="
)


# ============================================================
# FINAL OFFICIAL TEST
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


print("\nFINAL TEST RESULTS")
print("----------------------------")

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


# ============================================================
# PER-CLASS RECALL
# ============================================================

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