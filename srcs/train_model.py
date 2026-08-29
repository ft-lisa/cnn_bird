from __future__ import annotations


import shutil

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


class BirdImageDataset(Dataset):
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: Path, transform: transforms.Compose | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: list[str] = []
        
        self._discover_samples()


    def _discover_samples(self) -> None:
        class_directories = sorted(path for path in self.root_dir.iterdir() if path.is_dir())

        for class_index, class_directory in enumerate(class_directories):
            image_paths = sorted(
                path
                for path in class_directory.iterdir()
                if path.is_file() and path.suffix.lower() in self.allowed_extensions
            )

            if not image_paths:
                continue

            class_name = class_directory.name.removesuffix("_images")
            self.class_to_idx[class_name] = class_index
            self.idx_to_class.append(class_name)

            for image_path in image_paths:
                self.samples.append((image_path, class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            converted_image = image.convert("RGB")
            if self.transform is not None:
                converted_image = self.transform(converted_image)
            else:
                converted_image = transforms.ToTensor()(converted_image)

        return converted_image, label


# class SimpleBirdCNN(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 128),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         return self.classifier(self.features(inputs))



class BirdCNN(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True) -> None:
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def train_model() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "augmented_results"
    train_dir = source_dir / "train"
    stats_dir = source_dir / "stats"
    checkpoint_path = project_root / "bird_cnn.pth"

    if not source_dir.exists():
        print(f"Dataset directory not found: {source_dir}")
        return

    species_dirs = sorted(
        path for path in train_dir.iterdir()
        if path.is_dir() and path.name.endswith("_images")
    )
    if not species_dirs:
        print(f"No species folders found in {train_dir}")
        return

    train_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # split_summary = split_dataset(species_dirs, train_dir, stats_dir, seed=42)
    # print("Dataset split completed (90% train / 10% stats):")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Augmentation forte côté train pour compenser le peu de données
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    # Pas d'augmentation côté validation, juste la normalisation
    stats_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = BirdImageDataset(train_dir, transform=train_transforms)
    stats_dataset = BirdImageDataset(stats_dir, transform=stats_transforms)

    if len(train_dataset) == 0 or len(stats_dataset) == 0:
        print("The split did not produce enough images to train and evaluate the model.")
        return

    num_classes = len(train_dataset.idx_to_class)
    model = BirdCNN(num_classes=num_classes, freeze_backbone=True).to(device)

    num_classes = len(train_dataset.idx_to_class)
    model = BirdCNN(num_classes=num_classes, freeze_backbone=True).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,        # ← charge les images en parallèle (essaie 4-8)
        pin_memory=True,      # ← accélère le transfert CPU→GPU
        persistent_workers=True,  # ← évite de recréer les workers à chaque epoch
    )
    stats_loader = DataLoader(stats_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ---------- PHASE 1 : entraîner uniquement la tête ----------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    best_stats_loss = float("inf")
    epochs_without_improvement = 0
    patience = 10

    scaler = torch.amp.GradScaler('cuda')

    print("=== Phase 1 : entraînement de la tête (backbone gelé) ===")
    for epoch in range(1, 15):
        train_loss, train_accuracy = run_training_epoch(model, train_loader, criterion, optimizer, device, scaler)
        stats_loss, stats_accuracy = evaluate(model, stats_loader, criterion, device)
        print(
            f"[Phase 1] Epoch {epoch} - train loss: {train_loss:.4f} - train acc: {train_accuracy:.2%} - stats loss: {stats_loss:.4f} - stats acc: {stats_accuracy:.2%}"
        )

        if stats_loss < best_stats_loss:
            best_stats_loss = stats_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Phase 1 terminée (plateau atteint)")
                break

    # ---------- PHASE 2 : dégeler les dernières couches ----------
    print("=== Phase 2 : fine-tuning des dernières couches du backbone ===")

    # Dégeler seulement les 3 derniers blocs de MobileNetV2 (sur ~19)
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # lr très faible sur le backbone dégelé, lr normal sur la tête
    optimizer = torch.optim.Adam([
        {"params": model.features[-3:].parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ])

    best_stats_loss = float("inf")
    epochs_without_improvement = 0
    patience = 15

    for epoch in range(1, 30):
        train_loss, train_accuracy = run_training_epoch(model, train_loader, criterion, optimizer, device, scaler)
        stats_loss, stats_accuracy = evaluate(model, stats_loader, criterion, device)
        print(
            f"[Phase 2] Epoch {epoch} - train loss: {train_loss:.4f} - train acc: {train_accuracy:.2%} - stats loss: {stats_loss:.4f} - stats acc: {stats_accuracy:.2%}"
        )

        if stats_loss < best_stats_loss:
            best_stats_loss = stats_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": train_dataset.idx_to_class},
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping à l'epoch {epoch}")
                break

    print(f"Model saved to {checkpoint_path}")




def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, float]:

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_items = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Loss
        running_loss += loss.item() * images.size(0)

        # Accuracy
        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_items += labels.size(0)

    average_loss = running_loss / max(1, total_items)

    accuracy = correct_predictions / max(1, total_items)

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_items = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_items += labels.size(0)

    average_loss = running_loss / max(1, total_items)
    accuracy = correct_predictions / max(1, total_items)

    return average_loss, accuracy