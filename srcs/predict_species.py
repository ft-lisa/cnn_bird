from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from train_model import BirdCNN

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "bird_cnn.pth"

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BirdClassifier:
    """Wrapper qui charge le modèle et gère l'inférence."""

    def __init__(self, checkpoint_path: Path | str) -> None:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.idx_to_class = checkpoint["class_names"]

        self.model = BirdCNN(num_classes=len(self.idx_to_class), freeze_backbone=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_path: str) -> tuple[str, float]:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Le fichier '{image_path}' n'existe pas.")

        if path.suffix.lower() not in _VALID_EXTENSIONS:
            raise ValueError(f"'{path.suffix}' n'est pas une extension d'image supportée.")

        try:
            image = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError(f"'{image_path}' n'est pas une image valide ou est corrompue.")

        input_tensor = _transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.max(probabilities, dim=0)

        predicted_class = self.idx_to_class[top_idx.item()]
        confidence = top_prob.item() * 100
        return predicted_class, confidence


def predict_species(image_path: str | None) -> None:
    if not image_path:
        print("Erreur : aucun chemin d'image fourni.")
        return

    try:
        classifier = BirdClassifier(MODEL_PATH)
        predicted_class, confidence = classifier.predict(image_path)
        print(f"Espèce prédite : {predicted_class} ({confidence:.1f}% de confiance)")
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur : {e}")
        return