import io
from typing import Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import build_model

def load_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def predict(image_bytes: bytes, model_path: str, device: str = None) -> Tuple[str, float]:
    device = "cpu"
    img = load_image(image_bytes)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tensor = transform(img).unsqueeze(0).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", ["cat", "dog"])
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = classes[pred.item()]
        return label, float(confidence.item())
