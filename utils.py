from PIL import Image


def open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
