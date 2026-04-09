import os
import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from model import build_model

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# --- Global Model Loading (Optimized) ---
device = "cpu"
model = None
classes = []

def load_all():
    global model, classes
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        classes = checkpoint.get("classes", ["cat", "dog"])
        model = build_model(num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Model loaded with classes: {classes}")

load_all()

def url_for(name: str, **kwargs) -> str:
    if name == "static":
        return f"/static/{kwargs.get('path', '')}"
    return "#"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    tmpl = env.get_template("index.html")
    return tmpl.render(request=request, result=None, url_for=url_for, classes=classes)

@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()

    if model is None:
        result = ("Model not loaded", 0.0)
    else:
        # Fast Inference
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            result = (classes[pred.item()], float(conf.item()))

    tmpl = env.get_template("index.html")
    return tmpl.render(request=request, result=result, url_for=url_for, classes=classes)
