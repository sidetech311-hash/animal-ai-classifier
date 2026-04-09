# 🐾 Animal Intelligence AI Classifier

A deep learning-based web application that identifies animal species from uploaded images. Powered by **FastAPI** and **PyTorch (ResNet18)**.

## 🚀 Features
- **Instant Identification**: Upload an image and get predictions in milliseconds.
- **Dynamic Learning**: Supports any number of animal classes based on your training data.
- **Modern UI**: Clean, responsive interface with drag-and-drop support and visual feedback.
- **Confidence Scoring**: High-accuracy predictions with visual indicators for low-confidence results.

---

## 🛠️ Setup & Installation

### 1. Clone the Project
```bash
git clone <your-repo-url>
cd FastAPI_Pet_Classifier
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\Activate.ps1
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training for Other Animals
You can easily extend this model to identify **any** animal.

1.  **Prepare Data**: 
    - Create a folder for the animal in `data/train/` (e.g., `data/train/elephant`).
    - Create the same folder in `data/val/`.
    - Add images of that animal to both folders.
2.  **Run Training**:
    ```bash
    python train.py --data_dir ./data --epochs 10
    ```
3.  The model will automatically detect the new folders and update itself!

---

## 🖥️ Running the App
Start the web server:
```bash
python -m uvicorn app:app --reload
```
Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 📦 Deployment
This app is ready for deployment on **Hugging Face Spaces** or **Render**.
- Ensure `models/model.pth` is included in your upload.
- The app uses CPU-only inference for maximum compatibility with free hosting tiers.

---

## 👥 Credits
Developed by **Group 6 Deep Learning**.
Powered by PyTorch ResNet18.
