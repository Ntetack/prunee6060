from flask import Flask, render_template, request
import torch
from torchvision import transforms
import tensorflow as tf
from PIL import Image
import numpy as np
import os

from models.cnn import CNN1

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# =========================
# LOAD MODELS
# =========================

device = torch.device("cpu")

# PyTorch model
torch_model = CNN1()
torch_model.load_state_dict(torch.load("model.pth", map_location=device))
torch_model.to(device)
torch_model.eval()

# TensorFlow model
tf_model = tf.keras.models.load_model("lieumo_model.keras")

# =========================
# TRANSFORMS
# =========================

torch_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def preprocess_tf(image):
    image = image.resize((150,150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# ROUTE
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        model_choice = request.form['model']

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")

            # ===== PYTORCH =====
            if model_choice == "torch":
                input_tensor = torch_transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = torch_model(input_tensor)
                    _, pred = torch.max(output, 1)
                    prediction = CLASSES[pred.item()]

            # ===== TENSORFLOW =====
            elif model_choice == "tf":
                input_tensor = preprocess_tf(image)

                output = tf_model.predict(input_tensor)
                pred = np.argmax(output, axis=1)[0]
                prediction = CLASSES[pred]

            image_url = filepath

    return render_template("index.html",
                           prediction=prediction,
                           image_url=image_url)


if __name__ == "__main__":
    app.run()