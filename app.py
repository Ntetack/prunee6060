from flask import Flask, render_template, request
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from models.cnn import CNN1

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


# =========================
# LOADERS (LAZY LOADING)
# =========================

def load_torch_model():
    model = CNN1()
    model.load_state_dict(
        torch.load("yourname_model.pth", map_location="cpu")
    )
    model.eval()
    return model


def load_tf_model():
    return tf.keras.models.load_model("yourname_model.keras")


# =========================
# PREPROCESS TF
# =========================

def preprocess_tf(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# =========================
# ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        model_choice = request.form["model"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")

            # =========================
            # PYTORCH (LOAD ONLY IF USED)
            # =========================
            if model_choice == "torch":
                model = load_torch_model()

                transform = torch.nn.Sequential()  # placeholder (on simplifie)
                input_tensor = torch.tensor(
                    np.array(image.resize((150,150))).transpose