from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os

from models.cnn import CNN1

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# classes Intel dataset
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# DEVICE = CPU ONLY (PythonAnywhere limitation)
device = torch.device("cpu")

# charger modèle
model = CNN1()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# transformation (SANS augmentation)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                _, pred = torch.max(output, 1)
                prediction = CLASSES[pred.item()]

            image_url = filepath

    return render_template("index.html",
                           prediction=prediction,
                           image_url=image_url)

if __name__ == "__main__":
    app.run()