from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import timm
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load your model
model = torch.jit.load("C:\\Users\\Dell\\Desktop\\ML_CP\\mobilenetv4_scripted.pt")
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# If you need to get class labels, load the ImageFolder dataset
data_dir = "C:/Users/Dell/Desktop/ML_CP/DATASET/IHDS_dataset"
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=preprocess)
class_labels = train_dataset.classes  # Extract class labels from the dataset

@app.route('/')
def home():
    return render_template("index.html")  # Make sure index.html is inside templates folder

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    # Open and preprocess the image
    image = Image.open(file.stream).convert('RGB')
    image = preprocess(image).unsqueeze(0)  # Apply preprocessing and add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_labels[predicted.item()]

    return f"Predicted Class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
