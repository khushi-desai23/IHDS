import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models

# Streamlit app title
st.title("Heritage Site Classifier and Image Resizer")

# Instructions
st.write("Upload or take a picture, and it will classify the heritage site and resize the image to 224x224 pixels.")

# Load the PyTorch model
def load_model(model_path, num_classes):
    # Initialize the model architecture (replace with your specific model if needed)
    model = models.resnet18(pretrained=False)  # Example using ResNet18
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust for your dataset

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    model.eval()
    return model

# Define transformation for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load class labels from txt file
def load_class_labels(label_path):
    with open(label_path, 'r') as f:
        class_labels = {str(i): line.strip() for i, line in enumerate(f)}
    return class_labels

# Paths to model and class labels
MODEL_PATH = "today_final.pth"  # Replace with your .pth file path
LABELS_PATH = "labels.txt"  # Replace with your label file path

# Load the class labels
class_labels = load_class_labels(LABELS_PATH)
num_classes = len(class_labels)

# Load the model
model = load_model(MODEL_PATH, num_classes)

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)

    # Display the original image
    st.subheader("Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize the image to 224x224
    resized_image = image.resize((224, 224))

    # Display the resized image
    st.subheader("Resized Image (224x224)")
    st.image(resized_image, caption="Resized Image", use_column_width=True)

    # Classify the image
    st.subheader("Classification Result")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_name = class_labels[str(predicted.item())]

    st.write(f"Predicted Heritage Site: **{class_name}**")

    # Option to download the resized image
    st.download_button(
        label="Download Resized Image",
        data=resized_image.tobytes(),
        file_name="resized_image.png",
        mime="image/png"
    )
else:
    st.write("Please upload an image to proceed.")
