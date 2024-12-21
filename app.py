import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch.nn.functional as F

# Load the model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Initialize model with correct architecture
        model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=50)
    except Exception as e:
        raise RuntimeError(f"Error creating model: {e}")

    # Load the model weights (checkpoint)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Error loading model checkpoint: {e}")

    # Filter the state_dict to only load matching layers
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

    # Update the model's state_dict
    model_state_dict.update(pretrained_dict)

    try:
        model.load_state_dict(model_state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Error loading state dict: {e}")

    model.to(device)
    model.eval()

    # Fine-tuning: Freeze all layers except the classifier
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=50)  # Update for your number of classes
    
    return model

# Preprocess the image with augmentations
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Random rotation for augmentation
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.Resize((224, 224)),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict the class
def predict(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)  # Ensure tensor is on the same device as the model
    
    with torch.no_grad():
        outputs = model(image_tensor)

        # Debug: Print the raw logits before softmax
        st.write(f"Raw model outputs (logits): {outputs}")

        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Debug: Print the probabilities
        st.write(f"Class probabilities: {probabilities}")

        _, predicted = torch.max(outputs, 1)  # Get the predicted class
    return predicted.item()

# Streamlit app
st.title("Heritage Site Classifier")

uploaded_image = st.file_uploader("Upload an image of a heritage site", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        # Open and convert the image to RGB format
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()  # Stop execution if image loading fails

    st.write("Processing the image...")

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Load the model
    model_path = "FINAL_fulll_2.pth"  # Replace with the path to your .pth file
    try:
        model = load_model(model_path)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()  # Stop execution if model loading fails
    
    # Predict
    class_index = predict(model, image_tensor)

    # Map class index to class name
    class_names = [
        "Agra Fort", "Alhole", "Ambigera Gudi Complex, Alhole", "Amruteshwara Temple, Annigeri",
        "Billeshwar Temple Hanagal", "Brahmeshwar Temple, Kikkeri", "Channakeshwa Temple, Aralguppe",
        "Chennakeshwara Temple, Belur", "Digambar Basti, Belgum", "Doddabasappa Temple, Gadag", 
        "Galaganath Temple, Haveri", "Goudaragudi Temple, Aihole", "Hampi Monolithic Bull", 
        "Hampi Chariot", "Hazara Rama Temple, Hampi", "Hoysaleshwar Temple, Halebeedu", "Ibrahim Roza",
        "Jain Basadi, Bilagi", "Kaadasidheshwar Temple, Pattadakal", "Kadambeshwara Temple, Rattihalli, Haveri",
        "Kamal Basti, Belagavi", "Kappechenikeshwara Temple, Hassan", "Kedadeshwara Temple, Hassan",
        "Keshava Temple, Somanathapur, Mysore", "Kiatabeshwar Temple, Kubatur", "Koravangala Temple, Hassan",
        "Kotilingeshwara, Kotipur, Hanagal", "Kumaraswamy Temple, Sandur, Hospet", "Kunti Temple Complex, Aihole",
        "Lady of Mount, Goa", "Lakshmikant Temple, Nanjangudu, Mysore", "Lotus Mahai, Hampi", 
        "Madhukeshwara Temple, Banavasi", "Mahabodhi Temple", "Mahadev Temple, Tambdisurla, Goa",
        "Mahadeva Temple, Ittagi", "Mallikarjuna Temple, Mandya", "Moole Shankareswara Temple, Turuvekere",
        "Nagreshwara Temple, Bankapur", "Papanath Temple, Pattadakal", "Rameshwar Temple", 
        "Safa Masjid, Belgaum", "Sangameshwar Pattadakal", "Shiva Basadi, Shravanbelagola",
        "Someshwar Temple, Kaginele", "Someshwara Temple, Lakshmeshwara", "Tarakeshwara Temple, Hangal",
        "Trikuteshwara Temple, Gadag", "Twin Tower Temple, Sudi", "Veerabhadreshwara Temple, Hangal"
    ]
    
    st.write(f"Predicted Heritage Site: {class_names[class_index]}")
