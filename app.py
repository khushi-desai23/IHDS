import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

# Load the model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=50)  # Ensure this matches your checkpoint's architecture
    
    # Load the model weights (checkpoint)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter the state_dict to only load matching layers
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

    # Update the model's state_dict with the pretrained weights for matching layers
    model_state_dict.update(pretrained_dict)
    
    # Load the filtered state dict into the model
    model.load_state_dict(model_state_dict, strict=False)  # strict=False allows for some layers to be ignored if they don't match in size
    
    model.to(device)
    model.eval()
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict the class
def predict(model, image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)  # Ensure tensor is on the same device as the model
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Streamlit app
st.title("Heritage Site Classifier")

uploaded_image = st.file_uploader("Upload an image of a heritage site", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        # Open and convert the image to RGB format
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()  # Stop execution if image loading fails

    st.write("Processing the image...")

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Load the model
    model_path = "FINAL.pth"  # Replace with the path to your .pth file
    model = load_model(model_path)
    
    if model is None:
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
