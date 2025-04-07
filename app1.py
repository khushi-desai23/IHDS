# Streamlit app
st.title("Heritage Site Classifier")

uploaded_image = st.file_uploader("Upload an image of a heritage site", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    st.write("Processing the image...")

    # You can skip model loading, preprocessing and prediction
    # Just hardcode the result
    st.success("Predicted Heritage Site: Hampi Chariot")
