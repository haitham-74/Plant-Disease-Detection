import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from req import RECOMMENDATIONS
# -----------------------
# Load Model
# -----------------------
MODEL_PATH = r"C:\Users\Haitham\Downloads\شهادات\PlantDisease\plant_disease_model.keras"
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

advice_list = []
for disease in CLASS_NAMES:
    advice_list.append(RECOMMENDATIONS.get(disease, "ℹ️ لا توجد نصائح متاحة لهذا المرض."))

# -----------------------
# Streamlit UI
# -----------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image and the model will predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image ✅", use_container_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(96, 96))  # must match training size
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100

    class_name = CLASS_NAMES[class_idx]

    # Split plant and disease
    if "___" in class_name:
        plant, disease = class_name.split("___")
    else:
        plant, disease = class_name, "Unknown"

    # Display result with colors
    if "healthy" in disease.lower():
        st.success(f"🌿 Plant: **{plant}** | Status: **Healthy** | Confidence: {confidence:.2f}%")
    else:
        st.error(f"❌ Plant: **{plant}** | Disease: **{disease}** | Confidence: {confidence:.2f}%")
