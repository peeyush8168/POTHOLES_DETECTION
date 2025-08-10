import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("POTHOLE DETECTION")

classes = ['bump', 'potholes','road']

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("new_pothole_model.h5")
    return model

model = load_model()

def preprocess(image: Image.Image):
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = img.reshape(1, 150, 150, 3)
    return img

upload_f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if upload_f:
    image = Image.open(upload_f)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Analyze"):
        processed = preprocess(image)
        pred = model.predict(processed)[0]
        class_index = np.argmax(pred)
        pred_class = classes[class_index]
        st.success(f"Prediction: {pred_class}")
