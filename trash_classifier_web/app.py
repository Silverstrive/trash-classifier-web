import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from datetime import datetime
from tensorflow.keras.preprocessing import image #type:ignore

@st.cache_resource
def load_trash_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "trash_classifier1.5.h5")
    return tf.keras.models.load_model(model_path)

# Load model
model = load_trash_model()

# Kelas
CLASS_NAMES = ['Cardboard', 'Clothes', 'Glass', 'Metal', 'Paper', 'Plastic', 'Shoes', 'Tidak_diketahui']
CLASS_DESCRIPTIONS = {
    'Cardboard': 'Kardus dan packaging.',
    'Clothes': 'Pakaian bekas dan tekstil.',
    'Glass': 'Botol dan stoples gelas.',
    'Metal': 'Logam dan besi.',
    'Paper': 'Produk kertas seperti koran dan buku.',
    'Plastic': 'Botol, wadah, dan packaging plastik.',
    'Shoes': 'Sepatu bekas dan footwear.',
    'Tidak_diketahui': 'Sampah buangan yang tidak masuk ke kategori lain.'
}

# Judul halaman
st.title("Trash Classifier Image")

# Form upload
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    result = model.predict(img_array)
    prediction = CLASS_NAMES[np.argmax(result)]
    confidence = f"{np.max(result) * 100:.2f}%"
    probabilities = [(CLASS_NAMES[i], float(result[0][i])) for i in range(len(CLASS_NAMES))]

    upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence}")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # === TABEL PROBABILITAS DENGAN HIGHLIGHT ===
    st.subheader("Class Probabilities")
    highlight_style = "background-color: #ffe599; font-weight: bold; color: black;"  # kuning lembut
    prob_table = "<table style='margin: 0 auto; border-collapse: collapse;'>"
    prob_table += "<tr><th style='padding: 6px 14px;'>Class</th><th style='padding: 6px 14px;'>Probability</th><th style='padding: 6px 14px;'>Description</th></tr>"

    for CLASS_NAMES, prob in probabilities:
        row_style = highlight_style if CLASS_NAMES == prediction else ""
        prob_table += f"<tr style='{row_style}'><td style='padding: 6px 14px;'>{CLASS_NAMES}</td><td style='padding: 6px 14px;'>{prob*100:.2f}%</td><td style='padding: 6px 14px;'>{CLASS_DESCRIPTIONS[CLASS_NAMES]}</td></tr>"

    prob_table += "</table>"
    st.markdown(prob_table, unsafe_allow_html=True)

    st.write(f"**Upload Time:** {upload_time}")

    # Reset/Clear
    if st.button("Reset/Clear"):
        st.experimental_rerun()
