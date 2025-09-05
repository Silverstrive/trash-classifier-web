import os
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
from datetime import datetime
from tensorflow.keras.preprocessing import image
from PIL import Image

# === Konfigurasi path model ===
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trash_classifier1.6.h5")
HF_URL = "https://huggingface.co/Silverstrive/trash-classifier/resolve/main/trash_classifier1.6.h5"

# === Download Model dari HuggingFace ===
def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        placeholder = st.empty()
        placeholder.info("📥 Downloading model from HuggingFace... Please wait")

        try:
            r = requests.get(HF_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            placeholder.empty()
            st.success("✅ Model berhasil diunduh dari HuggingFace!")
        except Exception as e:
            placeholder.error(f"❌ Gagal mengunduh model: {e}")
            return False
    return True

# === Load Model ===
@st.cache_resource
def load_model():
    if not download_model():
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# === Kelas klasifikasi ===
CLASS_NAMES = ["cardboard", "clothes", "glass", "paper", "plastic", "shoes", "tidak_diketahui"]

CLASS_DESCRIPTIONS = {
    'cardboard': 'Kardus dan packaging.',
    'clothes': 'Pakaian bekas dan tekstil.',
    'glass': 'Botol dan stoples gelas.',
    'paper': 'Produk kertas seperti koran dan buku.',
    'plastic': 'Botol, wadah, dan packaging plastik.',
    'shoes': 'Sepatu bekas dan footwear.',
    'tidak_diketahui': 'Sampah buangan yang tidak masuk ke kategori lain.'
}

# === Judul Halaman ===
st.title("🗑️ Trash Classifier Image")

# === Upload Form ===
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Model tidak tersedia. Tidak dapat melakukan prediksi.")
    else:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            result = model.predict(img_array)
            prediction = CLASS_NAMES[np.argmax(result)]
            confidence = f"{np.max(result) * 100:.2f}%"
            probabilities = [(CLASS_NAMES[i], float(result[0][i])) for i in range(len(CLASS_NAMES))]

            upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # === Hasil Prediksi ===
            st.subheader("Prediction Result")
            st.write(f"**Prediction:** {prediction}")
            st.write(f"**Confidence:** {confidence}")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.image(img, caption="Uploaded Image", width=300)

            # === Tabel Probabilitas ===
            st.subheader("Class Probabilities")
            highlight_style = "background-color: #ffe599; font-weight: bold; color: black;"
            prob_table = "<table style='margin: 0 auto; border-collapse: collapse;'>"
            prob_table += "<tr><th style='padding: 6px 14px;'>Class</th><th style='padding: 6px 14px;'>Probability</th><th style='padding: 6px 14px;'>Description</th></tr>"

            for cls_name, prob in probabilities:
                row_style = highlight_style if cls_name == prediction else ""
                prob_table += f"<tr style='{row_style}'><td style='padding: 6px 14px;'>{cls_name}</td><td style='padding: 6px 14px;'>{prob*100:.2f}%</td><td style='padding: 6px 14px;'>{CLASS_DESCRIPTIONS.get(cls_name, 'N/A')}</td></tr>"

            prob_table += "</table>"
            st.markdown(prob_table, unsafe_allow_html=True)

            st.write(f"**Upload Time:** {upload_time}")

            # Reset
            if st.button("Reset/Clear"):
                st.session_state.clear()
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")

