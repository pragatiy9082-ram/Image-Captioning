import os
import csv
import numpy as np
import streamlit as st
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE, "images")
CSV_FILE = os.path.join(BASE, "captions_500.csv")

st.set_page_config(page_title="Image Captioning", page_icon="🖼️", layout="centered")
st.title("🖼️ Image Captioning")
st.write("Upload an image and generate its caption using Python + Computer Vision + Streamlit.")

@st.cache_data

def load_dataset():
    data = {}
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[row["image"]] = row["caption"].strip()
    return data

@st.cache_data

def build_gallery(image_names):
    features = []
    valid_names = []
    for name in image_names:
        path = os.path.join(IMAGE_DIR, name)
        try:
            im = Image.open(path).convert("RGB").resize((64, 48))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            features.append(arr.reshape(-1))
            valid_names.append(name)
        except Exception:
            pass
    return np.stack(features), valid_names

def extract_feature(image):
    im = image.convert("RGB").resize((64, 48))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)

def generate_caption(image):
    captions = load_dataset()
    gallery, names = build_gallery(list(captions.keys()))
    query = extract_feature(image)

    # Mean squared image difference. The supplied dataset has a fixed
    # synthetic visual style, so nearest-image retrieval is highly reliable.
    distances = np.mean((gallery - query) ** 2, axis=1)
    best_index = int(np.argmin(distances))
    best_name = names[best_index]
    best_distance = float(distances[best_index])

    # Convert distance to a simple similarity indicator for the UI.
    similarity = max(0.0, 1.0 - best_distance * 6.0)
    similarity_percent = similarity * 100.0

    return captions[best_name], best_name, similarity_percent

if not os.path.exists(CSV_FILE) or not os.path.isdir(IMAGE_DIR):
    st.error("Dataset files are missing. Keep captions_500.csv and the images folder beside app.py.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption", type="primary"):
        with st.spinner("Analyzing image and generating caption..."):
            caption, matched_image, similarity = generate_caption(image)

        st.success("Caption Generated!")
        st.markdown(f"### Generated Caption: {caption}")
        st.caption(f"Matched dataset image: {matched_image}  •  Visual similarity: {similarity:.1f}%")

st.divider()
st.caption("Educational mini project based on the supplied 500-image synthetic dataset.")
