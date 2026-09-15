
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os
import requests

from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Captioning")
st.write("Upload an image and generate a caption using VGG16 + LSTM.")


# ============================================================
# FILE SETTINGS
# ============================================================

MODEL_FILE = "image_captioning_model.1.h5"
TOKENIZER_FILE = "tokenizer.pkl"

MODEL_URL = (
    "https://github.com/pragatiy9082-ram/Image-Captioning/"
    "releases/download/v1.0/image_captioning_model.1.h5"
)


# ============================================================
# DOWNLOAD MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    with st.spinner("Downloading model..."):

        response = requests.get(
            MODEL_URL,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_resources():

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False
    )

    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    vgg = VGG16(
        weights="imagenet",
        include_top=False
    )

    vgg.trainable = False

    return model, tokenizer, vgg


model, tokenizer, vgg = load_resources()


# ============================================================
# CAPTION GENERATION
# ============================================================

def generate_caption(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = preprocess_input(
        np.expand_dims(image_array, axis=0)
    )

    feature = vgg.predict(
        image_array,
        verbose=0
    ).reshape(1, -1)

    text = "start"
    max_length = 10

    for _ in range(max_length):

        sequence = tokenizer.texts_to_sequences(
            [text]
        )[0]

        sequence = pad_sequences(
            [sequence],
            maxlen=max_length,
            padding="pre"
        )

        prediction = model.predict(
            [feature, sequence],
            verbose=0
        )[0]

        predicted_id = np.argmax(prediction)

        predicted_word = None

        for word, index in tokenizer.word_index.items():

            if index == predicted_id:
                predicted_word = word
                break

        if predicted_word is None:
            break

        if predicted_word == "end":
            break

        text += " " + predicted_word

    return text.replace("start", "").strip()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# DISPLAY IMAGE + GENERATE CAPTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            caption = generate_caption(image)

        st.success("Caption Generated!")

        st.write(
            f"**Generated Caption:** {caption}"
        )
