import streamlit as st
import numpy as np
import pickle
import os
import urllib.request

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️"
)

st.title("🖼️ Image Captioning")
st.write("Upload an image and generate a caption using VGG16 + LSTM.")


# -----------------------------
# Model download
# -----------------------------
MODEL_URL = "https://github.com/pragatiy9082-ram/Image-Captioning/releases/download/v1.0/image_captioning_model.1.h5"
MODEL_PATH = "image_captioning_model.1.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# -----------------------------
# Load model and tokenizer
# -----------------------------
@st.cache_resource
def load_resources():

    model = load_model(MODEL_PATH)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vgg_model = VGG16(
        weights="imagenet",
        include_top=False
    )

    return model, tokenizer, vgg_model


model, tokenizer, vgg_model = load_resources()

max_length = 10


# -----------------------------
# Generate caption
# -----------------------------
def generate_caption(photo):

    in_text = "startseq"

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = np.pad(
            sequence,
            (0, max_length - len(sequence)),
            mode="constant"
        )

        yhat = model.predict(
            [photo, sequence.reshape(1, -1)],
            verbose=0
        )

        yhat = np.argmax(yhat[0])

        word = None

        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        if word == "endseq":
            break

        in_text += " " + word

    caption = in_text.replace("startseq", "").strip()

    return caption


# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Generate Caption"):

        with st.spinner("Generating caption..."):

            image_resized = image.resize((224, 224))

            image_array = img_to_array(image_resized)

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = preprocess_input(image_array)

            features = vgg_model.predict(
                image_array,
                verbose=0
            )

            features = features.reshape(
                features.shape[0],
                -1
            )

            caption = generate_caption(features)

        st.success("Caption Generated!")
        st.subheader("Generated Caption")
        st.write(caption)
