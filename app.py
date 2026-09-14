
import gradio as gr
import numpy as np
import pickle

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


# Load trained LSTM model
model = load_model("image_captioning_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load VGG16
vgg_model = VGG16(weights="imagenet", include_top=False)

max_length = 10


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

        in_text += " " + word

        if word == "endseq":
            break

    return in_text.replace("startseq ", "").replace(" endseq", "")


def caption_image(image):

    image = image.resize((224, 224))
    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    features = vgg_model.predict(image_array, verbose=0)
    features = features.reshape(1, -1)

    caption = generate_caption(features)

    return caption


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Captioning using VGG16 + LSTM",
    description="Upload an image and generate an automatic caption."
)

demo.launch(share=True)
