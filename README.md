# 🖼️ Image Captioning Mini Project

## Overview
This is a Python + Streamlit image-captioning mini project built for the supplied **500-image synthetic educational/demo dataset**.

The dataset contains 500 generated JPG images and one matching caption for each image. The project uses **computer-vision image retrieval**: the uploaded image is resized into a compact pixel feature, compared with the 500 reference images, and the caption belonging to the closest visual match is displayed.

Example:

**Image:** blue flower in a yard

**Caption:** `A blue flower is shown in a yard.`

## Important dataset note
The supplied README states that this is a **synthetic starter dataset** and **not Flickr8k**. Do not describe it as Flickr8k in a college report.

## Files
- `app.py` – Streamlit application.
- `captions_500.csv` – 500 image/caption pairs.
- `images/` – 500 JPG reference images.
- `requirements.txt` – Python packages.
- `train.py` – optional CNN training experiment; the Streamlit app does not require it.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. User uploads an image.
2. The image is resized to a standard feature representation.
3. The feature is compared with all 500 reference images.
4. The closest visual match is selected.
5. Its verified caption from `captions_500.csv` is displayed.

## Scope
This approach is designed for the supplied synthetic dataset and images from the same visual domain. It is not a general-purpose captioning model for arbitrary real-world photographs. For a general image-captioning system, a larger natural-image dataset and a trained vision-language model would be required.
