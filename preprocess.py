# preprocess.py

import numpy as np
from PIL import Image


def preprocess_image(image_path, image_size=(8, 8)):
    """
    Preprocess the image to fit the model input requirements.
    """
    # Open the image
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert('L')

    # Resize to 8x8
    img = img.resize(image_size, Image.Resampling.LANCZOS)
    # Convert to numpy array and normalize
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    # Flatten the array
    img_array = img_array.flatten()

    return img_array
