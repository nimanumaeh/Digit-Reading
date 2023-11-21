import numpy as np
import q4  # Assuming q4.py contains the necessary functions and trained model parameters


def predict_digit(image_array):
    """
    Predict the digit from the preprocessed image array.
    """
    # Load the trained model parameters (means and covariances)
    means, covariances = q4.load_trained_model()  # You need to implement this function in q4.py

    # Predict the digit
    digit_prediction = q4.classify_data(np.array([image_array]), means, covariances)

    return digit_prediction[0]
