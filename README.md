# Digit Recognition with Naive Bayes and Conditional Gaussian Classifiers

## Project Overview
This project implements Naive Bayes and Conditional Gaussian classifiers to recognize handwritten digits. The dataset used is the MNIST dataset, a large database of handwritten digits that is commonly used for training various image processing systems.

## Modules
- `naive_bayes.py`: Implements the Naive Bayes classifier for digit recognition. It includes functions for downloading the MNIST dataset, preprocessing the data, training Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimators, and evaluating the model's performance.
- `q4.py`: Implements the Conditional Gaussian classifier. It includes functions for computing mean and covariance estimates, generative and conditional likelihoods, and classifying new data points.
- `data.py`: A utility module for loading and preprocessing the digit dataset.
- 'predict.py': A module containing the prediction function when uploading an image.
- 'GUI.py': The GUI module for interacting with the model and uploading an image. If no model_parameters.pkl, the GUI will automatically train the model. 
- 'preprocess.py': A module containing the preprocessing function for the images.
- 'model_parameters.pkl': Saved model features to avoid re-running the training. You may delete this and re-run the training on another data set. 

## Key Concepts
- Naive Bayes Classification
- Conditional Gaussian Models
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) Estimation
- Data Preprocessing and Visualization

## How to Run
Ensure you have Python installed with necessary libraries: NumPy, Matplotlib, and gzip. Run the GUI.
