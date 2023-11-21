# Digit Recognition with Naive Bayes and Conditional Gaussian Classifiers

## Project Overview
This project implements Naive Bayes and Conditional Gaussian classifiers to recognize handwritten digits. The dataset used is the MNIST dataset, a large database of handwritten digits that is commonly used for training various image processing systems.

## Modules
- `naive_bayes.py`: Implements the Naive Bayes classifier for digit recognition. It includes functions for downloading the MNIST dataset, preprocessing the data, training Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimators, and evaluating the model's performance.
- `q4.py`: Implements the Conditional Gaussian classifier. It includes functions for computing mean and covariance estimates, generative and conditional likelihoods, and classifying new data points.
- `data.py`: A utility module for loading and preprocessing the digit dataset.

## Key Concepts
- Naive Bayes Classification
- Conditional Gaussian Models
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) Estimation
- Data Preprocessing and Visualization

## How to Run
Ensure you have Python installed with necessary libraries: NumPy, Matplotlib, and gzip. Run each module using Python to see the results of the classifiers.
