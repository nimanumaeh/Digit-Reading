'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(10):
        means[i] = np.mean(train_data[train_labels == i], axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    for i in range(10):
        data_i = train_data[train_labels == i]
        mean_i = np.mean(data_i, axis=0)
        covariance_i = np.dot((data_i - mean_i).T, (data_i - mean_i)) / data_i.shape[0]
        covariances[i] = covariance_i + 0.01 * np.eye(64)  # Adding 0.01I for stability
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = digits.shape[0]
    generative_log_likelihood = np.zeros((n, 10))
    for i in range(10):
        diff = digits - means[i]
        inv_covariance = np.linalg.inv(covariances[i])
        det_covariance = np.linalg.det(covariances[i])
        for j in range(n):
            generative_log_likelihood[j, i] = -0.5 * (np.log(det_covariance) +
                                                      np.dot(diff[j], np.dot(inv_covariance, diff[j].T)) +
                                                      64 * np.log(2 * np.pi))
    return generative_log_likelihood

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    generative_log_likelihood = generative_likelihood(digits, means, covariances)
    log_prior = np.log(0.1)  # Since p(y=k) = 1/10
    conditional_log_likelihood = generative_log_likelihood + log_prior
    for i in range(conditional_log_likelihood.shape[0]):
        conditional_log_likelihood[i] -= np.logaddexp.reduce(conditional_log_likelihood[i])
    return conditional_log_likelihood

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    avg_likelihood = 0
    for i in range(digits.shape[0]):
        avg_likelihood += cond_likelihood[i, int(labels[i])]
    return avg_likelihood / digits.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation

if __name__ == '__main__':
    main()
