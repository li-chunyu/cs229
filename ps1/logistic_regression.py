# Created by lichunyu
# Solution for cs229 ps1 Logistic_regression

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    s = 1 / (1+np.exp(-x))
    return s

def read_data(x_path, y_path):
    X = np.loadtxt(x_path, dtype=float)
    Y = np.loadtxt(y_path, dtype=float)
    return X.T, Y.reshape((99, 1)).T

def initialize_parameter(X_train):
    theta = np.zeros((X_train.shape[0], 1))
    return theta

# X_train is of shape (3, m)
# Y_train is of shape (1, 99)
def fit(X_train, Y_train):
    X_train = np.vstack((X_train, np.ones((1, X_train.shape[1]), dtype=float)))
    theta = initialize_parameter(X_train)
    m = X_train.shape[1]
    # multiply by elemen
    # print(theta.shape)
    # z = np.dot(theta.T, X_train) * Y_train
    # assert(z.shape == (1, m))
    iter_var = 1e9
    iter_num = 0
    thetas = []
    while iter_var > 1e-6:
        thetas.append(theta)
        z = np.dot(theta.T, X_train) * Y_train
        H = np.zeros((X_train.shape[0], X_train.shape[0]))
        assert(H.shape == (X_train.shape[0], X_train.shape[0]))
        for i in range(H.shape[0]):
            for j in range(H.shape[0]):
                if i <= j:
                    g = sigmoid(z)
                    H[i, j] = np.sum(g * (1 - g) * X_train[i, :] * X_train[j, :]) / m
                else:
                    H[i, j] = H[j, i]
        tehta_cache = theta.copy()
        dtheta = - np.sum((1 - sigmoid(z)) * Y_train * X_train, axis=1, keepdims=True) / m
        theta = theta - np.dot(np.linalg.inv(H), dtheta)
        iter_var = np.sum(np.abs(theta - tehta_cache))
        iter_num += 1
    print("Converged after " + str(iter_num) + " times iteration.")
    return theta, thetas

if __name__ == "__main__":
    X_train, Y_train = read_data("data/logistic_x.txt", "data/logistic_y.txt")
    theta, thetas = fit(X_train, Y_train)
    print(theta.shape)
    colors = ['red' if i == 1.0 else 'blue' for i in Y_train[0,:]]
    plt.scatter(X_train[1, :], X_train[0, :], c=colors)
    x = np.array([np.min(X_train[1, :]), np.max(X_train[1, :])])
    for k, t in enumerate(thetas):
        t.squeeze()
        if (np.sum(np.abs(t))) == 0:
            continue
        plt.plot(x, (t[0] * x + t[2]) / (-t[0]), label='iter {0}'.format(k+1), lw=0.5)
    plt.show()