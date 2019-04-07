import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(path):
    data_fram = pd.read_csv(path)
    cols = data_fram.columns.values.astype(float)
    value = data_fram.values
    return (value, cols)

def optima(X, Y, W=None):
    if W is not None:
        theta = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    else:
        theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    return theta

def construct_weight_matrix(X, x_eval):
    tau =  5
    tmp = -(X[:, 0] - x_eval) ** 2 / (2*(tau**2))
    return np.diag(np.exp(tmp))

def weighted_linear_regression(X, Y):
    hyp = []
    for _, x_eval in enumerate(X[:, 0]):
        W = construct_weight_matrix(X, x_eval)
        assert(W.shape == (450, 450))
        theta = optima(X, Y, W)
        x_eval = np.array([x_eval, 1])
        hyp.append(np.dot(x_eval, theta))
    y_hyp = np.array(hyp)
    return y_hyp

def smooth(X, Y):
    smoothed_Y = weighted_linear_regression(X, Y[:, 0].reshape((Y.shape[0], 1)))
    for col in range(1, Y.shape[1]):
        smoothed_y = weighted_linear_regression(X, Y[:, col].reshape((Y.shape[0], 1)))
        smoothed_Y = np.hstack((smoothed_Y, smoothed_y))
        print("Smoothed percent: {0}%".format(int(col / Y.shape[1] * 100)), end='\r')
    return smoothed_Y

def ker(t):
    return max(1-t, 0)

def f_left_estimate(X_train, Y_train_smoothed):
    # X_train.shape (450, 1), Y_train_smoothed.shape (450, 200)
    k = 3
    fs_left = Y_train_smoothed[0:50, :]
    fs_right = Y_train_smoothed[150:450, :]
    erros = []
    f_left_predict = []
    for i, f_right in enumerate(fs_right.T):
        f_right = f_right.reshape((fs_right.shape[0], 1)) 
        dists = np.sum((fs_right - f_right) ** 2, axis=0, keepdims=True)

        ds = []
        for index, d in enumerate(dists.squeeze()):
            ds.append((d, index))
        ds.sort(key=lambda ele: ele[0])
        h = ds[len(ds) - 1][0]
        ds = ds[1:k+1]
        p_up = 0
        p_down = 0
        for nei in ds:
            p_up += ker(nei[0] / h) * (fs_left[:, nei[1]].reshape((fs_left.shape[0], 1)))
            p_down += ker(nei[0] / h) 
        f_left_hat = p_up / p_down
        f_left = fs_left[:, i].reshape((fs_left.shape[0], 1))
        f_left_predict.append(f_left_hat)
        error = np.sum((f_left_hat - f_left) ** 2)
        erros.append(error)
    print("average error:" + str(np.mean(erros)))
    return f_left_predict

if __name__ == "__main__":
    '''
    flux_train, wav_len_train = read_data('data/quasar_train.csv')
    x = wav_len_train.reshape((450, 1))
    y = flux_train[0].reshape((450, 1))
    x = np.hstack((x, np.ones((450 ,1))))
    theta = optima(x, y)
    y_hyp = np.dot(x, theta)
    assert(y_hyp.shape == (450, 1))
    X_train = x
    Y_train_smoothed = weighted_linear_regression(x, y)
    '''
    flux_train, wav_len_train = read_data('data/quasar_train.csv')
    X_train = wav_len_train.reshape((450, 1))
    X_train = np.hstack((X_train, np.ones((450, 1))))
    # Y_train.shape = (m, 1)
    Y_train = flux_train.T
    Y_train_smoothed = smooth(X_train, Y_train)
    predic = f_left_estimate(X_train, Y_train_smoothed)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.ravel()
    for k, indx in enumerate([0, 6, 10 , 30, 40, 41, 42, 43, 50]):
        ax = axes[k]
        wav_len = X_train[:, 0]
        ax.plot(wav_len, Y_train_smoothed[:, indx], label='smoothed')
        ax.plot(wav_len[:50], predic[indx], label='predict')
        ax.legend()
        ax.set_title('Example {0}'.format(k))
    plt.tight_layout()
    plt.xlim(X_train.min(), X_train.max())
    plt.show()
    '''
    flux_test, wav_len_test = read_data('data/quasar_test.csv')
    X_test = wav_len_test.reshape((450, 1))
    X_test = np.hstack((X_test, np.ones((450, 1))))
    Y_test = flux_test.T
    # Y_test_smoothed = smooth(X_test, Y_test)
    '''


