import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
X_train=X_train/255
#X_train = np.array(X_train).T
Y_train = (np.arange(10)==y_train[:, None]).astype(int)
#Y_train = np.array(Y_train).T
  
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
X_test=X_test/255
#X_test = np.array(X_test).T
Y_test = (np.arange(10) == y_test[:, None]).astype(int)
#Y_test = np.array(Y_test).T
    


# stack together for next step
X = np.vstack((x_train, x_test))
y = np.vstack((y_train, y_test))


# one-hot encoding
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)


# number of training set
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]


# shuffle training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]