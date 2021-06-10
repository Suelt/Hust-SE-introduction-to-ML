import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
X_train=X_train/255

Y_train = (np.arange(10)==y_train[:, None]).astype(int)

  
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
X_test=X_test/255

Y_test = (np.arange(10) == y_test[:, None]).astype(int)

    


