from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD
from  keras.layers import Dense, Dropout, Activation
import numpy as np



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # plt.subplot(331)
    # plt.imshow(X_test[0], cmap=plt.get_cmap('gray'))
    # plt.subplot(332)
    # plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
    # plt.subplot(333)
    # plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
    # plt.subplot(334)
    # plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
    # plt.subplot(335)
    # plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
    # plt.subplot(336)
    # plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))
    # plt.subplot(337)
    # plt.imshow(X_train[6], cmap=plt.get_cmap('gray'))
    # plt.subplot(338)
    # plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))
    # plt.subplot(339)
    # plt.imshow(X_train[8], cmap=plt.get_cmap('gray'))
    # plt.show()

    model = Sequential()

    model.add(Dense(500, input_shape=(784,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(500))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(y_train[0])
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
    Y_train = (np.arange(10)==y_train[:, None]).astype(int)
  
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)

    model.fit(X_train, Y_train, batch_size=128, epochs=50, shuffle=True, verbose=2, validation_split=0.3)

    result = model.predict(X_test, batch_size=128, verbose=1)
    result_max = np.argmax(result, axis=1)
    test_max = np.argmax(Y_test, axis=1)

    result_bool = np.equal(result_max, test_max)
    true_num = np.sum(result_bool)
    print("is %f" % (true_num/len(result_bool)))
