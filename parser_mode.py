import tensorflow as tf
import numpy as np
import feature_extraction as fe
class parser():
    def model(self,train):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(16,input_shape=(1000,48),activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        x = []
        y_pred = []
        aa=[]
        for i in range(1000):
            aa.extend(train[0][0][i])
            aa.extend(train[0][1][i])
            aa.extend(train[0][2][i])
            x.append(aa)
            y_pred.append(train[1][i])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc']
                      )
        np.array(y_pred,dtype=object)
        history = model.fit(x, y_pred, epochs=1, batch_size=32)
        print(history)


