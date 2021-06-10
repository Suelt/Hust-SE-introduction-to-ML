from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
np.random.seed(10)
(x_Train,y_Train),(x_Test,y_Test)=mnist.load_data()


X_Train=x_Train.reshape(x_Train.shape[0],28,28,1).astype('int')
X_Test=x_Test.reshape(x_Test.shape[0],28,28,1).astype('int')

Y_Train = (np.arange(10)==y_Train[:, None]).astype(int)
Y_Test = (np.arange(10) == y_Test[:, None]).astype(int)

model=Sequential()
model.add(Conv2D(filters=6,
                 kernel_size=(3,3),
                 input_shape=(28,28,1),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=X_Train,
                        y=Y_Train,validation_split=0.3,
                        epochs=10,batch_size=128,verbose=2)

scores=model.evaluate(X_Test,Y_Test)

print()
print('accuracy',scores[1])
