from keras.datasets import mnist
from keras.utils import np_utils
from plot_image_1 import plot_image_1
from plot_prediction_1 import plot_image_labels_prediction_1
from show_train_history import show_train_history
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
np.random.seed(10)
(x_Train,y_Train),(x_Test,y_Test)=mnist.load_data()
print('train data=',len(x_Train))
print('test data=',len(x_Test))
print('x_train_image:',x_Train.shape)
print('y_train_label:',y_Train.shape)
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
x_Train4D_normalize=x_Train4D/255
x_Test4D_normalize=x_Test4D/255
y_TrainOneHot=np_utils.to_categorical(y_Train)
y_TestOneHot=np_utils.to_categorical(y_Test)
model=Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
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
train_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainOneHot,validation_split=0.2,
                        epochs=5,batch_size=300,verbose=2)
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
scores=model.evaluate(x_Test4D_normalize,y_TestOneHot)
print()
print('accuracy',scores[1])
prediction=model.predict_classes(x_Test4D_normalize)
print("prediction[:10]",prediction[:10])
plot_image_labels_prediction_1(x_Test,y_Test,prediction,idx=0)
pd.crosstab(y_Test,prediction,rownames=['label'],colnames=['predict'])