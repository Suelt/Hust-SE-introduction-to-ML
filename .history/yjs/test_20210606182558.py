import numpy as np
from tensorflow.keras.datasets import mnist
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

size=[784,30,10]
num=2
weight = [np.random.randn(ch2,ch1)
                       for ch1,ch2 in zip(size[:-1], size[1:])]
        # [784,30],[30,10]  z=wxx+b [30,1]
bias = [np.random.rand(s, 1) for s in size[1:]]
# for b, w in zip(bias, weight):
#             # [30,784]@[784,1]->[30,1]+[30,1]=[30,1]
#             z = np.dot(w, x) + b

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_data = []
train_x = train_x.reshape([60000, 784])
for i in range(train_x.shape[0]):
    # print(convert_to_one_hot(train_y[i],10).shape)
    train_data.append([train_x[i]/255, convert_to_one_hot(train_y[i], 10)])

test_data = []
test_x = test_x.reshape([10000, 784])
for i in range(10000):
    test_data.append([test_x[i]/255, test_y[i]])


x,y=test_data[0]
print(x)