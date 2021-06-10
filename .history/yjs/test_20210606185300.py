import numpy as np
from tensorflow.keras.datasets import mnist
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def sigmoid(inX):
    from numpy import exp
    return 1.0 / (1 + exp(-inX))


size=[784,30,10]
num=2
weight = [np.random.randn(ch2,ch1)
                       for ch1,ch2 in zip(size[:-1], size[1:])]
        # [784,30],[30,10]  z=wxx+b [30,1]
bias = [np.random.rand(s, 1) for s in size[1:]]


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


x,y=train_data[0]
print(y.size)
print(x.size)
#print(x)
for b, w in zip(bias, weight):
    z=[]
    for i in range(30):
        z_temp=np.dot(w[i],x)+b[i]
        z.append(z_temp)
    print(z)
    x=sigmoid(z)
