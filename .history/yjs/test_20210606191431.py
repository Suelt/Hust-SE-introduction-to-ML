import numpy as np
from tensorflow.keras.datasets import mnist
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T

def sigmoid(inX):
    from numpy import exp
    return 1.0 / (1 + exp(-inX))


size=[784,30,10]

weight=[]
weight.append(np.random.randn(30,784))
weight.append(np.random.randn(10,30))
bias=[]
bias.append(np.random.rand(30,1))
bias.append(np.random.rand(10,1))


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


x,y=train_data[1]
print(y.size)
print(x.size)
#print(x)
times=0

for i in range(2):
    b=bias[i]
    w=weight[i]
    b_axis1=[]
    for i in range(len(b)):
        b_axis1.append(b[i][0])

    z = w@x+b_axis1
    #print(z)
    x = sigmoid(z)
   
   