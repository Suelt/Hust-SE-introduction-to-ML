import numpy as np
from keras.datasets import mnist

weight=[]
weight.append(np.random.randn(30,784))
weight.append(np.random.randn(10,30))
bias=[]
bias.append(np.random.rand(30,1))
bias.append(np.random.rand(10,1))


def sigmoid(X):

    return 1.0 / (1 + np.exp(-X))
intermediate_list = [x]
for i in range(2):
    b=bias[i]
    w=weight[i]
    z = w@intermediate + b
    intermediate = sigmoid(z)

    
    intermediate_list.append(intermediate)
print(len(intermediate_list))