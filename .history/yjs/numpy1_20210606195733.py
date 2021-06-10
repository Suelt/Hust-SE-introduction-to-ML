import numpy as np
from keras.datasets import mnist

weight=[]
weight.append(np.random.randn(30,784))
weight.append(np.random.randn(10,30))
bias=[]
bias.append(np.random.rand(30,1))
bias.append(np.random.rand(10,1))

nabla_w = [np.zeros(w.shape) for w in weight]
nabla_b = [np.zeros(b.shape) for b in bias]
print(nabla_w)