import numpy as np
from keras.datasets import mnist

weight=[]
weight.append(np.random.randn(30,784))
weight.append(np.random.randn(10,30))
bias=[]
bias.append(np.random.rand(30,1))
bias.append(np.random.rand(10,1))