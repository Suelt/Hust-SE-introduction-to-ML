import numpy as np
from keras.datasets import mnist

weight=[]
weight.append(np.random.randn(30,784))
weight.append(np.random.randn(10,30))
bias=[]
bias.append(np.random.rand(30,1))
bias.append(np.random.rand(10,1))

gradient_w=[]
gradient_b=[]
for w,b in weight,bias:
    gradient_w.append(np.zeros(w.shape))
    gradient_b.append(np.zeros(b.shape))
print(gradient_w.shape)