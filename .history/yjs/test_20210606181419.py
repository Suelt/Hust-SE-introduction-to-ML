import numpy as np

size=[784,30,10]
num=2
weight = [np.random.randn(ch2,ch1)
                       for ch1,ch2 in zip(size[:-1], size[1:])]
        # [784,30],[30,10]  z=wxx+b [30,1]
bias = [np.random.rand(s, 1) for s in size[1:]]
print(size[:-1])
print(size[1:])