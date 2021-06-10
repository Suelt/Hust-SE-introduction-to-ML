import numpy as np
from tensorflow.keras.datasets import mnist
# def convert_to_one_hot(y, C):
#     return np.eye(C)[y.reshape(-1)].T

def sigmoid(X):

    return 1.0 / (1 + np.exp(-X))

class NumpyinMnist():
    def _init_(self,message):
        print(message)
        self.weight=[]
        self.bias=[]
        self.weight.append(np.random.randn(30,784))
        self.weight.append(np.random.randn(10,30))
        self.bias.append(np.random.rand(30,1))
        self.bias.append(np.random.rand(10,1))
        print(self.weight)

    def forward(self,x):
        for i in range(2):
            b=self.bias[i]
            w=self.weight[i]
            b_axis1=[]
            for i in range(len(b)):
                b_axis1.append(b[i][0])
            z = w@x+b_axis1
            x = sigmoid(z)
        return x




if __name__ == '__main__':
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    X_train=X_train/255

    Y_train = (np.arange(10)==y_train[:, None]).astype(int)

    
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    X_test=X_test/255

    Y_test = (np.arange(10) == y_test[:, None]).astype(int)

    demo=NumpyinMnist( )


# train_data = []
# for i in range(train_x.shape[0]):
#     # print(convert_to_one_hot(train_y[i],10).shape)
#     train_data.append([train_x[i]/255, convert_to_one_hot(train_y[i], 10)])

# test_data = []
# test_x = test_x.reshape([10000, 784])
# for i in range(10000):
#     test_data.append([test_x[i]/255, test_y[i]])



# for i in range(2):
#     b=bias[i]
#     w=weight[i]
#     b_axis1=[]
#     for i in range(len(b)):
#         b_axis1.append(b[i][0])

#     z = w@x+b_axis1
#     #print(z)
#     x = sigmoid(z)
   

   