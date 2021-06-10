import numpy as np
from tensorflow.keras.datasets import mnist
import random


class NumpyinMnist():

    def __init__(self):
        self.layers=2
        self.weight=[]
        self.weight.append(np.random.randn(30,784))
        self.weight.append(np.random.randn(10,30))
        self.bias=[]
        self.bias.append(np.random.rand(30,1))
        self.bias.append(np.random.rand(10,1))
        # size[784,30,10]
        # w:[output, input]
        # b:[output]
    

    def forward(self,x):
        for i in range(2):
            b=self.bias[i]
            w=self.weight[i]
            # b_axis1=[]
            # for i in range(len(b)):
            #     b_axis1.append(b[i][0])
            z = w@x+b
            x = sigmoid(z)
        return x

    def backpropagation(self, x, y):
       
        x=x.reshape(784,1)
      
        gradient_w = [np.zeros(w.shape) for w in self.weight]
        gradient_b = [np.zeros(b.shape) for b in self.bias]

       
        intermediate_list = []
        zs = []
        intermediate = x
        for i in range(2):
            b=self.bias[i]
            w=self.weight[i]
            z = w@intermediate + b
            intermediate = sigmoid(z)

            zs.append(z)
            intermediate_list.append(intermediate)

        #隐层->输出层
        loss=np.power((intermediate_list[-1]-y),2).sum()
        delta = intermediate_list[-1] * (1 - intermediate_list[-1]) * (intermediate_list[-1] - y)
        gradient_b[-1] = delta
    
        intermediate_output=intermediate_list[-2].T
        delta_w=delta@intermediate_output
        gradient_w[-1] = delta_w



        #隐层->输入层
        
        z = zs[-2]
        a = intermediate_list[-2]
        delta = np.dot(self.weight[-1].T, delta) * a * (1 - a)

        gradient_b[-2] = delta
        delta_w=delta@x.T
        gradient_w[-2] = delta_w

        return gradient_w, gradient_b,loss

    def train(self, training_data,test_data, epoches, batch_size, lr):
        """

        :param training_data: list of (x,y)
        :param epoches: 1000
        :param batchsz: 10
        :param lr: 0.1
        :param test_data: list of (x,y)
        :return:
        """
        n = 60000
        for j in range(epoches):
            #random.shuffle(train_data)
            mini_batches = [
                training_data[k:k + batch_size]
                for k in range(0, n, batch_size)]

            # for every batch in current batch
            for mini_batch in mini_batches:



                loss=self.update_mini_batch(mini_batch, lr)
            if test_data:
                n_test = len(test_data)
                #print("Epoch "+str(j+1))
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test),loss)
            else:
                print("Epoch"+str(j+1)+" complete")

    def update_mini_batch(self, batch, lr):
        """

        :param batch: list of (x,y)
        :param lr: 0.01
        :return:
        """
        gradient_w = [np.zeros(w.shape) for w in self.weight]
        gradient_b = [np.zeros(b.shape) for b in self.bias]
        loss=0

        for x, y in batch:
         
            gradient_w, gradient_b,loss = self.backpropagation(x, y)
            gradient_w_sum = [accu + cur for accu, cur in zip(gradient_w, gradient_w)]
            gradient_b_sum = [accu + cur for accu, cur in zip(gradient_b, gradient_b)]
            loss_sum+=loss
        gradient_w_sum = [w / len(batch) for w in gradient_w]
        gradient_b_sum = [b / len(batch) for b in gradient_b]
        loss_sum=loss_sum/len(batch)

        self.weight = [w - lr * gradient for w, gradient in zip(self.weight, gradient_w_sum)]
        self.bias = [b - lr * gradient for b, gradient in zip(self.bias, gradient_b_sum)]

        return loss

    def evaluate(self, test_data):
        """

        :param test_data: list of (x,y)
        :return:
        """
        result = [(np.argmax(self.forward(x.reshape([784,1]))), y)
                  for x, y in test_data]

        correct = sum(int(pred == y) for pred, y, in result)

        return correct


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
def sigmoid(X):

    return 1.0 / (1 + np.exp(-X))

if __name__ == '__main__':
    
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_data = []
    train_x = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    test_data = []
    test_x = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])
    for i in range(train_x.shape[0]):
        train_data.append([train_x[i]/255, convert_to_one_hot(y_train[i], 10)])
    for i in range(test_x.shape[0]):
        test_data.append([test_x[i]/255, y_test[i]])

    demo=NumpyinMnist()
    demo.train(train_data,test_data,10,10,0.1)


   