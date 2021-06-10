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
        n = 60000
        for j in range(epoches):

            #random.shuffle(train_data)
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, n, batch_size)]
     
            for batch in batches:
                batch_gradient_w = [np.zeros(w.shape) for w in self.weight]
                batch_gradient_b = [np.zeros(b.shape) for b in self.bias]
                batch_loss=0
    

                for x, y in batch:
         
                    gradient_w, gradient_b,loss = self.backpropagation(x, y)
                    batch_gradient_w = [batch_w + w for batch_w, w in zip(batch_gradient_w, gradient_w)]
                    batch_gradient_b = [batch_b + b for batch_b, b in zip(batch_gradient_b, gradient_b)]
                    batch_loss+=loss
               
                batch_gradient_w = [w / len(batch) for w in batch_gradient_w]
                batch_gradient_b = [b / len(batch) for b in batch_gradient_b]
                batch_loss=batch_loss/len(batch)
      
                self.weight = [w - lr * batch_w for w,batch_w in zip(self.weight, batch_gradient_w)]
                self.bias = [b - lr * batch_b for b, batch_b in zip(self.bias, batch_gradient_b)]
                loss=batch_loss
       

            if test_data:
                n_test = len(test_data)
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test),loss)
            else:
                print("Epoch {0} complete".format(j))

    # def update_mini_batch(self, batch, lr):
    
    #     batch_gradient_w = [np.zeros(w.shape) for w in self.weight]
    #     batch_gradient_b = [np.zeros(b.shape) for b in self.bias]
    #     batch_loss=0
    #     # for every sample in current batch
    #     for x, y in batch:
    #         # list of every w/b gradient
    #         # [w1,w2,w3]
    #         gradient_w, gradient_b,loss = self.backpropagation(x, y)
    #         batch_gradient_w = [batch_w + w for batch_w, w in zip(batch_gradient_w, gradient_w)]
    #         batch_gradient_b = [batch_b + b for batch_b, b in zip(batch_gradient_b, gradient_b)]
    #         batch_loss+=loss
    #     batch_gradient_w = [w / len(batch) for w in batch_gradient_w]
    #     batch_gradient_b = [b / len(batch) for b in batch_gradient_b]
    #     batch_loss=batch_loss/len(batch)
    #     # w = w - lr * nabla_w
    #     self.weight = [w - lr * batch_w for w,batch_w in zip(self.weight, batch_gradient_w)]
    #     self.bias = [b - lr * batch_b for b, batch_b in zip(self.bias, batch_gradient_b)]

    #     return batch_loss

    def evaluate(self, test_data):
        
        sum=0
        pred=[]
        for x,y in test_data:
            pred.append(np.argmax(self.forward(x.reshape([784,1]))))
        result_bool = np.equal(result_max, test_data)
        true_num = np.sum(result_bool)



        return sum


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
    demo.train(train_data,test_data,10,100,0.1)


   