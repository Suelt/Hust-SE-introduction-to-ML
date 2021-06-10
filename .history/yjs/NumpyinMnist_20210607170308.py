import numpy as np
from tensorflow.keras.datasets import mnist



class NumpyinMnist():

    def __init__(self):
        self.layers=2
        self.weight=[]
        self.weight.append(np.random.randn(30,784))
        self.weight.append(np.random.randn(10,30))
        self.bias=[]
        self.bias.append(np.random.rand(30,1))
        self.bias.append(np.random.rand(10,1))
       
    

    def forward_compute(self,x):
        for i in range(2):
            b=self.bias[i]
            w=self.weight[i]
    
            z = w@x+b
            x = sigmoid(z)
        return x

    def backpropagation(self, x, y):
       
        x=x.reshape(784,1)
      
        gradient_w = [np.zeros(w.shape) for w in self.weight]
        gradient_b = [np.zeros(b.shape) for b in self.bias]

       
        intermediate_list = []
        res = []
        intermediate = x
        for i in range(2):
            b=self.bias[i]
            w=self.weight[i]
            z = w@intermediate + b
            intermediate = sigmoid(z)

            res.append(z)
            intermediate_list.append(intermediate)

        #隐层->输出层
        loss=np.power((intermediate_list[-1]-y),2).sum()
        delta = intermediate_list[-1] * (1 - intermediate_list[-1]) * (intermediate_list[-1] - y)
        gradient_b[-1] = delta
    
        intermediate_output=intermediate_list[-2].T
        delta_w=delta@intermediate_output
        gradient_w[-1] = delta_w



        #隐层->输入层
        
        z = res[-2]
        a = intermediate_list[-2]
        delta = np.dot(self.weight[-1].T, delta) * a * (1 - a)

        gradient_b[-2] = delta
        delta_w=delta@x.T
        gradient_w[-2] = delta_w

        return gradient_w, gradient_b,loss

    def train(self, training_data,test_data, epoches, batch_size, lr):
        n = 60000
        for j in range(epoches):

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
       
           
            print("Epoch"+str(j+1)+" :"+"   accuracy:"+str(self.ifprint(test_data)/10000)+"   loss:"+str(loss)) 
            
         


    def ifprint(self, test_data):
        sum=0
        for x,y in test_data:
            pred=np.argmax(self.forward_compute(x.reshape([784,1])))
            if pred==y:
                sum+=1
        return sum




def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def main(self):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_data = []
    train_x = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
    train_x=train_x/255

  
    test_data = []
    test_x = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])
    test_x=test_x/255
    for i in range(60000):
        train_data.append([train_x[i],np.eye(10)[y_train[i].reshape(-1)].T])
    for i in range(10000):
        test_data.append([test_x[i], y_test[i]])

    demo=NumpyinMnist()
    demo.train(train_data,test_data,50,10,0.2)

if __name__ == '__main__':
    
    main()
    