# conding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
from keras.datasets import mnist

def tanh(z):#tanh函数
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def relu(z):#relu函数
    return np.maximum(0,z)
def tanh_1(z):#tanh函数的导数
    return 1-tanh(z)**2
def relu_1(z):#relu函数的导数
    return np.maximum(0, z/np.abs(z))
def sigmoid(z):
    return 1/(1+np.exp(-z))

def ward(L,n,m,dim):#对参数进行初始化
    np.random.seed(1)
    w = []
    b = []
    Vdw, Vdb, Sdw, Sdb = [],[],[],[]#优化算法的值
    for i in range(0, L):
        if i != 0 and i != L - 1:
            #p = np.random.randn(dim, dim) *0.001
            p = np.random.randn(dim, dim) * np.sqrt(2 / dim)
        elif i == 0:
            #p = np.random.randn(dim, m) * 0.001
            p = np.random.randn(dim, m) * np.sqrt(2 / m)
        else:
            #p = np.random.randn(1, dim) * 0.001
            p = np.random.randn(10, dim) * np.sqrt(2 / dim)
        w.append(p)
        b.append(1)
        Vdw.append(0)#初始化为0
        Vdb.append(0)
        Sdw.append(0)
        Sdb.append(0)

    return w, b,Vdw,Vdb,Sdw,Sdb

def forward_back(w,b,a,Y,L,m,lambd):#用于训练的归一化前向传播
    z = []
    add = 0
    sigma2,mu = [],[]
    for i in range(0, L):
        zl = np.dot(w[i], a) + b[i]

        #归一化输入z
        muL = (1/m)*np.sum(zl,axis=1,keepdims=True)
        sigmaL = (1/m)*np.sum(np.power(zl-muL,2),axis=1,keepdims=True)
        z_norm = (zl-muL)/(np.sqrt(sigmaL+0.00000001))
        gamma,beta_1 = 1*np.sqrt(sigmaL+0.00000001),muL+0#此时z的方差为1，均值为0
        zl = np.multiply(z_norm,gamma) + beta_1
        mu.append(muL)
        sigma2.append(sigmaL)
        #可以发现，如果使用zl = z_norm此时平均值为0，方差为1，对于我们这个数据是没有影响的，因为我们图片像素点已经除以了255，也比较均匀，是基本处于这个范围内的数
        #因为z = γ*z_norm+β可以看成线性函数，类似于z = wx+b，也可以对γ和β进行梯度下降更新。

        add += np.sum((lambd / (2 * m)) * np.dot(w[i], w[i].T))#L2正则化项
        z.append(zl)
        a = relu(zl)

    #使用softmax回归
    t = np.exp(zl)
    ti = np.sum(t,axis = 0,keepdims=True)#axis=0表示对行求和，对输出zl的行求和，这样可以保证最后概率之和为1
    a = np.divide(t,ti)#矩阵除法函数，也可以用/。不过我这样用/有时候会提示一些奇怪错误，提示无法进行除法
    #损失函数也应该重新定义
    J = (-1/m)*np.sum(Y*np.log(a))+ add #注意Y*nu.log(a)计算得到是多行的矩阵，但只有一行是非0行，我们之前定义的损失函数相当于这个特殊情况

    # a = sigmoid(zl)
    #J = (-1/m)*np.sum(1 * Y * np.log(a) + (1 - Y) * np.log(1 - a)) + add  # 损失函数
    return z, a, J,sigma2,mu

def forward_test(w, b, a, Y, L, m, lambd,sigma2,mu):  # 用于测试的归一化前向传播
    for i in range(0, L):
        zl = np.dot(w[i], a) + b[i]

        # 归一化输入z
        z_norm = (zl - mu[i]) / (np.sqrt(sigma2[i] + 0.00000001))
        gamma, beta_1 = 1 * np.sqrt(sigma2[i] + 0.00000001), mu[i] + 0  # 此时z的方差为1，均值为0
        zl = np.multiply(z_norm, gamma) + beta_1
        # 可以发现，如果使用zl = z_norm此时平均值为0，方差为1，对于我们这个数据是没有影响的，因为我们图片像素点已经除以了255，也比较均匀，是基本处于这个范围内的数
        a = relu(zl)
    # 使用softmax回归
    t = np.exp(zl)
    ti = np.sum(t, axis=0, keepdims=True)  # axis=0表示对行求和，对输出zl的行求和，这样可以保证最后概率之和为1
    a = np.divide(t, ti)  # 矩阵除法函数，也可以用/。不过我这样用/有时候会提示一些奇怪错误，提示无法进行除法
    # a = sigmoid(zl)
    return a

def backward(w,b,X,Y,m,L,lambd):#反向传播
    z,a,J,sigma2,mu = forward_back(w,b,X,Y,L,m,lambd)
    dw,db = [],[]
    for i in range(L - 1, 0, -1):
        if i == L - 1:
            dz = a - Y
        else:
            dz = np.dot(w[i + 1].T, dz) * relu_1(z[i])
        Dw = 1 / m * (np.dot(dz, relu(z[i - 1]).T)) + (lambd / m) * w[i]
        Db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        dw.append(Dw)
        db.append(Db)

    dz = np.dot(w[1].T, dz) * relu_1(z[0])
    Dw = 1 / m * np.dot(dz, X.T) + (lambd / m) * w[0]
    Db = 1 / m * np.sum(dz, axis=1, keepdims=True)
    dw.append(Dw)
    db.append(Db)
    return dw, db, J,sigma2,mu
def back_momentum(w,b,X,Y,learning,m,L,lambd,beta,Vdw,Vdb):#使用momentum梯度下降的反向传播
    dw, db, J,sigma2,mu = backward(w,b,X,Y,m,L,lambd)
    # 不使用偏差修正不影响最后结果
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Vdw[i] = beta * Vdw[i] + (1 - beta) * dw[L-i-1]
        Vdb[i] = beta * Vdb[i] + (1 - beta) * db[L-i-1]
        w[i] = w[i] - learning * Vdw[i]
        b[i] = b[i] - learning * Vdb[i]

    return w,b,J,Vdw,Vdb,sigma2,mu
def back_RMSprop(w,b,X,Y,learning,m,L,lambd,beta,Sdw,Sdb):#使用RMSprop梯度下降的反向传播
    dw, db, J ,sigma2,mu= backward(w,b,X,Y,m,L,lambd)
    # 不使用偏差修正不影响最后结果
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Sdw[i] = beta * Sdw[i] + (1 - beta) * np.power(dw[L-i-1],2)
        Sdb[i] = beta * Sdb[i] + (1 - beta) * np.power(db[L-i-1],2)
        w[i] = w[i] - learning * dw[L-i-1] / (np.power(Sdw[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大
        b[i] = b[i] - learning * db[L-i-1] / (np.power(Sdb[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大

    return w,b,J,Sdw,Sdb,sigma2,mu
def back_Adam(w,b,X,Y,learning,m,L,lambd,beta,Vdw,Vdb,Sdw,Sdb):#使用Adam梯度下降的反向传播
    dw, db, J ,sigma2,mu= backward(w,b,X,Y,m,L,lambd)
    # 通常使用偏差修正
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Vdw[i] = beta * Vdw[i] + (1 - beta) * dw[L - i - 1]
        Vdb[i] = beta * Vdb[i] + (1 - beta) * db[L - i - 1]
        Sdw[i] = beta * Sdw[i] + (1 - beta) * np.power(dw[L-i-1],2)
        Sdb[i] = beta * Sdb[i] + (1 - beta) * np.power(db[L-i-1],2)
        '''
        Vdw[i] /= 1 - np.power(beta, t)#t表示代数（所有mini_batch训练一次称为一代）
        Vdb[i] /= 1 - np.power(beta, t)
        Sdw[i] /= 1 - np.power(beta, t)
        Sdb[i] /= 1 - np.power(beta, t)
        '''
        w[i] = w[i] - learning * Vdw[i] / (np.power(Sdw[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大
        b[i] = b[i] - learning * Vdb[i] / (np.power(Sdb[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大

    return w,b,J,Vdw,Vdb,Sdw,Sdb,sigma2,mu

def text(x,y,w,b,L,m,lambd,sigma2, mu):#查看测试集准确率
    a = forward_test(w, b, x, y, L, m,lambd,sigma2,mu)
    #得到的a是一个10行，10000列的矩阵，里面数表示概率
    a = a.T#先将a转置
    for i in range(x.shape[1]):
        a[i] = np.where(a[i].max() == a[i], 1, 0)#最大数位置改为存1，不是最大数改为存0，这样就处理好y^了
    a = a.T  # 将a转回去
    lop_1 = np.sum(np.abs(y-a),axis = 0,keepdims=True)/2#这个lop_1中存的是1和0，1表示预测错误，0表示正确
    lop_2 = 1-np.sum(lop_1,axis=1,keepdims=True)/10000#表示正确所占比例，也就是准确率
    lop_2=np.squeeze(lop_2)

    print("测试集准确性：{0}%".format(lop_2*100))

    return 0

if __name__ == "__main__":
    L = 10#神经网络层数
    dim = 20#隐藏层节点个数
    learning_rate = 0.05#学习率
    loss= []#损失函数
    lambd = 0.01 # L2正则化参数
    beta = 0.9#β值；1/（1-β）表示平均前多少轮的指数加权平均数
    decay_rate = 0.0009#学习率衰减率
    mini_batch = 300#一次的训练量，60000万张图片，要带入200次，全部训练一遍称为一代
    sigma2, mu = 0, 0  # 用于决定将batch正则化拟合进神经网络的均值和方差


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    X_train=X_train/255
    X_train = np.array(X_train).T
    Y_train = (np.arange(10)==y_train[:, None]).astype(int)
    Y_train = np.array(Y_train).T
  
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    X_test=X_test/255
    X_test = np.array(X_test).T
    Y_test = (np.arange(10) == y_test[:, None]).astype(int)
    Y_test = np.array(Y_test).T
    



    
    w,b,Vdw,Vdb,Sdw,Sdb = ward(L,X_train.shape[1],X_train.shape[0],dim)#vdw表示momentum，Sdw表示RMSprop
    for i in range(0,200):
        Sigma2, Mu,J_average = 0,0,0#用于决定将batch正则化拟合进神经网络的均值和方差
        for j in range(0,(X_train.shape[1]//mini_batch)):
            #w,b,J,Vdw,Vdb = back_momentum(w,b,train_set_x,train_set_y,learning_rate,train_set_x.shape[1],L,lambd,beita,Vdw,Vdb)
            #w,b,J,Vdw,Vdb = back_RMSprop(w, b, train_set_x,train_set_y,learning_rate,train_set_x.shape[1],L,lambd,beita, Sdw, Sdb)
            w,b,J,Vdw,Vdb,Sdw,Sdb,sigma2,mu = back_Adam(w,b,((X_train.T)[j*mini_batch:(j+1)*mini_batch]).T,((Y_train.T)[j*mini_batch:(j+1)*mini_batch]).T,learning_rate,mini_batch,L,lambd,beta,Vdw,Vdb, Sdw, Sdb)
            #如果有多个mini_batch应该在此处对返回的sigam2和mu使用指数加权平均数，也是滑动平均数
            Sigma2 = np.multiply(beta,Sigma2) + np.multiply((1-beta),sigma2)
            Mu = np.multiply(beta,Mu) + np.multiply((1-beta),mu)
            J_average = np.multiply(beta,J) + np.multiply((1-beta),J)
        learning_rate = learning_rate * (1 / (1 + i*decay_rate) )#使用学习率衰减
        if i % 10 == 0:
            print("loss：",J_average)
            loss.append(J_average)
    plt.plot(loss)#打印损失函数
    plt.show()

    text(X_test,Y_test,w,b,L,X_test.shape[1],lambd,Sigma2,Mu)
    

