import pandas as pd
import numpy as np

def load_dataset():
    df = pd.read_csv('train.csv')  # 得到的是一个字典集
    f1 = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    f2 = 'label'
    train_x = np.array(df[f1].values)  # 通过键获取字典数据，并且转化为矩阵
    train_y = np.array(df[f2].values)
    train_y=pd.Series(train_y)
    train_y=np.array(pd.get_dummies(train_y))#独热码实现softmax
    #print(train_y[0:12])
    train_set_y, test_set_y=train_y[0:40000].T,train_y[40000:42000].T

    #print(train_x.shape[0], train_x.shape[1])  # 输出维度
    print(train_y.shape[0],train_y.shape[1])  # 输出维度

    dp = pd.read_csv('test.csv')  # 得到的是一个字典集
    f = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    test_x = np.array(dp[f].values)  # 通过键获取字典数据，并且转化为矩阵
    #print(test_x.shape[0], test_x.shape[1])  # 输出维度

    train_set_x_orig,  test_set_x_orig = train_x[0:40000],train_x[40000:42000]
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y

def load_dataset2():
    df = pd.read_csv('train.csv')  # 得到的是一个字典集
    f1 = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    f2 = 'label'
    train_x = np.array(df[f1].values)  # 通过键获取字典数据，并且转化为矩阵
    train_y = np.array(df[f2].values)
    train_y=pd.Series(train_y)
    train_y=np.array(pd.get_dummies(train_y))#独热码实现softmax
    #print(train_y[0:12])
    train_set_y=train_y.T

    #print(train_x.shape[0], train_x.shape[1])  # 输出维度
    print(train_y.shape[0],train_y.shape[1])  # 输出维度

    dp = pd.read_csv('test.csv')  # 得到的是一个字典集
    f = [f"pixel{i}" for i in range(0, 28 * 28)]  # 产生字符串列表，从pixel0到pixel783
    test_x = np.array(dp[f].values)  # 通过键获取字典数据，并且转化为矩阵
    #print(test_x.shape[0], test_x.shape[1])  # 输出维度

    train_set_x_orig,  test_set_x_orig = train_x,test_x
    return train_set_x_orig, train_set_y, test_set_x_orig

#train_set_x_orig, train_set_y, test_set_x_orig, test_set_y=load_dataset()
#print(train_set_y.shape[0],train_set_y.shape[1])
#print(train_set_x_orig.shape[0],train_set_x_orig.shape[1])
