import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")
y = credit['credit_risk']

X = credit.loc[:,'status':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
cols = ['status','duration','credit_history', 'purpose','amount','savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','age','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']
dict_main_true={}
dict_main_false={}

train=credit.loc[y_train.index]

train_good=train.loc[train['credit_risk']=='good']
length_good=train_good.shape[0]
train_bad=train.loc[train['credit_risk']=='bad']
length_bad=train_bad.shape[0]


for col in cols:
    dict_main_true[col]={}
    dict_main_false[col]={}
#满足P(Xij|yk)的个数
number_value=0
#满足P(Xij|yk)的概率
rate=0
for col in cols:
    dict_new={}
    list=train_good[col]
    print(list)
    values =list.value_counts().keys().tolist()
    print(values)
    for value in values:
        number_value=list.count(value)
        print(number_value)
        rate=number_value/length_good





