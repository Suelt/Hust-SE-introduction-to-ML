import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import copy
from scipy.stats import norm

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")
y = credit['credit_risk']

X = credit.loc[:,'status':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
cols = ['status','duration','credit_history', 'purpose','amount','savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','age','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']


train=credit.loc[y_train.index]

train_good=train.loc[train['credit_risk']=='good']
length_good=train_good.shape[0]
train_bad=train.loc[train['credit_risk']=='bad']
length_bad=train_bad.shape[0]


dict_main_true={}
dict_main_false={}

for col in cols:
    dict_main_true[col]={}
    dict_main_false[col]={}
#满足P(Xij|yk)的个数
number_value=0
#满足P(Xij|yk)的概率
rate=0
cols.remove('duration')
cols.remove('amount')
cols.remove('age')
# print(cols)
for col in cols:
    dict_new_good={}
    dict_new_bad={}
    values =train_good[col].value_counts().keys().tolist()
    for value in values:
        number_value=train_good[col].value_counts()[value]
        rate=number_value/length_good
        dict_new_good[value]=rate

        number_value=train_bad[col].value_counts()[value]
        rate=number_value/length_bad
        dict_new_bad[value]=rate
    dict_main_true[col]=dict_new_good
    dict_main_false[col]=dict_new_bad

dict_gaussian={}
dict_gaussian['duration']={}
dict_gaussian['amount']={}
dict_gaussian['age']={}

for key in dict_gaussian:
    dict_new={}
    list_good=train_good[key]
    arr_mean = np.mean(list_good)
    arr_std = np.std(list_good,ddof=1)
    dict_new['good']=[arr_mean,arr_std]
    list_bad=train_bad[key]
    arr_mean = np.mean(list_bad)
    arr_std = np.std(list_bad,ddof=1)
    dict_new['bad']=[arr_mean,arr_std]
    dict_gaussian[key]=dict_new


y=copy.deepcopy(y_test)
#print(X_test.index)
result_true=1
result_false=1
print(X_test.loc[507])
for i in y_test.index:
    result_true=1
    result_false=1
    list_new=X_test.loc[i]
    for index in X_test.loc[i].keys().tolist():
        if index=='duration':
            rate=norm.pdf(list_new['duration'],loc=dict_gaussian['duration']['good'][0],scale=dict_gaussian['duration']['good'][1])
        elif index=='amount':
            rate=norm.pdf(list_new['amount'],loc=dict_gaussian['amount']['good'][0],scale=dict_gaussian['amount']['good'][1])
        elif index=='age':
            rate=norm.pdf(list_new['age'],loc=dict_gaussian['age']['good'][0],scale=dict_gaussian['age']['good'][1])
        else:
            rate=dict_main_true[index][list_new[index]]
        result_true=result_true*rate*10
# print(dict_main_true)
# print(dict_main_false)