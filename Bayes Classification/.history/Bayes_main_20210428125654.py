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

train_true=credit[train['credit_risk'].isin(['good'])]

train_bad=credit[train['credit_risk'].isin(['bad'])]

print(train_true.shape[0])
print(train_true)
for col in cols:
    dict_main_true[col]={}
    dict_main_false[col]={}



