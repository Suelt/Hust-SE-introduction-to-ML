import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import graphviz
import pydotplus
from mpl_toolkits.mplot3d import Axes3D



def calculatent(DataSet):
    risk_type = DataSet['credit_risk'].value_counts()
    m = DataSet.shape[0]
    p = risk_type/m
    ent = (-p*np.log2(p)).sum()
    return ent
credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")

col_dicts = {}
cols = ['status','credit_history', 'purpose', 'savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']

col_dicts = {
'status':{
'... >= 200 DM / salary for at least 1 year':0,
'no checking account':1,
'... < 0 DM':2,
'0<= ... < 200 DM':3
},
'credit_history':{
'no credits taken/all credits paid back duly':0,
'all credits at this bank paid back duly':1,
'existing credits paid back duly till now':2,
'critical account/other credits elsewhere':3,
'delay in paying off in the past':4
},
'purpose':{
'furniture/equipment':0,
'others':1,
'car (used)':2,
'car (new)':3,
'retraining':4,
'repairs':5,
'domestic appliances':6,
'radio/television':7,
'business':8,
'vacation':9
},
'savings':{
'unknown/no savings account':0,
'... >= 1000 DM':1,
'... <  100 DM':2,
'100 <= ... <  500 DM':3,
'500 <= ... < 1000 DM':4
},
'employment_duration':{
'1 <= ... < 4 yrs':0,
'>= 7 yrs':1,
'4 <= ... < 7 yrs':2,
'< 1 yr':3,
'unemployed':4
},
'installment_rate':{
'< 20':0,
'25 <= ... < 35':1,
'20 <= ... < 25':2,
'>= 35':3
},
'personal_status_sex':{
'male : married/widowed':0,
'female : non-single or male : single':1,
'female : single':2,
'male : divorced/separated':3
},
'other_debtors':{
'none':0,
'guarantor':1,
'co-applicant':2
},
'present_residence':{
'>= 7 yrs':0,
'1 <= ... < 4 yrs':1,
'4 <= ... < 7 yrs':2,
'< 1 yr':3
},
'property':{
'building soc. savings agr./life insurance':0,
'unknown / no property':1,
'car or other':2,
'real estate':3
},
'other_installment_plans':{
'none':0,
'bank':1,
'stores':2
},
'housing':{
'rent':0,
'for free':1,
'own':2
},
'number_credits':{
'1':0,
'2-3':1,
'4-5':2,
'>= 6':3
},
'job':{
'skilled employee/official':0,
'unskilled - resident':1,
'manager/self-empl./highly qualif. employee':2,
'unemployed/unskilled - non-resident':3
},
'people_liable':{
'0 to 2':0,
'3 or more':1
},
'telephone':{
'no':0,
'yes (under customer name)':1
},
'foreign_worker':{
'no':0,
'yes':1
}}
for col in cols:
     credit[col] = credit[col].map(col_dicts[col])
    
#credit.drop(columns = ['people_liable','present_residence','telephone','job'],inplace = True)
y = credit['credit_risk']
X = credit.loc[:,'status':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
class_weights = {'good':1,'bad':2}
credit_model = DecisionTreeClassifier(min_samples_leaf = 13,max_leaf_nodes=17,class_weight = class_weights,)
credit_model.fit(X_train, y_train)

credit_pred = credit_model.predict(X)

credit.drop(columns = ['duration','amount','age'],inplace = True)
length=len(credit)
credit.drop(columns='credit_risk',inplace=True)
credit['credit_risk']=credit_pred


dict_result=credit['credit_risk'].value_counts()
p_result=dict_result/length
ent_A = (-p_result*np.log2(p_result)).sum()

print(ent_A)
for col in cols:
     credit[col] = credit[col].map(col_dicts[col])

dict={}
ent_B=0
for i in range(len(cols)):
    index_ = credit.iloc[:,i].value_counts().index
    ent_B=0
    for j in index_:
        child_DataSet = credit[credit.iloc[:,i]==j]
        ent=calculatent(child_DataSet)
        ent_B += (child_DataSet.shape[0]/credit.shape[0])*ent
    print(ent_B)
    dict[cols[i]]=ent_B

dict_minus={}
for index,result in dict.items():
    dict_minus[index]=ent_A-result

corr={}

for col in cols:
    numerator=dict_minus[col]
    denominator=ent_A*dict[col]
    denominator=denominator**0.5
    corr[col]=numerator/denominator
for index,result in corr.items():
    print(index,"     ",result)
sorted_dataset= sorted(corr.items(),key = lambda item:item[1],reverse=True)
x = []
y = []
for d in sorted_dataset:
    x.append(d[0])
    y.append(d[1])
plt.bar(x[0:17], y[0:17])
#plt.show()

#print (metrics.classification_report(y_test, credit_pred))
#print (metrics.confusion_matrix(y_test, credit_pred))
#print (metrics.accuracy_score(y_test, credit_pred))




