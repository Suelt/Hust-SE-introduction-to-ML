import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection


credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")

col_dicts = {}
cols = ['status','credit_history', 'purpose', 'savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']
j=0
print("col_dicts = {")
for x in cols:
    j+=1
    print("'"+x+"':{")
    i=0
    #print(credit[x].value_counts().index.size)
    for y in credit[x].value_counts().index:
        print("'"+y+"':%d"%(i),end='')
        i+=1
        if i<credit[x].value_counts().index.size:
            print(',',end='')
        print()
    print('}',end='')
    if j<len(cols):
        print(',')
    print('')
print("}")


# col_dicts = {
# 'status':{
# '... >= 200 DM / salary for at least 1 year':             ,
# 'no checking account':             ,
# '... < 0 DM':             ,
# '0<= ... < 200 DM':
# },

# 'credit_history':{
# 'no credits taken/all credits paid back duly':             ,
# 'all credits at this bank paid back duly':             ,
# 'existing credits paid back duly till now':             ,
# 'critical account/other credits elsewhere':             ,
# 'delay in paying off in the past':
# },

# 'purpose':{
# 'furniture/equipment':             ,
# 'others':             ,
# 'car (used)':             ,
# 'car (new)':             ,
# 'retraining':             ,
# 'repairs':             ,
# 'domestic appliances':             ,
# 'business':             ,
# 'radio/television':             ,
# 'vacation':
# },

# 'savings':{
# 'unknown/no savings account':             ,
# '... >= 1000 DM':             ,
# '... <  100 DM':             ,
# '100 <= ... <  500 DM':             ,
# '500 <= ... < 1000 DM':
# },

# 'employment_duration':{
# '1 <= ... < 4 yrs':             ,
# '>= 7 yrs':             ,
# '4 <= ... < 7 yrs':             ,
# '< 1 yr':             ,
# 'unemployed':
# },

# 'installment_rate':{
# '< 20':             ,
# '25 <= ... < 35':             ,
# '20 <= ... < 25':             ,
# '>= 35':
# },

# 'personal_status_sex':{
# 'male : married/widowed':             ,
# 'female : non-single or male : single':             ,
# 'female : single':             ,
# 'male : divorced/separated':
# },

# 'other_debtors':{
# 'none':             ,
# 'guarantor':             ,
# 'co-applicant':
# },

# 'present_residence':{
# '>= 7 yrs':             ,
# '1 <= ... < 4 yrs':             ,
# '4 <= ... < 7 yrs':             ,
# '< 1 yr':
# },

# 'property':{
# 'building soc. savings agr./life insurance':             ,
# 'unknown / no property':             ,
# 'car or other':             ,
# 'real estate':
# },

# 'other_installment_plans':{
# 'none':             ,
# 'bank':             ,
# 'stores':
# },

# 'housing':{
# 'rent':             ,
# 'for free':             ,
# 'own':
# },

# 'number_credits':{
# '1':             ,
# '2-3':             ,
# '4-5':             ,
# '>= 6':
# },

# 'job':{
# 'skilled employee/official':             ,
# 'unskilled - resident':             ,
# 'manager/self-empl./highly qualif. employee':             ,
# 'unemployed/unskilled - non-resident':
# },

# 'people_liable':{
# '0 to 2':             ,
# '3 or more':
# },

# 'telephone':{
# 'no':             ,
# 'yes (under customer name)':
# },

# 'foreign_worker':{
# 'no':             ,
# 'yes':
# }
# }
# col_dicts = {
#   'status': {'0<= ... < 200 DM': 0,
#   '... < 0 DM': 1,
#   '... >= 200 DM / salary for at least 1 year': 2,
#   'no checking account': 3},
#   'credit_history': {'critical account/other credits elsewhere': 0,
#   'delay in paying off in the past': 1,
#   'no credits taken/all credits paid back duly': 2,
#   'all credits at this bank paid back duly': 3,
#   'existing credits paid back duly till now': 4},
#   'purpose': {'business': 0,
#   'car (new)': 1,
#   'car (used)': 2,
#   'domestic appliances': 3,
#   'vacation': 4,
#   'furniture/equipment': 5,
#   'others': 6,
#   'radio/television': 7,
#   'repairs': 8,
#   'retraining': 9},
#   'savings': {'100 <= ... <  500 DM': 0,
#   '500 <= ... < 1000 DM': 1,
#   '... <  100 DM': 2,
#   '... >= 1000 DM': 3,
#   'unknown/no savings account': 4},
#   'employment_duration': {'< 1 yr': 0,
#   '1 <= ... < 4 yrs': 1,
#   '4 <= ... < 7 yrs': 2,
#   '>= 7 yrs': 3,
#   'unemployed': 4},
#   'installment_rate': {'< 20': 0,
#   '25 <= ... < 35': 1,
#   '20 <= ... < 25': 2,
#   '>= 35': 3},
#   'personal_status_sex': {'male : married/widowed': 0,
#   'female : non-single or male : single': 1,
#   'female : single': 2,
#   'male : divorced/separated': 3},
#   'other_debtors': {'co-applicant': 0, 
#   'guarantor': 1, 
#   'none': 2},
#   'present_residence': {'>= 7 yrs': 0,
#   '1 <= ... < 4 yrs': 1,
#   '4 <= ... < 7 yrs': 2,
#   '< 1 yr': 3},
#   'property': {'building soc. savings agr./life insurance': 0,
#   'car or other': 1,
#   'real estate': 2,
#   'unknown / no property': 3},
#   'other_installment_plans': {'bank': 0, 
#   'none': 1, 
#   'stores': 2},
#   'housing': {'for free': 0,
#    'own': 1, 'rent': 2},
#   'number_credits': {'1': 0,
#   '2-3': 1,
#   '4-5': 2,
#   '>= 6': 3},
#   'job': {'manager/self-empl./highly qualif. employee': 0,
#   'skilled employee/official': 1,
#   'unemployed/unskilled - non-resident': 2,
#   'unskilled - resident': 3},
#   'people_liable': {'0 to 2': 0,
#    '3 or more': 1},
#   'telephone': {'no': 0, 
#   'yes (under customer name)': 1},
#   'foreign_worker': {'no': 0,
#    'yes': 1}}

# for col in cols:
#     credit[col] = credit[col].map(col_dicts[col])
    
# y = credit['credit_risk']

# X  = credit.loc[:,'status':'foreign_worker']
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
# print(X_train)


