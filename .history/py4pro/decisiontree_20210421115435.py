import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")

col_dicts = {}
cols = ['status','credit_history', 'purpose', 'savings', 'employment_duration','installment_rate', 'personal_status_sex', 'other_debtors',
        'present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker']

col_dicts = {'status': {'0<= ... < 200 DM': 2,
  '... < 0 DM': 1,
  '... >= 200 DM / salary for at least 1 year': 3,
  'no checking account': 0},

  'credit_history': {'critical account/other credits elsewhere': 0,
  'delay in paying off in the past': 2,
  'no credits taken/all credits paid back duly': 3,
  'all credits at this bank paid back duly': 4,
  'existing credits paid back duly till now': 1},

  'purpose': {'business': 5,
  'car (new)': 3,
  'car (used)': 4,
  'domestic appliances': 6,
  'vacation': 1,
  'furniture/equipment': 2,
  'others': 8,
  'radio/television': 0,
  'repairs': 7,
  'retraining': 9},

  'savings': {'100 <= ... <  500 DM': 2,
  '500 <= ... < 1000 DM': 3,
  '... <  100 DM': 1,
  '... >= 1000 DM': 4,
  'unknown/no savings account': 0},

  'employment_duration': {'< 1 yr': 1,
  '1 <= ... < 4 yrs': 2,
  '4 <= ... < 7 yrs': 3,
  '>= 7 yrs': 4,
  'unemployed': 0},

  'installment_rate': {'< 20': 3,
  '25 <= ... < 35': 2,
  '20 <= ... < 25': 0,
  '>= 35': 1},

  'personal_status_sex': {'male : married/widowed': 2,
  'female : non-single or male : single': 1,
  'female : single': 3,
  'male : divorced/separated': 0},

  'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},

  'present_residence': {'>= 7 yrs': 1,
  '1 <= ... < 4 yrs': 3,
  '4 <= ... < 7 yrs': 0,
  '< 1 yr': 2},

  'property': {'building soc. savings agr./life insurance': 1,
  'car or other': 3,
  'real estate': 0,
  'unknown / no property': 2},

  'other_installment_plans': {'bank': 1, 'none': 0, 'stores': 2},

  'housing': {'for free': 1, 'own': 0, 'rent': 2},

  'number_credits': {'1': 3,
  '2-3': 2,
  '4-5': 0,
  '>= 6': 1},

  'job': {'manager/self-empl./highly qualif. employee': 3,
  'skilled employee/official': 2,
  'unemployed/unskilled - non-resident': 0,
  'unskilled - resident': 1},

  'people_liable': {'0 to 2': 1, '3 or more': 0},

  'telephone': {'no': 1, 'yes (under customer name)': 0},
  
  'foreign_worker': {'no': 1, 'yes': 0}}

for col in cols:
    credit[col] = credit[col].map(col_dicts[col])
    
credit.head(5)