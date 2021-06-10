import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("./transformed.csv")
credit.head(5)
col_dicts = {}
cols = ['checking_balance','credit_history', 'purpose', 'savings_balance', 'employment_length', 'personal_status', 
        'other_debtors','property','installment_plan','housing','job','telephone','foreign_worker']

col_dicts = {'checking_balance': {'1 - 200 DM': 2,
  '< 0 DM': 1,
  '> 200 DM': 3,
  'unknown': 0},
 'credit_history': {'critical': 0,
  'delayed': 2,
  'fully repaid': 3,
  'fully repaid this bank': 4,
  'repaid': 1},
 'employment_length': {'0 - 1 yrs': 1,
  '1 - 4 yrs': 2,
  '4 - 7 yrs': 3,
  '> 7 yrs': 4,
  'unemployed': 0},
 'foreign_worker': {'no': 1, 'yes': 0},
 'housing': {'for free': 1, 'own': 0, 'rent': 2},
 'installment_plan': {'bank': 1, 'none': 0, 'stores': 2},
 'job': {'mangement self-employed': 3,
  'skilled employee': 2,
  'unemployed non-resident': 0,
  'unskilled resident': 1},
 'other_debtors': {'co-applicant': 2, 'guarantor': 1, 'none': 0},
 'personal_status': {'divorced male': 2,
  'female': 1,
  'married male': 3,
  'single male': 0},
 'property': {'building society savings': 1,
  'other': 3,
  'real estate': 0,
  'unknown/none': 2},
 'purpose': {'business': 5,
  'car (new)': 3,
  'car (used)': 4,
  'domestic appliances': 6,
  'education': 1,
  'furniture': 2,
  'others': 8,
  'radio/tv': 0,
  'repairs': 7,
  'retraining': 9},
 'savings_balance': {'101 - 500 DM': 2,
  '501 - 1000 DM': 3,
  '< 100 DM': 1,
  '> 1000 DM': 4,
  'unknown': 0},
 'telephone': {'none': 1, 'yes': 0}}

for col in cols:
    credit[col] = credit[col].map(col_dicts[col])
    
credit.head(5)