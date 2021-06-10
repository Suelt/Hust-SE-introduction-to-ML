import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly


credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")
df=pd.DataFrame(credit,columns=['duration','amount','age','credit_risk'])
col_dicts = {
'credit_risk':{
'good':0,
'bad':1,
}}
df['credit_risk'] = df['credit_risk'].map(col_dicts['credit_risk'])
print(df.corr('kendall'))
