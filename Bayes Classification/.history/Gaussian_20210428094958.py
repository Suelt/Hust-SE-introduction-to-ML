import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

df=credit[credit['credit_risk'].isin(['good'])]
list_status=df['duration']
length=list_status.shape[0]

values = list_status.value_counts().keys().tolist()
values.sort()
Gaussian_result={}
whole_percent=0
rate=0
for value in values:
    rate=list_status.value_counts()[value]/length
    whole_percent+=rate
    Gaussian_result[value]=whole_percent

