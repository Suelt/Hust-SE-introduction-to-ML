import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

df=credit[credit['credit_risk'].isin(['good'])]
list_status=df['duration']
print(list_status.shape[1])
print(df.head(5))
print(df.shape[0])