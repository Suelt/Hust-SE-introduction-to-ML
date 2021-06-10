import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly

def calculatent(DataSet):
    risk_type = DataSet['credit_risk'].value_counts()
    m = DataSet.shape[0]
    p = risk_type/m
    ent = (-p*np.log2(p)).sum()
    return ent
    
credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")
df=pd.DataFrame(credit,columns=['duration','amount','age','credit_risk'])
