import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly

credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")
df=pd.DataFrame(credit,columns=['duration','amount','age','credit_risk'])
print(df.head(5))