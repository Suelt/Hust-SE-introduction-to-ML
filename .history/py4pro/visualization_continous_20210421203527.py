import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly

credit = pd.read_csv("C:\\pyproject\\py4pro\\transformed.csv")
dict={}
dict['duration']=credit['duration']
print(dict)