import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

for index,value in credit.status.value_counts():
    print(index)