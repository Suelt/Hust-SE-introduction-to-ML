import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")
y = credit['credit_risk']

X = credit.loc[:,'status':'foreign_worker']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1,)
