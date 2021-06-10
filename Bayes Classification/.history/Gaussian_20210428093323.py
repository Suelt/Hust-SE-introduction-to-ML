import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

df=credit.query('credit_risk==good')