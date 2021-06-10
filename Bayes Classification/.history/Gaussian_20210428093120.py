import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

values = df['country'].value_counts().keys().tolist()
counts = df['country'].value_counts().tolist()