import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
credit = pd.read_csv("C:\\pyproject\\Bayes Classification\\transformed.csv")

df=credit[credit['credit_risk'].isin(['good'])]
list_status=df['duration']
length=list_status.shape[0]

# values = list_status.value_counts().keys().tolist()
# values.sort()
# Gaussian_result={}
# whole_percent=0
# rate=0
# #calculate the whole rate
# for value in values:
#     rate=list_status.value_counts()[value]/length
#     whole_percent+=rate
#     Gaussian_result[value]=whole_percent
# fig = plt.figure()
# ax = fig.add_subplot()
# x=[]
# y=[]
# for value in values:
#     x.append(value)
#     y.append(Gaussian_result[value])
# parameter = np.polyfit(x, y, 3)
# y2=[]

# for i in x:
#     y2_value = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + parameter[3]
#     y2.append(y2_value)
# print(x)
# plt.scatter(x, y)
# plt.plot(x, y2, color='g')
# # for value in values:
# #     ax.scatter(value, list_status.value_counts()[value])
# ax.set_xlabel('value')
# ax.set_ylabel('value_counts')
# plt.grid()
# plt.show()

# sns.set()
# x=list(df['duration'])
# sns.distplot(x)
# plt.show()

print(list_status)
arr_mean = np.mean(list_status)
arr_var = np.var(list_status)
arr_std = np.std(list_status,ddof=1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

r_2 = norm.pdf(list_status.value_counts, loc=arr_mean, scale=arr_var)
print(len(r_2))
print(r_2)