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
#calculate the whole rate
for value in values:
    rate=list_status.value_counts()[value]/length
    whole_percent+=rate
    Gaussian_result[value]=whole_percent

# fig = plt.figure()
# ax = fig.add_subplot()


# for value in values:
#     x.append(value)
#     y.append(Gaussian_result[value])

# parameter = np.polyfit(x, y, 3)
# y2=[]

# for i in x:
#     y2_value = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + parameter[3]
#     y2.append(y2_value)

# plt.scatter(x, y)
# plt.plot(x, y2, color='g')
# ax.set_xlabel('value')
# ax.set_ylabel('counts_rate')
# plt.show()
x=[]
y=[]
fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot() 
for value in values:
    x.append(value)
    y.append(list_status.value_counts()[value])
    ax1.scatter(value, list_status.value_counts()[value])

ax1.set_xlabel('value')
ax1.set_ylabel('value_counts')
parameter = np.polyfit(x, y, 3)
y2=[]

for i in x:
    y2_value = parameter[0] * i ** 3 + parameter[1] * i ** 2 + parameter[2] * i + parameter[3]
    y2.append(y2_value)
plt.plot(x, y2, color='g')
plt.grid()

plt.show()

#fig = plt.figure(figsize = (10,6))
#ax1 = fig.add_subplot(2,1,1)  # 创建子图1
#ax1.scatter(s.index, s.values)
#plt.grid()
# 绘制数据分布图

#ax2 = fig.add_subplot(2,1,2)  # 创建子图2
#s.hist(bins=30,alpha = 0.5,ax = ax2)
#s.plot(kind = 'kde', secondary_y=True,ax = ax2)
#plt.grid()
