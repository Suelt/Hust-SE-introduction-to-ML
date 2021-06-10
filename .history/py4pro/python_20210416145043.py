import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

y_pred=[1,1,2,3,5]
y_true=[1,3,2,5,5]
accuracy_num=0
for i in range(len(y_pred)):
    if y_pred[i]==y_true[i]:
        accuracy_num=accuracy_num+1
accuracy_rate=accuracy_num/len(y_pred)
print(accyracy_rate)

