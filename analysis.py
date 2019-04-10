import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings

train_path1 = 'train_N.csv'
train_path2 = 'train_P.csv'
df1 = pd.read_csv(train_path1, header = 0)
df2 = pd.read_csv(train_path2, header = 0)
x_idx = 8
y_idx = 9
plt.scatter(df1[df1.columns[x_idx]], df1[df1.columns[y_idx]], color='red', label = 'class N', marker = '^')
plt.scatter(df2[df2.columns[x_idx]], df2[df2.columns[y_idx]], color='blue', label = 'class P', marker = 's')

#plt.hist(data1, bins=150, color='blue', alpha=0.3)
#plt.hist(data2, bins=150, color='red', alpha=0.3)
plt.show()
