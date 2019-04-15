import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsfresh.feature_extraction.feature_calculators as fc
import matplotlib.pyplot as plt
import warnings

train_path1 = '../Motor-Data/Motor_tain/N/00aab5a5-e096-4e4e-803f-a8525506cbd8_F.csv'
train_path2 = '../Motor-Data/Motor_tain/P/00cb3f19-6216-429c-9537-973e60a863a1_F.csv'
df1 = pd.read_csv(train_path1, header = 0)
df2 = pd.read_csv(train_path2, header = 0)
data1 = df1[df1.columns[0]]
data2 = df2[df2.columns[0]]


#plt.hist(data1, bins=150, color='blue', alpha=0.3)
#plt.hist(data2, bins=150, color='red', alpha=0.3)
#plt.plot(range(79999),data1, color='blue')
plt.plot(range(79999),data2, color='red')
plt.plot(range(79999),data1, color='blue')

print fc.number_peaks(data1, 100)
print fc.number_peaks(data2, 100)

plt.show()

