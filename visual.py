from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsfresh.feature_extraction.feature_calculators as fc
import matplotlib.pyplot as plt
import warnings

# train_path1 = '../MotorData/Motor_tain/Negative/00aab5a5-e096-4e4e-803f-a8525506cbd8_F.csv'
# train_path2 = '../MotorData/Motor_tain/Positive/00cb3f19-6216-429c-9537-973e60a863a1_F.csv'
# df1 = pd.read_csv(train_path1, header = 0)
# df2 = pd.read_csv(train_path2, header = 0)
# data1 = df1[df1.columns[0]]
# data2 = df2[df2.columns[0]]
#
#
# plt.hist(data1, bins=150, color='blue', alpha=0.3)
# plt.hist(data2, bins=150, color='red', alpha=0.3)
# #plt.plot(range(79999),data1, color='blue')
# # plt.plot(range(79999),data2, color='red')
# # plt.plot(range(79999),data1, color='blue')
#
# print fc.number_peaks(data1, 100)
# print fc.number_peaks(data2, 100)
#
# plt.show()
#
train_N_path = '../MotorData/Features/train_N.csv'
train_P_path = '../MotorData/Features/train_P.csv'

train_N_df = pd.read_csv(train_N_path, header=0)
train_P_df = pd.read_csv(train_P_path, header=0)

plt.plot(range(30), train_P_df[train_P_df.columns[10]], color='r')
plt.plot(range(500), train_N_df[train_N_df.columns[10]], color='b')
plt.show()
