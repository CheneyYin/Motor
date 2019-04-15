import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tsfresh.feature_extraction.feature_calculators as fc
import matplotlib.pyplot as plt
import warnings

train_path1 = '../Motor-Data/Motor_tain/N/00aab5a5-e096-4e4e-803f-a8525506cbd8_F.csv'
train_path1 = '../Motor-Data/Motor_tain/N/00aab5a5-e096-4e4e-803f-a8525506cbd8_B.csv'

df1 = pd.read_csv(train_path1, header = 0)
df2 = pd.read_csv(train_path2, header = 0)

df = pd.DataFrame(data = np.column_stack([df1['ai1'],df1['ai2'], df2['ai1'], df2['ai2'], range(79999), '1']), columns = ['F_ai1','F_ai2', 'B_ai1', 'B_ai2', 'time', 'id'])

