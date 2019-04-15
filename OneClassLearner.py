import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_validate
train_N_path = '../Motor-Data/train_N.csv'
train_P_path = '../Motor-Data/train_P.csv'
test_path = '../Motor-Data/test.csv'
sub_path = '../Motor-Data/sub.csv'

train_N_df = pd.read_csv(train_N_path, header = 0)
train_P_df = pd.read_csv(train_P_path, header = 0)
test_df = pd.read_csv(test_path, header = 0)
test_size = int(test_df[test_df.columns[0]].count())

train_df = pd.concat([train_N_df, train_P_df], axis = 0)
filter_features = [train_N_df.columns[idx] for idx in [2, 5, 8, 10, 11, 21, 22, 24, 29]]
scaler = MinMaxScaler()
oneclass_df = pd.DataFrame(data = scaler.fit_transform(train_P_df[train_P_df.columns[0:32]]), columns = train_P_df.columns[0:32])
class_df = pd.DataFrame(data = scaler.transform(train_N_df[train_N_df.columns[0:32]]), columns = train_N_df.columns[0:32])

ocsvm = OneClassSVM(\
            kernel='rbf', \
            degree=3, \
            gamma='auto', \
            coef0=0.0, \
            tol=0.00002, \
            nu=0.005, \
            shrinking=True, \
            cache_size=200, \
            verbose=True, \
            max_iter=-1, \
            random_state=None)

ocsvm.fit(oneclass_df[filter_features])
print ocsvm.predict(oneclass_df[filter_features])
print ocsvm.predict(class_df[filter_features])

